import bisect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, TypeAlias, Union
import re
from sympy import Expr
from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.triton import TritonScheduling
from dataclasses import dataclass

from torch._inductor.scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)
from torch._inductor.virtualized import V
import torch
import logging

from torch.utils._ordered_set import OrderedSet
from torch._inductor.codegen.simd_kernel_features import NodeScheduleEntry, DisableReduction, EnableReduction
from torch._inductor.codegen.common import BackendFeature
from torch._inductor.codegen.simd import SIMDKernel
from torch._inductor.virtualized import V
import inspect

from lmtracer.core.probe_context import get_current_probe_context
from lmtracer.core.stack_trace import FxNodeStackTraceInvoke, extract_scheduler_node_stack_traces, get_scheduler_node_origin_fx_node, track_probe_stack_trace, update_stack_trace_state

log = logging.getLogger(__name__)

_IntLike: TypeAlias = Union[int, Expr]

extra_attrs = {}
def set_attr(obj, name, value):
    key = (id(obj), name)
    d = extra_attrs.setdefault(key, {})
    d[name] = value

def get_attr(obj, name, default=None):
    key = (id(obj), name)
    return extra_attrs.get(key, {}).get(name, default)


@dataclass
class FxStackTrace:
    filename: str
    lineno: int
    function: str
    line: str

@dataclass
class CUDAKernelTraceback:
    kernel_name: str
    origin_fx_nodes: list[torch.fx.Node]
    stack_traces: list[FxStackTrace]

collected_kernel_tracebacks: Dict[str, CUDAKernelTraceback] = {}

def generate_kernel_traceback(kernel_name: str, all_origin_fx_nodes: list[torch.fx.Node]) -> CUDAKernelTraceback:
    stack_traces = []
    stack_trace_pattern = r'File "([^"]+)", line (\d+), in ([^\s]+)'
    for fx_node in all_origin_fx_nodes:
        if fx_node.stack_trace is not None:
            stack_trace_lines = fx_node.stack_trace.splitlines()
            line_num = len(stack_trace_lines)
            i = 0
            while i < line_num:
                if 'File "' not in stack_trace_lines[i]:
                    print(f"Skipping non-matching line: {stack_trace_lines[i]}")
                    i += 1
                    continue
                line = stack_trace_lines[i]
                i += 1
                match = re.search(stack_trace_pattern, line)

                if match:
                    filename, lineno, func = match.groups()
                    # print(f"Matched stack trace: {filename}, {lineno}, {func}")
                else:
                    print(f"Unmatched stack trace line: {line}")
                    i += 1
                    continue
                line = stack_trace_lines[i]
                line = line.strip()
                i += 1
                stack_traces.append(FxStackTrace(
                    filename=filename,
                    lineno=int(lineno),
                    function=func,
                    line=line
                ))
        else:
            # print(f"FX node {fx_node} has no stack trace")
            pass
    return CUDAKernelTraceback(
        kernel_name=kernel_name,
        origin_fx_nodes=all_origin_fx_nodes,
        stack_traces=stack_traces
    )

class lmtracerTritonScheduling(TritonScheduling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def define_kernel(self, src_code, node_schedule: list[NodeScheduleEntry], kernel: SIMDKernel):
        # print(f"[lmtracerTritonScheduling] Generating code for node schedule with kernel: {kernel.kernel_name}, node_schedule: {node_schedule}")
        kernel_name = super().define_kernel(src_code, node_schedule, kernel)

        all_origin_fx_nodes = []
        for scheduler_node in node_schedule:
            if scheduler_node is DisableReduction or scheduler_node is EnableReduction:
                continue
            origin_fx_nodes = get_scheduler_node_origin_fx_node(scheduler_node)
            all_origin_fx_nodes.extend(origin_fx_nodes)
        # print(f"[lmtracerTritonScheduling] Origin FX nodes for kernel {kernel_name}: {all_origin_fx_nodes}")
        kernel_traceback = generate_kernel_traceback(kernel_name, all_origin_fx_nodes)
        # print(f"[lmtracerTritonScheduling] Kernel Traceback for kernel {kernel_name}: {kernel_traceback}")
        collected_kernel_tracebacks[kernel_name] = kernel_traceback
        # with open(f"/tmp/lmtracer_triton_kernel_{kernel_name}.txt", "w") as f:
        #     f.write(f"Kernel Name: {kernel_name}\n\n")
        #     for fx_node in all_origin_fx_nodes:
        #         f.write(f"FX Node: {fx_node.format_node()}\n")
        #         f.write(f"Stack trace:\n{fx_node.stack_trace}\n")
        #         f.write("\n")
        return kernel_name

def extract_paths(node: FxNodeStackTraceInvoke, path=None) -> List[List[Tuple[str, int]]]:
    if path is None:
        path = []
    current = path + [(node.filename, node.lineno)]
    if not node.child_stack_by_filename:
        return [current]
    paths = []
    for fname, children in node.child_stack_by_filename.items():
        for _, child in children.items():
            paths.extend(extract_paths(child, current))
    return paths

def common_prefix(path1, path2):
    prefix = []
    for (f1, l1), (f2, l2) in zip(path1, path2):
        if f1 == f2 and l1 == l2:
            prefix.append((f1, l1))
        else:
            break
    return prefix


def is_between_stacks(prev_root: FxNodeStackTraceInvoke,
                      curr_root: FxNodeStackTraceInvoke,
                      filename: str,
                      lineno: int) -> bool:
    prev_paths = extract_paths(prev_root)
    curr_paths = extract_paths(curr_root)

    for prev_path in prev_paths:
        for curr_path in curr_paths:
            prefix = common_prefix(prev_path, curr_path)
            prev_diff = prev_path[len(prefix):]
            curr_diff = curr_path[len(prefix):]

            if not prev_diff or not curr_diff:
                continue

            prev_last_file, prev_last_line = prev_diff[-1]
            curr_first_file, curr_first_line = curr_diff[0]

            if filename == prev_last_file == curr_first_file:
                if prev_last_line < lineno < curr_first_line:
                    return True

            if filename == prev_last_file and lineno > prev_last_line:
                return True
            if filename == curr_first_file and lineno < curr_first_line:
                return True

    return False

def print_stack_trace_entry(stack_trace_entry: FxNodeStackTraceInvoke, depth=0):
    indent = '  ' * (depth + 1)
    print(f'{indent}File "{stack_trace_entry.filename}", line {stack_trace_entry.lineno}, in {stack_trace_entry.function}')
    print(f'{indent}  {stack_trace_entry.line_code}')
    for child_filename in stack_trace_entry.child_stack_by_filename:
        sorted_child_linenos = sorted(stack_trace_entry.child_stack_by_filename[child_filename].keys())
        for child_lineno in sorted_child_linenos:
            child_invoke = stack_trace_entry.child_stack_by_filename[child_filename][child_lineno]
            print_stack_trace_entry(child_invoke, depth + 1)

def check_stack_switch_between_scheduler_nodes(
    prev_scheduler_node: Union[FusedSchedulerNode, SchedulerNode],
    curr_scheduler_node: Union[FusedSchedulerNode, SchedulerNode]
) -> bool:
    if prev_scheduler_node is None:
        return False

    prev_stack_traces = extract_scheduler_node_stack_traces(prev_scheduler_node)
    curr_stack_traces = extract_scheduler_node_stack_traces(curr_scheduler_node)


prev_scheduled_nodes = []

class lmtracerCUDACombinedScheduling(CUDACombinedScheduling):
    def __init__(self, scheduler: Optional[Scheduler]) -> None:
        super().__init__(scheduler)
        self._triton_scheduling = lmtracerTritonScheduling(scheduler)
        self._model_context_stack = []

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        return super().get_backend_features(device)

    def choose_node_backend(self, node: BaseSchedulerNode) -> BaseScheduling:
        return super().choose_node_backend(node)

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        return super().can_fuse_vertical(node1, node2)

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        return super().can_fuse_horizontal(node1, node2)

    def group_fn(
        self, sizes: Sequence[Sequence[_IntLike]]
    ) -> tuple[tuple[_IntLike, ...], ...]:
        return super().group_fn(sizes)

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> Optional[str]:
        
        return super().codegen_template(template_node, epilogue_nodes, prologue_nodes)

    def codegen_node(self, node: Union[FusedSchedulerNode, SchedulerNode]) -> None:

        prev_node = prev_scheduled_nodes[-1] if len(prev_scheduled_nodes) > 0 else None
        check_stack_switch_between_scheduler_nodes(
            prev_node,
            node
        )

        prev_scheduled_nodes.append(node)

        update_stack_trace_state(node, get_current_probe_context())
        super().codegen_node(node)

    def codegen_sync(self) -> None:
        return super().codegen_sync()

    def flush(self) -> None:
        return super().flush()

    def codegen_combo_kernel(self, *args: Any, **kwargs: Any) -> None:
        return super().codegen_combo_kernel(*args, **kwargs)

    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[float, str]:
        return super().benchmark_fused_nodes(nodes)

    def benchmark_codegened_module(self, module):
        return super().benchmark_codegened_module(module)

    def generate_kernel_code_from_nodes(
        self, nodes: Sequence[Any], benchmark_kernel: bool = False
    ) -> str:
        return super().generate_kernel_code_from_nodes(
            nodes, benchmark_kernel
        )

    def benchmark_combo_kernel(
        self, node_list: Sequence[BaseSchedulerNode]
    ) -> tuple[float, float, list[Optional[str]]]:
        return super().benchmark_combo_kernel(node_list)

    