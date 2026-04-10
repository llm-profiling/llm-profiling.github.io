import inspect
import traceback
from typing import Any, Callable, Iterable
from unittest.mock import patch
import weakref


from torch.fx.node import Argument, Target
import torch
import torch._inductor.graph
import torch._inductor.scheduler
from torch._inductor.scheduler import BaseSchedulerNode, ExternKernelSchedulerNode
import torch.fx._symbolic_trace

from torch._inductor.ir import IRNode, Operation
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
import torch._inductor.codegen.common
from torch._inductor.virtualized import V

from lmtracer.core.probe_context import get_current_probe_context
from lmtracer.core.stack_trace import last_node_makeup, update_stack_trace_state
from lmtracer.plugins.torch._inductor.cuda_combined_scheduling import FxNodeStackTraceInvoke, lmtracerCUDACombinedScheduling, get_scheduler_node_origin_fx_node, set_attr, track_probe_stack_trace
from lmtracer.plugins.torch._inductor.wrapper import LMTracerPythonWrapperCodegen
from types import ModuleType
from lmtracer.plugins.torch._inductor.cuda_combined_scheduling import generate_kernel_traceback, prev_scheduled_nodes, check_stack_switch_between_scheduler_nodes

from typing import TYPE_CHECKING, Any, Callable, Optional
from torch._inductor.codegen.common import device_codegens, DeviceCodegen

from unittest.mock import patch

from torch._inductor.codegen.common import register_backend_for_device, init_backend_registration

from lmtracer.plugins.torch._inductor.cuda_combined_scheduling import lmtracerCUDACombinedScheduling
import torch._inductor.config as inductor_config

from lmtracer.plugins.torch._inductor.wrapper import LMTracerPythonWrapperCodegen
from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
from torch._inductor.codegen.halide import HalideScheduling
import torch._inductor.compile_fx
import torch._dynamo.backends.common
from torch._dynamo.backends.common import AotAutograd
import torch._functorch.aot_autograd
# import torch._functorch._aot_autograd.graph_compile
from torch._inductor.compile_fx import fx_compile_mode

WrapperConstructor = type[PythonWrapperCodegen]

init_backend_registration()
cuda_backends = {
    "triton": lmtracerCUDACombinedScheduling,
    "halide": HalideScheduling,
}
register_backend_for_device(
    "cuda",
    lambda scheduling: cuda_backends[inductor_config.cuda_backend](scheduling),
    LMTracerPythonWrapperCodegen,
    CppWrapperGpu,
)

original_call_method = torch.fx.Interpreter.call_method
original_run_node = torch._inductor.graph.GraphLowering.run_node
original_compile_to_module = torch._inductor.graph.GraphLowering._compile_to_module
original_call_module = torch.fx._symbolic_trace.Tracer.call_module
original_code_gen = torch._inductor.scheduler.Scheduler._codegen
original_codegen_extern_call = torch._inductor.scheduler.Scheduler.codegen_extern_call
original_get_wrapper_codegen_for_device = torch._inductor.codegen.common.get_wrapper_codegen_for_device
original_compile_fx_inner = torch._inductor.compile_fx._compile_fx_inner
original_compile_fx = torch._inductor.compile_fx.compile_fx
original_aot_autograd = torch._dynamo.backends.common.aot_autograd
original_aot_autograd_call = AotAutograd.__call__
original_SerializableAOTDispatchCompiler_init = torch._functorch.aot_autograd.SerializableAOTDispatchCompiler.__init__
original_SerializableAOTDispatchCompiler_call = torch._functorch.aot_autograd.SerializableAOTDispatchCompiler.__call__
original_fx_codegen_and_compile = torch._inductor.compile_fx.fx_codegen_and_compile

original_torch_compile = torch.compile

def lmtracer_torch_compile(*args, **kwargs):
    # set_in_torch_compile_context(True)
    ret = original_torch_compile(*args, **kwargs)
    # set_in_torch_compile_context(False)
    return ret

torch.compile = lmtracer_torch_compile

def get_wrapper_codegen_for_device(
    device: str, cpp_wrapper: bool = False
) -> Optional[WrapperConstructor]:
    if device in device_codegens:
        wrapper_codegen_obj: DeviceCodegen = device_codegens[device]
        return (
            wrapper_codegen_obj.cpp_wrapper_codegen
            if cpp_wrapper
            else wrapper_codegen_obj.wrapper_codegen
        )
    return None

def lmtracer_call_method(
    self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
) -> Any:
    ret = original_call_method(self, target, args, kwargs)
    return ret



def lmtracer_run_node(self, n: torch.fx.Node) -> object:
    obj = original_run_node(self, n)

    if isinstance(obj, IRNode):
        defining_op = obj.get_defining_op()
        if isinstance(defining_op, Operation):
            origin_ir_node = defining_op.get_origin_node()
        else:
            pass
    else:
        pass
    return obj

def lmtracer_compile_to_module(self) -> ModuleType:
    mod = original_compile_to_module(self)
    return mod

def lmtracer_codegen(self, nodes: list[BaseSchedulerNode]) -> None:
    return original_code_gen(self, nodes)


def lmtracer_codegen_extern_call(self, scheduler_node: ExternKernelSchedulerNode) -> None:
    prev_node = prev_scheduled_nodes[-1] if len(prev_scheduled_nodes) > 0 else None
    check_stack_switch_between_scheduler_nodes(
        prev_node,
        scheduler_node
    )

    prev_scheduled_nodes.append(scheduler_node)
    update_stack_trace_state(scheduler_node, get_current_probe_context())

    ret = original_codegen_extern_call(self, scheduler_node)

    last_node_makeup(inspect.currentframe().f_back, scheduler_node, get_current_probe_context())
    return ret

torch._inductor.graph.GraphLowering.run_node = lmtracer_run_node
torch._inductor.graph.GraphLowering._compile_to_module = lmtracer_compile_to_module
torch._inductor.scheduler.Scheduler._codegen = lmtracer_codegen
torch._inductor.scheduler.Scheduler.codegen_extern_call = lmtracer_codegen_extern_call

torch._inductor.codegen.common.get_wrapper_codegen_for_device = get_wrapper_codegen_for_device


import torch._inductor.runtime.compile_tasks
_original_reload_python_module = torch._inductor.runtime.compile_tasks._reload_python_module

def reload_python_module(*args, **kwargs) -> ModuleType:
    mod = _original_reload_python_module(*args, **kwargs)
    setattr(mod, "trace_buffer", get_current_probe_context().trace_buffer.device_trace_buffer)
    setattr(mod, "current_buffer", get_current_probe_context().trace_buffer.device_write_buffer_index_tensor)
    setattr(mod, "doorbell_mapped", get_current_probe_context().trace_buffer.doorbell_mapped_device_ptr)
    return mod

patcher = patch("torch._inductor.runtime.compile_tasks._reload_python_module", reload_python_module)
patcher.start()
patcher = patch("torch._inductor.codecache._reload_python_module", reload_python_module)
patcher.start()
