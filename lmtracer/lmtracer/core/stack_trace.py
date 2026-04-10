
import inspect
import re
from types import FrameType
from typing import Dict, Tuple, Union

import torch
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)

from lmtracer.core.probe_context import PhaseStackTraceState, ProbeContext, ProbeSource
from torch._inductor.virtualized import V

class FxNodeStackTraceInvoke:
    def __init__(self):
        self.filename: str = None
        self.lineno: int = None
        self.function: str = None
        self.line_code: str = None

        self.child_stack_by_filename: Dict[str, Dict[int, 'FxNodeStackTraceInvoke']] = {} # filename -> lineno -> FxNodeStackTraceInvoke

def get_scheduler_node_origin_fx_node(scheduler_node: BaseSchedulerNode) -> list[torch.fx.Node]:
    all_origin_nodes = set()
    for n in scheduler_node.get_nodes():
        direct_origin_fx_node = n.node.get_origin_node() # if result = run_node(fx_node), result IRNode is the direct origin of fx_node
        
        if direct_origin_fx_node is not None:
            # assert isinstance(direct_origin_fx_node, torch.fx.Node)
            if isinstance(direct_origin_fx_node, torch.fx.Node):
                all_origin_nodes.add(direct_origin_fx_node)
        
    return list(all_origin_nodes)

def extract_scheduler_node_stack_traces(scheduler_node: Union[FusedSchedulerNode, SchedulerNode]) -> Dict[Tuple[str, int], FxNodeStackTraceInvoke]:
    node_fusion = scheduler_node.get_nodes()
    if not node_fusion or len(node_fusion) == 0:
        raise ValueError("Scheduler node has no fused nodes.")
    stack_trace_roots: Dict[Tuple[str, int], FxNodeStackTraceInvoke] = {} # (filename, lineno) -> FxNodeStackTraceInvoke
    origin_fx_nodes = get_scheduler_node_origin_fx_node(scheduler_node)
    for origin_fx_node in origin_fx_nodes:
        if origin_fx_node is not None and origin_fx_node.stack_trace is not None:
            stack_trace_lines = origin_fx_node.stack_trace.splitlines()
            line_num = len(stack_trace_lines)
            is_root = True
            current_invoke: FxNodeStackTraceInvoke = None
            i = 0
            while i < line_num:
                if 'File "' not in stack_trace_lines[i]:
                    print(f"Skipping non-matching line: {stack_trace_lines[i]}")
                    i += 1
                    continue
                line = stack_trace_lines[i]
                i += 1
                match = re.search(r'File "([^"]+)", line (\d+), in ([^\s]+)', line)

                if match:
                    filename, lineno, func = match.groups()
                else:
                    print(f"Unmatched stack trace line: {line}")
                    i += 1
                    continue
                line = stack_trace_lines[i]
                line = line.strip()
                i += 1

                if is_root:
                    if (filename, int(lineno)) not in stack_trace_roots:
                        stack_trace_roots[(filename, int(lineno))] = FxNodeStackTraceInvoke()
                    current_invoke = stack_trace_roots[(filename, int(lineno))]
                    is_root = False
                else:
                    if filename not in current_invoke.child_stack_by_filename:
                        current_invoke.child_stack_by_filename[filename] = {}
                    if int(lineno) not in current_invoke.child_stack_by_filename[filename]:
                        current_invoke.child_stack_by_filename[filename][int(lineno)] = FxNodeStackTraceInvoke()
                    
                    current_invoke = current_invoke.child_stack_by_filename[filename][int(lineno)]
                
                current_invoke.filename = filename
                current_invoke.lineno = int(lineno)
                current_invoke.function = func
                current_invoke.line_code = line
        else:
            pass
                
    return stack_trace_roots

def compare_loop_stack_traces(trace1: Dict[str, Dict[int, 'FxNodeStackTraceInvoke']], trace2: Dict[str, Dict[int, 'FxNodeStackTraceInvoke']]) -> bool:
    if trace1 is None and trace2 is None:
        return True
    if trace1 is None or trace2 is None:
        return False
    for child_filename in trace1:
        if child_filename not in trace2:
            return False
        sorted_child_linenos1 = sorted(trace1[child_filename].keys())
        sorted_child_linenos2 = sorted(trace2[child_filename].keys())
        if sorted_child_linenos1 != sorted_child_linenos2:
            return False
        for child_lineno in sorted_child_linenos1:
            child_invoke1 = trace1[child_filename][child_lineno]
            child_invoke2 = trace2[child_filename][child_lineno]

            if child_invoke1.filename != child_invoke2.filename or \
               child_invoke1.lineno != child_invoke2.lineno or \
               child_invoke1.function != child_invoke2.function or \
               child_invoke1.line_code != child_invoke2.line_code:
                return False
            
    return True

def track_probe_stack_trace(stack_trace: FxNodeStackTraceInvoke, probe_source: ProbeSource, probe_phase_state: Dict[str, PhaseStackTraceState]) -> Tuple[bool, bool]:
    
    filename = probe_source.source_file
    lineno_start = probe_source.line_no_start
    lineno_end = probe_source.line_no_end
    opname = probe_source.phase_name

    if filename == stack_trace.filename and lineno_start <= stack_trace.lineno <= lineno_end:
        loop_back = False

        if stack_trace.lineno < probe_phase_state[opname].prev_line_no:
            if probe_source.loop_hint:
                for lineno in probe_phase_state[opname].child_stack_by_lineno:
                    child_stack = probe_phase_state[opname].child_stack_by_lineno[lineno]
                    if compare_loop_stack_traces(stack_trace.child_stack_by_filename, child_stack):
                        loop_back = True
                        break
        probe_phase_state[opname].prev_line_no = stack_trace.lineno
        return True, loop_back, stack_trace.child_stack_by_filename # hit the trace range
    for child_filename in stack_trace.child_stack_by_filename:
        sorted_child_linenos = sorted(stack_trace.child_stack_by_filename[child_filename].keys())
        for child_lineno in sorted_child_linenos:
            child_invoke = stack_trace.child_stack_by_filename[child_filename][child_lineno]
            result, loop_back, child_stack = track_probe_stack_trace(child_invoke, probe_source, probe_phase_state)
            if result:
                return True, loop_back, child_stack
    
    return False, False, None

def last_node_makeup(frame: FrameType, node: BaseSchedulerNode, probe_context: ProbeContext):
    if probe_context is None:
        return
    if probe_context.is_last_graph == False:
        return
    if not 'nodes' in frame.f_locals:
        print("Warning: 'nodes' not found in frame locals.")
        return
    nodes = frame.f_locals['nodes']
    if nodes.index(node) != len(nodes) - 1:
        return
    
    probe_phase_state = probe_context.get_probe_phase_state()
    for phase_name in probe_phase_state:
        phase_stack_trace_state = probe_phase_state[phase_name]
        print(phase_stack_trace_state.__dict__)
        if phase_stack_trace_state.entered:
            # need to exit this phase
            trace_point_id = probe_context.get_probe_source_by_phase_name(phase_name).probe_end_id
            V.graph.wrapper_code.writeline(f'timer_kernel.record(trace_buffer, current_buffer, doorbell_mapped, {trace_point_id}, {probe_context.config.num_flush}) # {phase_name} end')
            phase_stack_trace_state.entered = False
            phase_stack_trace_state.prev_line_no = -1
            phase_stack_trace_state.child_stack_by_lineno = {}
        

def update_stack_trace_state(node: BaseSchedulerNode, probe_context: ProbeContext):
    stack_traces = extract_scheduler_node_stack_traces(node)

    if probe_context.get_probe_phase_state() is None:
        probe_context.init_probe_phase_state()
    probe_phase_state = probe_context.get_probe_phase_state()
    
    exited = []
    entered = []

    for probe_source in probe_context.probe_sources:
        if not probe_source.inner_graph:
            continue
        phase_stack_trace_state = probe_phase_state[probe_source.phase_name]
        for fn, lineno in stack_traces:
            stack_trace_entry: FxNodeStackTraceInvoke = stack_traces[(fn, lineno)]
            prev_lineno = phase_stack_trace_state.prev_line_no
            probe_in_stack_trace, loop_back, hit_child_stack = track_probe_stack_trace(stack_trace_entry, probe_source, probe_phase_state)
            current_lineno = phase_stack_trace_state.prev_line_no

            if loop_back:
                if not phase_stack_trace_state.entered:
                    print(f'[lmtracer] Warning: loop back detected but phase {probe_source.phase_name} not marked as entered.')
                else:
                    # re-entering the same phase due to loop back
                    phase_stack_trace_state.prev_line_no = current_lineno
                    # record the stack trace of this loop back point
                    phase_stack_trace_state.child_stack_by_lineno = {}
                    phase_stack_trace_state.child_stack_by_lineno[current_lineno] = hit_child_stack

                    if probe_source.phase_name not in exited:
                        exited.append(probe_source.phase_name)
                    if probe_source.phase_name not in entered:
                        entered.append(probe_source.phase_name)
            elif not phase_stack_trace_state.entered and probe_in_stack_trace:
                # first time entering this phase
                phase_stack_trace_state.entered = True
                phase_stack_trace_state.prev_line_no = current_lineno
                phase_stack_trace_state.child_stack_by_lineno[current_lineno] = hit_child_stack

                if probe_source.phase_name not in entered:
                    entered.append(probe_source.phase_name)
            elif phase_stack_trace_state.entered and probe_in_stack_trace:
                # still inside this phase
                phase_stack_trace_state.prev_line_no = current_lineno
                phase_stack_trace_state.child_stack_by_lineno[current_lineno] = hit_child_stack
            elif phase_stack_trace_state.entered and not probe_in_stack_trace:
                # exited this phase
                # print(f'[lmtracer] Exited trace range {probe_source.phase_name} at {probe_source.source_file}:{probe_source.line_no_start}-{probe_source.line_no_end}')
                phase_stack_trace_state.entered = False
                phase_stack_trace_state.prev_line_no = -1
                phase_stack_trace_state.child_stack_by_lineno = {}

                if probe_source.phase_name not in exited:
                    exited.append(probe_source.phase_name)
            else:
                # still outside this phase
                pass
            
    for e in exited:
        trace_point_id = probe_context.get_probe_source_by_phase_name(e).probe_end_id
        V.graph.wrapper_code.writeline(f'timer_kernel.record(trace_buffer, current_buffer, doorbell_mapped, {trace_point_id}, {probe_context.config.num_flush}) # {e} end')
    for e in entered:
        trace_point_id = probe_context.get_probe_source_by_phase_name(e).probe_start_id
        V.graph.wrapper_code.writeline(f'timer_kernel.record(trace_buffer, current_buffer, doorbell_mapped, {trace_point_id}, {probe_context.config.num_flush}) # {e} start')
    
    last_node_makeup(inspect.currentframe().f_back.f_back, node, probe_context)