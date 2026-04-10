
import functools
import inspect
from typing import Callable, Dict, Optional, TypeVar, Union
import torch.nn as nn
import re
from dataclasses import dataclass
from torch.utils.cpp_extension import load
import torch
import lmtracer.cuda.timer_op

_T = TypeVar("_T", bound=type[nn.Module])

def module_probing(
    cls: Optional[_T] = None,
) -> Union[Callable[[_T], _T], _T]:
    
    def _call_with_torch_compile(original_call) -> Callable:
        def __call__(self, *args, **kwargs):
            if self.module_bindings is None:
                print("[lmtracer] module_bindings is None, skipping tracing.")
                return original_call(self, *args, **kwargs)
            
            trace_buffer = self.module_bindings.trace_buffer
            current_buffer = self.module_bindings.current_buffer
            num_flush = self.module_bindings.num_flush
            doorbell_mapped_device_ptr = self.module_bindings.doorbell_mapped_device_ptr
            probe_id_before = self.module_bindings.probe_id_before
            probe_id_after = self.module_bindings.probe_id_after

            if torch.compiler.is_compiling():
                return original_call(self, *args, **kwargs)

            torch.ops.timer_op.record(trace_buffer, current_buffer, doorbell_mapped_device_ptr, probe_id_before, num_flush)
            model_output = original_call(self, *args, **kwargs)
            torch.ops.timer_op.record(trace_buffer, current_buffer, doorbell_mapped_device_ptr, probe_id_after, num_flush)
            return model_output
        return __call__
    
    def __setattr__(self, name, value):
        if name == "__call__":
            value = _call_with_torch_compile(value)
        super(cls, self).__setattr__(name, value)

    original_call = cls.__call__
    cls.__call__ = _call_with_torch_compile(original_call)
    cls.__setattr__ = __setattr__

    return cls

