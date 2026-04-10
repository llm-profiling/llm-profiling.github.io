
import ctypes
import torch
from lmtracer.cuda import timer_mem
import numpy as np

class TraceBuffer:
    def __init__(self, size: int, gpu_time_offset_ns: int = 0):
        self.buffer_size = size
        self.device_trace_buffer = torch.zeros(self.buffer_size, dtype=torch.uint8, device="cuda")
        # dual buffer pointer, where zero indicates first half, one indicates second half
        self.device_write_buffer_index_tensor = torch.zeros(1, dtype=torch.bool, device="cuda")
        self.doorbell_mapped = timer_mem.alloc_mapped_tensor(1, torch.bool)
        self.doorbell_mapped_cpu_ptr = self.doorbell_mapped.tensor.data_ptr()
        self.doorbell_mapped_device_ptr = self.doorbell_mapped.device_ptr

        self.current_read_buffer_index = None
        self.after_warmup = False
        self.gpu_time_offset_ns = gpu_time_offset_ns

        self.copy_stream = torch.cuda.Stream()
        self.host_trace_buffer = torch.zeros(self.buffer_size, dtype=torch.uint8, device="cpu")

    def dump_trace_buffer(self, read_buffer_index):
        self.host_trace_buffer.copy_(self.device_trace_buffer, non_blocking=True)
        clock_values_original = self.host_trace_buffer.numpy()
        if read_buffer_index == 0:
            clock_values_original = clock_values_original[: self.buffer_size // 2]
        else:
            clock_values_original = clock_values_original[self.buffer_size // 2 : ]
        ptr_value_original = np.frombuffer(clock_values_original[0:8].tobytes(), dtype=np.uint64)
        clock_values_all = np.frombuffer(clock_values_original.tobytes(), dtype=np.uint64).copy()
        clock_values_all = clock_values_all.astype(np.int64) # fix out of range issue for newer numpy versions
        ptr_value = ptr_value_original[0]
        for i in range(0, ptr_value):
            clock_values_all[i * 2 + 2] += self.gpu_time_offset_ns
        clock_values_original = np.frombuffer(clock_values_all.tobytes(), dtype=np.uint8).copy()
        clock_values_original = clock_values_original[8: 8 + 16 * int(ptr_value)].tobytes()

        return clock_values_original
    
    def check_and_dump(self):
        new_value = ctypes.c_bool.from_address(self.doorbell_mapped_cpu_ptr).value
        if self.current_read_buffer_index != new_value:
            self.current_read_buffer_index = new_value
            with torch.cuda.stream(self.copy_stream):
                trace_data = self.dump_trace_buffer(self.current_read_buffer_index)
                return trace_data
        return None