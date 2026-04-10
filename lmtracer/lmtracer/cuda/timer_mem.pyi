import torch

class MappedTensor:
    tensor: torch.Tensor
    device_ptr: int

def alloc_mapped_tensor(size: int, dtype: torch.dtype) -> MappedTensor:
    ...

def mem_size_by_num_flush(num_flush: int) -> int:
    ...