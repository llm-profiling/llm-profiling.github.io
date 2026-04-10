#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include "layout.h"

MappedTensor alloc_mapped_tensor(int64_t size, torch::Dtype dtype) {
    void* host_ptr;
    cudaHostAlloc(&host_ptr, size, cudaHostAllocMapped);

    void* device_ptr;
    cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);

    MappedTensor mapped_tensor;
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    mapped_tensor.tensor = torch::from_blob(host_ptr, { (long)size }, options);
    mapped_tensor.device_ptr = reinterpret_cast<uint64_t>(device_ptr);

    return mapped_tensor;
}

int mem_size_by_num_flush(uint64_t num_flush) {
    return (sizeof(uint64_t) + sizeof(trace_point) * num_flush) * 2; // dual buffer
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    try {
        if (py::hasattr(m, "MappedTensor")) {
            // class already bound in this module instance -> still register functions if needed
        } else {
            py::class_<MappedTensor>(m, "MappedTensor")
                .def_readonly("tensor", &MappedTensor::tensor)
                .def_readonly("device_ptr", &MappedTensor::device_ptr);
        }
    } catch (const std::exception &e) {
        // Ignore errors in class registration
    }
    m.def("alloc_mapped_tensor", &alloc_mapped_tensor, "allocate a mapped tensor");
    m.def("mem_size_by_num_flush", &mem_size_by_num_flush, "get memory size by num_flush");
}

