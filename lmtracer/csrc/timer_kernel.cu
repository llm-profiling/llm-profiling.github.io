#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint> 
#include "layout.h"
#include "timer_kernel.cuh"

__global__ void record_timer_kernel(uint8_t* trace_buffer, bool* current_buffer, bool* door_bell, uint64_t tracepoint_id, uint64_t num_flush) {
    int layout_size = sizeof(uint64_t) + sizeof(trace_point) * num_flush;
    trace_layout* layout = (trace_layout*)(trace_buffer + (*current_buffer) * layout_size);
    layout->points[layout->ptr].tracepoint_id = tracepoint_id;
    // layout->points[layout->ptr].stream_id = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(layout->points[layout->ptr].clock));
    layout->ptr += 1;
    if (layout->ptr >= num_flush) {
        *door_bell = *current_buffer;
        *current_buffer = !(*current_buffer);
        layout = (trace_layout*)(trace_buffer + (*current_buffer) * layout_size);
        layout->ptr = 0;
        __threadfence_system();
    }
}

void record(torch::Tensor trace_buffer, torch::Tensor current_buffer, uint64_t door_bell, uint64_t tracepoint_id, uint64_t num_flush) {

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    record_timer_kernel<<<1, 1, 0, stream>>>(trace_buffer.data_ptr<uint8_t>(), current_buffer.data_ptr<bool>(),
        reinterpret_cast<bool*>(door_bell), tracepoint_id, num_flush);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("record", &record, "read globaltimer register and write the value into the preallocated trace buffer");
}
