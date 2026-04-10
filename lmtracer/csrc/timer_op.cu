#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include "layout.h"

extern "C" {
  PyObject* PyInit_timer_op(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "timer_op",
          NULL,
          -1,
          NULL,
      };
      return PyModule_Create(&module_def);
  }
}

namespace timer_op {

__global__ void record_timer_kernel(uint8_t* trace_buffer, bool* current_buffer, bool* door_bell, uint64_t tracepoint_id, uint64_t num_flush) {
    int layout_size = sizeof(uint64_t) + sizeof(trace_point) * num_flush;
    trace_layout* layout = (trace_layout*)(trace_buffer + (*current_buffer) * layout_size);
    layout->points[layout->ptr].tracepoint_id = tracepoint_id;
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

void record(torch::Tensor trace_buffer, torch::Tensor current_buffer, int64_t door_bell, int64_t tracepoint_id, int64_t num_flush) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    record_timer_kernel<<<1, 1, 0, stream>>>(
        trace_buffer.data_ptr<uint8_t>(),
        current_buffer.data_ptr<bool>(),
        reinterpret_cast<bool*>(door_bell),
        (uint64_t) tracepoint_id,
        (uint64_t) num_flush
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

void record_fake(torch::Tensor trace_buffer, torch::Tensor current_buffer, int64_t door_bell, int64_t tracepoint_id, int64_t num_flush) {
    printf(">>> FAKE/CPU record called\n");
    return;
}

TORCH_LIBRARY(timer_op, m) {
    m.def("record(Tensor trace_buffer, Tensor current_buffer, int door_bell, int tracepoint_id, int num_flush) -> ()");
}

TORCH_LIBRARY_IMPL(timer_op, Meta, m) {
    m.impl("record", &record_fake);
}

TORCH_LIBRARY_IMPL(timer_op, Fake, m) {
    m.impl("record", &record_fake);
}

TORCH_LIBRARY_IMPL(timer_op, CPU, m) {
    m.impl("record", &record_fake);
}

TORCH_LIBRARY_IMPL(timer_op, CUDA, m) {
  m.impl("record", &record);
}

}
