#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__global__ void read_globaltimer_kernel(unsigned long long* out) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    out[0] = t;
}

torch::Tensor read_reg() {
    auto out = torch::empty({1}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    uint64_t* ptr = out.data_ptr<uint64_t>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    read_globaltimer_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<unsigned long long*>(ptr));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read_reg", &read_reg, "read globaltimer register on GPU");
}
