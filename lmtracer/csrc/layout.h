#include <torch/extension.h>

struct trace_point {
    uint64_t  tracepoint_id;
    uint64_t  clock;
    // uint64_t stream_id;
};

// byte alignment with the maximum alignment requirement (i.e., uint64_t clock)
struct trace_layout {
    uint64_t  ptr;
    trace_point points[];
};

struct MappedTensor {
    torch::Tensor tensor;   // CPU tensor (host_ptr)
    uint64_t device_ptr;       // GPU-visible address, which is converted from void*
};