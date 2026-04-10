
#include <cuda_runtime.h>

__global__ void record_timer_kernel(uint8_t* trace_buffer, bool* current_buffer, bool* door_bell, uint64_t tracepoint_id, uint64_t num_flush);
