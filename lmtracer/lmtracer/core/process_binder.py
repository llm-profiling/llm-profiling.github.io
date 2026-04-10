
from lmtracer.core.logger import lmtracer_log
from lmtracer.core.probe_context import get_current_probe_context, init_probe_context
from lmtracer.core.time_sync import sync_worker_gpu_time
from lmtracer.cuda.timer_mem import mem_size_by_num_flush
from lmtracer.core.trace_buffer import TraceBuffer
from concurrent.futures import ThreadPoolExecutor
import threading

class ProcessBinder:
    def __init__(self, llm_engine: str, model_name: str = None, metric_labels: dict = None):
        import lmtracer.plugins.torch._inductor
        self.probe_context = init_probe_context(llm_engine, model_name)
        self.metric_labels = metric_labels
        self.allocated_gpu_memory_bytes = mem_size_by_num_flush(get_current_probe_context().config.num_flush)
        self.gpu_time_offset_ns = sync_worker_gpu_time()
        lmtracer_log(f"Allocated GPU memory for lmtracer: {self.allocated_gpu_memory_bytes / (1024 * 1024):.2f} MB, GPU time offset to master rank: {self.gpu_time_offset_ns} ns")
        self.trace_buffer = TraceBuffer(size=self.allocated_gpu_memory_bytes, gpu_time_offset_ns=self.gpu_time_offset_ns)
        
        self.probe_context.update_trace_buffer_bindings(self.trace_buffer, self.probe_context.config.num_flush)
        self.trace_exporters = get_current_probe_context().config.exporters
        
        # Create thread pool for async export
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="lmtracer_exporter")

    def after_execution(self):
        trace_data = self.trace_buffer.check_and_dump()

        if trace_data is None:
            return

        context = get_current_probe_context()
        
        # Export to all configured exporters asynchronously
        for idx, exporter in enumerate(self.trace_exporters):
            self.executor.submit(self._export_async, idx, exporter, trace_data, context.probe_sources)
    
    def _export_async(self, idx, exporter, trace_data, probe_sources):
        try:
            exporter.export_trace_data(trace_data, probe_sources, self.metric_labels)
        except Exception as e:
            lmtracer_log(f"Error exporting to exporter {idx+1} ({exporter.__class__.__name__}): {e}")
    
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)