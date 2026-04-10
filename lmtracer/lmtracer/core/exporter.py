import ctypes
import platform
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os

from lmtracer.core.probe_context import ProbeSource
from lmtracer.core.dump_parser import DumpParser, BinaryDumpParser

class DataExporter(ABC):
    @abstractmethod
    def export_trace_data(self, data: Any, probe_sources: List[ProbeSource] = None, metric_labels: dict = None) -> None:
        pass
    
    @abstractmethod
    def export_meta_data(self, meta: Any) -> None:
        pass
class ByteDataExporter(DataExporter):
    @abstractmethod
    def export_trace_data(self, byte_data: bytes, probe_sources: List[ProbeSource] = None, metric_labels: dict = None) -> None:
        pass
    
    @abstractmethod
    def export_meta_data(self, meta: Any) -> None:
        pass

class FileByteDataExporter(ByteDataExporter):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.trace_writer = open(f'{output_dir}/trace_data_{platform.node()}_{os.getpid()}.bin', 'wb')

    def export_trace_data(self, byte_data: bytes, probe_sources: List[ProbeSource] = None, metric_labels: dict = None) -> None:
        if byte_data is not None:
            self.trace_writer.write(byte_data)
            self.trace_writer.flush()
        
        if probe_sources is not None:
            import json
            data = {}
            for probe_source in probe_sources:
                data[f'{probe_source.phase_name}_start'] = probe_source.probe_start_id
                data[f'{probe_source.phase_name}_end'] = probe_source.probe_end_id
            self.export_meta_data(data)
    
    def export_meta_data(self, meta: Any) -> None:
        import json
        self.meta_writer = open(f'{self.output_dir}/metadata_{platform.node()}_{os.getpid()}.json', 'w')
        self.meta_writer.write(json.dumps(meta, indent=2))
        self.meta_writer.flush()
        self.meta_writer.close()

    def __del__(self):
        self.trace_writer.flush()
        self.trace_writer.close()

class SGLangMetricPrometheusExporter(DataExporter):
    def __init__(self):
        from lmtracer.core.metrics import GPUBubbleRateMetric, ExecutionTimeStats
        import threading
        
        self.parser = BinaryDumpParser()
        self.bubble_rate_metric = GPUBubbleRateMetric()
        self.execution_time_metrics: Dict[str, ExecutionTimeStats] = {}  # Dict[str, ExecutionTimeStats] for each phase_name
        self._metrics_lock = threading.Lock()  # Lock for thread-safe access to metrics
        
        # Get GPU information
        self.torch_global_rank = self._get_torch_global_rank()
        self.gpu_device_index = self._get_gpu_device_index()
        self.hostname = platform.node()
        self.pid = os.getpid()
        
        # Gauges will be lazily initialized in export_trace_data
        self.gpu_bubble_rate_gauge = None
        self.model_execution_time_gauge = None
        self.model_execution_time_median_gauge = None
        self.model_execution_time_p95_gauge = None
        self.model_execution_time_p99_gauge = None
        self.model_execution_time_std_gauge = None
        self.model_execution_time_min_gauge = None
        self.model_execution_time_max_gauge = None
        self._gauge_labels = None
        
        # Start background thread to periodically update metrics
        self._stop_event = threading.Event()
        self._update_thread = threading.Thread(target=self._periodic_metrics_update, daemon=True)
        self._update_thread.start()
    
    def _get_torch_global_rank(self) -> str:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return str(dist.get_rank())
        except ImportError:
            pass
        
        return '-1'
    
    def _get_gpu_device_index(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                # Get current CUDA device
                device_index = torch.cuda.current_device()
                return str(device_index)
        except ImportError:
            pass
        
        return '-1'

    def _periodic_metrics_update(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(1)  # Check every 1 second
            self._update_execution_time_metrics()
    
    def _update_execution_time_metrics(self) -> None:
        if self._gauge_labels is None or self.model_execution_time_gauge is None:
            return
        
        with self._metrics_lock:
            for phase_name, metric in self.execution_time_metrics.items():
                # Check if we should flush this metric
                if metric.should_flush():
                    # Get current metrics
                    metrics = metric.get_metrics()
                    labels = {**self._gauge_labels, 'phase_name': phase_name}
                    
                    # Update prometheus gauges
                    self.model_execution_time_gauge.labels(**labels).set(metrics['avg'])
                    self.model_execution_time_median_gauge.labels(**labels).set(metrics['median'])
                    self.model_execution_time_p95_gauge.labels(**labels).set(metrics['p95'])
                    self.model_execution_time_p99_gauge.labels(**labels).set(metrics['p99'])
                    self.model_execution_time_std_gauge.labels(**labels).set(metrics['std'])
                    self.model_execution_time_min_gauge.labels(**labels).set(metrics['min'])
                    self.model_execution_time_max_gauge.labels(**labels).set(metrics['max'])

    def export_trace_data(self, data: Any, probe_sources: List['ProbeSource'] = None, metric_labels: dict = None) -> None:
        if data is not None and self.parser is not None:
            # Merge default labels with provided metric_labels
            labels = {
                'torch_global_rank': self.torch_global_rank,
                'gpu_device_index': self.gpu_device_index,
                'hostname': self.hostname,
                'pid': str(self.pid)
            }
            if metric_labels:
                labels.update(metric_labels)
            
            # Lazily initialize gauges with merged labels
            if self.gpu_bubble_rate_gauge is None or self._gauge_labels != labels:
                from prometheus_client import Gauge
                self._gauge_labels = labels
                label_keys = list(labels.keys())
                
                self.gpu_bubble_rate_gauge = Gauge(
                    'lmtracer:gpu_bubble_rate', 
                    'GPU bubble rate measured by forward duration',
                    label_keys
                )
                self.model_execution_time_gauge = Gauge(
                    'lmtracer:model_execution_time_ns',
                    'Average model execution time in nanoseconds for different operations',
                    label_keys + ['phase_name']
                )
                self.model_execution_time_median_gauge = Gauge(
                    'lmtracer:model_execution_time_median_ns',
                    'Median model execution time in nanoseconds for different operations',
                    label_keys + ['phase_name']
                )
                self.model_execution_time_p95_gauge = Gauge(
                    'lmtracer:model_execution_time_p95_ns',
                    'P95 model execution time in nanoseconds for different operations',
                    label_keys + ['phase_name']
                )
                self.model_execution_time_p99_gauge = Gauge(
                    'lmtracer:model_execution_time_p99_ns',
                    'P99 model execution time in nanoseconds for different operations',
                    label_keys + ['phase_name']
                )
                self.model_execution_time_std_gauge = Gauge(
                    'lmtracer:model_execution_time_std_ns',
                    'Standard deviation of model execution time in nanoseconds for different operations',
                    label_keys + ['phase_name']
                )
                self.model_execution_time_min_gauge = Gauge(
                    'lmtracer:model_execution_time_min_ns',
                    'Minimum model execution time in nanoseconds for different operations',
                    label_keys + ['phase_name']
                )
                self.model_execution_time_max_gauge = Gauge(
                    'lmtracer:model_execution_time_max_ns',
                    'Maximum model execution time in nanoseconds for different operations',
                    label_keys + ['phase_name']
                )
            
            trace_spans = self.parser.parse(data, probe_sources)
            
            model_spans = [span for span in trace_spans if span.op_name == 'model']
            
            for span in model_spans:
                self.bubble_rate_metric.record(
                    timestamp=span.start_timestamp,
                    forward_duration=span.duration
                )
            
            if model_spans:
                bubble_rate = self.bubble_rate_metric.get_metrics()
                self.gpu_bubble_rate_gauge.labels(**labels).set(bubble_rate)
            
            from lmtracer.core.metrics import ExecutionTimeStats
            for span in trace_spans:
                with self._metrics_lock:
                    if span.op_name not in self.execution_time_metrics:
                        self.execution_time_metrics[span.op_name] = ExecutionTimeStats()
                    
                    self.execution_time_metrics[span.op_name].record(span.duration)
    
    def export_meta_data(self, meta: Any) -> None:
        pass
    
    def __del__(self):
        self._stop_event.set()
        if self._update_thread.is_alive():
            self._update_thread.join(timeout=1)

class VLLMMetricPrometheusExporter(DataExporter):
    
    def __init__(self):
        from lmtracer.core.metrics import GPUBubbleRateMetric, ExecutionTimeStats
        import threading
        
        self.parser = BinaryDumpParser()
        self.bubble_rate_metric = GPUBubbleRateMetric()
        self.execution_time_metrics: Dict[str, ExecutionTimeStats] = {}  # Dict[str, ExecutionTimeStats] for each phase_name
        self._metrics_lock = threading.Lock()  # Lock for thread-safe access to metrics
        
        self.torch_global_rank = self._get_torch_global_rank()
        self.gpu_device_index = self._get_gpu_device_index()
        self.hostname = platform.node()
        self.pid = os.getpid()
        
        self.gpu_bubble_rate_gauge = None
        self.model_execution_time_gauge = None
        self.model_execution_time_median_gauge = None
        self.model_execution_time_p95_gauge = None
        self.model_execution_time_p99_gauge = None
        self.model_execution_time_std_gauge = None
        self.model_execution_time_min_gauge = None
        self.model_execution_time_max_gauge = None
        self._gauge_labels = None
        self._metrics_initialized = False
        
        self._stop_event = threading.Event()
        self._update_thread = threading.Thread(target=self._periodic_metrics_update, daemon=True)
        self._update_thread.start()
    
    def _ensure_metrics_initialized(self, labels: dict) -> None:
        if self._metrics_initialized and self._gauge_labels == labels:
            return
        
        multiproc_dir = os.environ.get('PROMETHEUS_MULTIPROC_DIR')
        if not multiproc_dir:
            print(f"[lmtracer-vLLM] WARNING: PROMETHEUS_MULTIPROC_DIR not set!")
            print(f"[lmtracer-vLLM] Worker metrics will not be visible in main process /metrics")
        else:
            print(f"[lmtracer-vLLM] Using multiprocess directory: {multiproc_dir}")


        from prometheus_client import Gauge
        self._gauge_labels = labels
        label_keys = list(labels.keys())
        
        print(f"[lmtracer-vLLM] Creating metrics in worker PID={self.pid}")
        print(f"[lmtracer-vLLM] Labels: {label_keys}")
        
        self.gpu_bubble_rate_gauge = Gauge(
            name='lmtracer_gpu_bubble_rate',
            documentation='GPU bubble rate measured by forward duration',
            labelnames=label_keys,
            multiprocess_mode='sum'  # Aggregate across workers by summing
        )
        self.model_execution_time_gauge = Gauge(
            name='lmtracer_model_execution_time_ns',
            documentation='Average model execution time in nanoseconds for different operations',
            labelnames=label_keys + ['phase_name'],
            multiprocess_mode='sum'
        )
        self.model_execution_time_median_gauge = Gauge(
            name='lmtracer_model_execution_time_median_ns',
            documentation='Median model execution time in nanoseconds for different operations',
            labelnames=label_keys + ['phase_name'],
            multiprocess_mode='sum'
        )
        self.model_execution_time_p95_gauge = Gauge(
            name='lmtracer_model_execution_time_p95_ns',
            documentation='P95 model execution time in nanoseconds for different operations',
            labelnames=label_keys + ['phase_name'],
            multiprocess_mode='sum'
        )
        self.model_execution_time_p99_gauge = Gauge(
            name='lmtracer_model_execution_time_p99_ns',
            documentation='P99 model execution time in nanoseconds for different operations',
            labelnames=label_keys + ['phase_name'],
            multiprocess_mode='sum'
        )
        self.model_execution_time_std_gauge = Gauge(
            name='lmtracer_model_execution_time_std_ns',
            documentation='Standard deviation of model execution time in nanoseconds for different operations',
            labelnames=label_keys + ['phase_name'],
            multiprocess_mode='sum'
        )
        self.model_execution_time_min_gauge = Gauge(
            name='lmtracer_model_execution_time_min_ns',
            documentation='Minimum model execution time in nanoseconds for different operations',
            labelnames=label_keys + ['phase_name'],
            multiprocess_mode='min'  # Take minimum across workers
        )
        self.model_execution_time_max_gauge = Gauge(
            name='lmtracer_model_execution_time_max_ns',
            documentation='Maximum model execution time in nanoseconds for different operations',
            labelnames=label_keys + ['phase_name'],
            multiprocess_mode='max'  # Take maximum across workers
        )
        
        self._metrics_initialized = True
        print(f"[lmtracer-vLLM] Successfully created 8 metric families")
        print(f"[lmtracer-vLLM] Metrics will be written to multiprocess files for main process to collect")
    
    def _get_torch_global_rank(self) -> str:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return str(dist.get_rank())
        except ImportError:
            pass
        
        import os
        if 'VLLM_RANK' in os.environ:
            return os.environ['VLLM_RANK']
        if 'RANK' in os.environ:
            return os.environ['RANK']
        
        return '-1'
    
    def _get_gpu_device_index(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                # Get current CUDA device
                device_index = torch.cuda.current_device()
                return str(device_index)
        except ImportError:
            pass
        
        import os
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
            if devices:
                return devices[0]
        
        return '-1'

    def _periodic_metrics_update(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(1)  # Check every 1 second
            self._update_execution_time_metrics()
    
    def _update_execution_time_metrics(self) -> None:
        if self._gauge_labels is None or self.model_execution_time_gauge is None:
            return
        
        with self._metrics_lock:
            for phase_name, metric in self.execution_time_metrics.items():
                # Check if we should flush this metric
                if metric.should_flush():
                    # Get current metrics
                    metrics = metric.get_metrics()
                    labels = {**self._gauge_labels, 'phase_name': phase_name}
                    
                    # Update prometheus gauges
                    self.model_execution_time_gauge.labels(**labels).set(metrics['avg'])
                    self.model_execution_time_median_gauge.labels(**labels).set(metrics['median'])
                    self.model_execution_time_p95_gauge.labels(**labels).set(metrics['p95'])
                    self.model_execution_time_p99_gauge.labels(**labels).set(metrics['p99'])
                    self.model_execution_time_std_gauge.labels(**labels).set(metrics['std'])
                    self.model_execution_time_min_gauge.labels(**labels).set(metrics['min'])
                    self.model_execution_time_max_gauge.labels(**labels).set(metrics['max'])

    def export_trace_data(self, data: Any, probe_sources: List['ProbeSource'] = None, metric_labels: dict = None) -> None:
        if data is not None and self.parser is not None:
            labels = {
                'torch_global_rank': self.torch_global_rank,
                'gpu_device_index': self.gpu_device_index,
                'hostname': self.hostname,
                'pid': str(self.pid),
                'engine': 'vllm'  # Add engine label for vLLM
            }
            if metric_labels:
                labels.update(metric_labels)
            
            self._ensure_metrics_initialized(labels)
            trace_spans = self.parser.parse(data, probe_sources)
            model_spans = [span for span in trace_spans if span.op_name == 'model']
            
            for span in model_spans:
                self.bubble_rate_metric.record(
                    timestamp=span.start_timestamp,
                    forward_duration=span.duration
                )
            
            if model_spans:
                bubble_rate = self.bubble_rate_metric.get_metrics()
                self.gpu_bubble_rate_gauge.labels(**labels).set(bubble_rate)
            
            from lmtracer.core.metrics import ExecutionTimeStats
            for span in trace_spans:
                with self._metrics_lock:
                    if span.op_name not in self.execution_time_metrics:
                        self.execution_time_metrics[span.op_name] = ExecutionTimeStats()
                    
                    self.execution_time_metrics[span.op_name].record(span.duration)
    
    def export_meta_data(self, meta: Any) -> None:
        pass
    
    def __del__(self):
        self._stop_event.set()
        if self._update_thread.is_alive():
            self._update_thread.join(timeout=1)
