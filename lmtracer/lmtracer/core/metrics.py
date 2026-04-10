from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List
import time

@dataclass
class MetricRecord:
    timestamp: int
    value: Any

class StatefulMetric(ABC):
    @abstractmethod
    def record(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_metrics(self, *args, **kwargs) -> Any:
        pass

class GPUBubbleRateMetric(StatefulMetric):
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.prev_data = 0.0
        self.durations: List[MetricRecord] = []

    def record(self, timestamp: int, forward_duration: int) -> None:
        self.durations.append(MetricRecord(timestamp, forward_duration))

    def get_metrics(self) -> dict:
        if len(self.durations) < 2 or len(self.durations) < self.window_size:
            return self.prev_data
        total_duration = sum(record.value for record in self.durations)
        time_span = self.durations[-1].timestamp - self.durations[0].timestamp + self.durations[-1].value
        bubble_rate = 1.0 - (total_duration / time_span)
        self.prev_data = bubble_rate
        self.durations = []  # Reset after calculation
        return bubble_rate

class ExecutionTimeStats(StatefulMetric):
    def __init__(self, window_size: int = 10, time_window_seconds: int = 60):
        self.window_size = window_size
        self.time_window_seconds = time_window_seconds  # Default 60 seconds
        self.prev_data = {'avg': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        self.durations: List[int] = []
        self.last_update_time: float = None  # System time when last record was added

    def record(self, duration: int) -> None:
        self.last_update_time = time.time()  # Update on every record
        self.durations.append(duration)

    def should_flush(self) -> bool:
        import time
        if self.last_update_time is not None:
            time_elapsed = time.time() - self.last_update_time
            if time_elapsed > self.time_window_seconds:
                return True
        else:
            self.last_update_time = time.time()
        
        if len(self.durations) == 0:
            return False
        
        import time
        if len(self.durations) > 16:
            return True
        
        return False

    def get_metrics(self) -> dict:
        
        if not self.should_flush():
            return self.prev_data
        
        if len(self.durations) == 0:
            self.prev_data = 0
            return {'avg': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        import numpy as np
        avg_duration = float(np.mean(self.durations))
        median_duration = float(np.median(self.durations))
        p95 = float(np.percentile(self.durations, 95))
        p99 = float(np.percentile(self.durations, 99))
        std_duration = float(np.std(self.durations))
        min_duration = float(np.min(self.durations))
        max_duration = float(np.max(self.durations))
        
        self.prev_data = {
            'avg': avg_duration,
            'median': median_duration,
            'p95': p95,
            'p99': p99,
            'std': std_duration,
            'min': min_duration,
            'max': max_duration
        }

        self.durations = []  # Reset after calculation
        self.last_update_time = time.time()
        
        return self.prev_data
