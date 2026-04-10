from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TracePoint:
    trace_point_id: int
    timestamp: int
    op_name: str
    start_or_end: str  # 'start' or 'end'


@dataclass
class TraceSpan:
    op_name: str
    start_timestamp: int
    end_timestamp: int
    start_trace_id: int
    end_trace_id: int
    
    @property
    def duration(self) -> int:
        return self.end_timestamp - self.start_timestamp


class DumpParser(ABC):
    
    @abstractmethod
    def parse(self, byte_data: bytes, probe_sources: Any = None) -> List[TraceSpan]:
        pass


class BinaryDumpParser(DumpParser):
    def __init__(self):
        self.op_name_begin = {}  # op_name -> (timestamp, trace_id)
    
    def parse(self, byte_data: bytes, probe_sources: Any = None) -> List[TraceSpan]:
        if byte_data is None:
            return []
        
        trace_point_id_to_name = {}
        if probe_sources is not None:
            for probe_source in probe_sources:
                trace_point_id_to_name[probe_source.probe_start_id] = f'{probe_source.phase_name}_start'
                trace_point_id_to_name[probe_source.probe_end_id] = f'{probe_source.phase_name}_end'
        
        trace_points = []
        num_entries = len(byte_data) // 16
        
        for i in range(num_entries):
            offset = i * 16
            trace_point_id = int.from_bytes(byte_data[offset:offset+8], byteorder='little')
            timestamp = int.from_bytes(byte_data[offset+8:offset+16], byteorder='little')
            
            op_name_start_end = trace_point_id_to_name.get(trace_point_id, f'unknown_op_{trace_point_id}')
            
            parts = op_name_start_end.rsplit('_', 1)
            if len(parts) == 2:
                op_name = parts[0]
                start_or_end = parts[1]
            else:
                op_name = op_name_start_end
                start_or_end = 'unknown'
            
            trace_points.append(TracePoint(
                trace_point_id=trace_point_id,
                timestamp=timestamp,
                op_name=op_name,
                start_or_end=start_or_end
            ))
        
        trace_spans = []
        
        for tp in trace_points:
            if tp.start_or_end == 'start':
                self.op_name_begin[tp.op_name] = (tp.timestamp, tp.trace_point_id)
            elif tp.start_or_end == 'end':
                if tp.op_name in self.op_name_begin:
                    start_ts, start_id = self.op_name_begin[tp.op_name]
                    trace_spans.append(TraceSpan(
                        op_name=tp.op_name,
                        start_timestamp=start_ts,
                        end_timestamp=tp.timestamp,
                        start_trace_id=start_id,
                        end_trace_id=tp.trace_point_id
                    ))
                    del self.op_name_begin[tp.op_name]
        
        return trace_spans
    
    def parse_raw_points(self, byte_data: bytes, probe_sources: Any = None) -> List[TracePoint]:
        if byte_data is None:
            return []
        
        trace_point_id_to_name = {}
        if probe_sources is not None:
            for probe_source in probe_sources:
                trace_point_id_to_name[probe_source.probe_start_id] = f'{probe_source.phase_name}_start'
                trace_point_id_to_name[probe_source.probe_end_id] = f'{probe_source.phase_name}_end'
        
        trace_points = []
        num_entries = len(byte_data) // 16
        
        for i in range(num_entries):
            offset = i * 16
            trace_point_id = int.from_bytes(byte_data[offset:offset+8], byteorder='little')
            timestamp = int.from_bytes(byte_data[offset+8:offset+16], byteorder='little')
            
            op_name_start_end = trace_point_id_to_name.get(trace_point_id, f'unknown_op_{trace_point_id}')
            
            parts = op_name_start_end.rsplit('_', 1)
            if len(parts) == 2:
                op_name = parts[0]
                start_or_end = parts[1]
            else:
                op_name = op_name_start_end
                start_or_end = 'unknown'
            
            trace_points.append(TracePoint(
                trace_point_id=trace_point_id,
                timestamp=timestamp,
                op_name=op_name,
                start_or_end=start_or_end
            ))
        
        return trace_points
