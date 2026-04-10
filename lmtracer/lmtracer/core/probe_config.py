
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List
from enum import Enum

if TYPE_CHECKING:
    from lmtracer.core.exporter import DataExporter

class SupportedEngines(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    MEGATRON = "megatron"

@dataclass
class Probe:

    phase_name: str = ""
    class_name: str = ""
    inner_graph: bool = False

@dataclass
class ExporterConfig:
    class_name: str
    params: Dict[str, Any]

def default_torchinductor_dir() -> str:
    import os
    home_dir = os.path.expanduser("~")
    default_dir = os.path.join(home_dir, '.cache', "lmtracer", "torchinductor_cache")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir

@dataclass
class ProbeConfig:
    dump_mode: str = "doorbell"
    llm_engine: SupportedEngines = SupportedEngines.VLLM
    num_flush: int = 1024
    torchinductor_dir: str = default_torchinductor_dir()
    probes: List[Probe] = field(default_factory=list)
    exporters: List['DataExporter'] = field(default_factory=list)
    
    @property
    def exporter(self) -> 'DataExporter':
        """Backward compatibility: return first exporter"""
        return self.exporters[0] if self.exporters else None

_global_probe_config: ProbeConfig = None

def get_global_probe_config() -> ProbeConfig:
    global _global_probe_config
    return _global_probe_config

def set_global_probe_config(config: ProbeConfig) -> None:
    global _global_probe_config
    _global_probe_config = config