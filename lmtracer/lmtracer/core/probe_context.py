from dataclasses import dataclass, field
from importlib import resources
import inspect
import os
from typing import Any, Dict, List, Optional

from lmtracer.core.logger import lmtracer_log
from lmtracer.core.probe_config import ProbeConfig, Probe, SupportedEngines, set_global_probe_config
from lmtracer.core.decorators import module_probing
from lmtracer.core.env_config import get_env_config

from lmtracer.core.trace_buffer import TraceBuffer
from lmtracer.core.utils import hash_to_uint32, resolve_obj_by_qualname
from torch import Tensor, nn

@dataclass
class ProbeSource(Probe):
    source_file: str = ""
    line_no_start: int = 0
    line_no_end: int = 0
    probe_start_id: int = 0
    probe_end_id: int = 0
    inner_graph: bool = False
    loop_hint: bool = False

    def __post_init__(self):
        self.probe_start_id = hash_to_uint32(f"{self.source_file}_{self.class_name}_{self.line_no_start}")
        self.probe_end_id = hash_to_uint32(f"{self.source_file}_{self.class_name}_{self.line_no_end}")

@dataclass
class ModuleBindings:
    trace_buffer: Tensor = None
    buffer_size: int = 0
    num_flush: int = 0
    current_buffer: Tensor = None
    doorbell_mapped_device_ptr: Any = None
    probe_id_before: int = None
    probe_id_after: int = None


@dataclass
class PhaseStackTraceState:
    phase: str
    entered: bool = False
    # The below two fields should be inited every time when entering this phase
    child_stack_by_lineno: Dict[int, Dict[str, Dict[int, Any]]] = field(default_factory=dict) # filename -> lineno -> FxNodeStackTraceInvoke of each previous invoke
    prev_line_no: int = -1

def is_in_torch_compile_context() -> bool:
    global _is_in_torch_compile
    return _is_in_torch_compile

class ProbeContext:

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.probe_sources: List[ProbeSource] = []

        self._phase_stack_trace_state: Dict[str, PhaseStackTraceState] = None
        self._module_bindings: Dict[str, ModuleBindings] = {}

        self.trace_buffer: TraceBuffer = None
        self.is_last_graph = False
        self.logger = None
        
        self._patch_nn_modules()

        

    def _patch_nn_modules(self):
        for probe in self.config.probes:
            class_name = probe.class_name
            module_cls = resolve_obj_by_qualname(class_name)

            if module_cls is None:
                raise ValueError(f"cannot resolve class: {class_name}")
            if not issubclass(module_cls, nn.Module):
                raise ValueError(f"class {class_name} is not a subclass of nn.Module")
            
            if not hasattr(module_cls, "forward"):
                raise ValueError(f"class {class_name} does not have a forward method")
            
            source_lines, starting_line = inspect.getsourcelines(module_cls.forward)
            file = inspect.getsourcefile(module_cls.forward)
            line_no_start = starting_line
            line_no_end = starting_line + len(source_lines) - 1
            graph_probe_source = ProbeSource(
                phase_name=probe.phase_name,
                class_name=class_name,
                source_file=file,
                line_no_start=line_no_start,
                line_no_end=line_no_end,
                inner_graph=probe.inner_graph
            )
            self.probe_sources.append(graph_probe_source)

            if not probe.inner_graph or True:

                module_bindings = ModuleBindings(
                    probe_id_before=graph_probe_source.probe_start_id,
                    probe_id_after=graph_probe_source.probe_end_id
                )

                self._module_bindings[class_name] = module_bindings

                if hasattr(module_cls, "forward"):
                    setattr(module_cls, "module_bindings", module_bindings)
                    module_probing(module_cls)

            else:
                # For inner graph modules, we will handle them during graph tracing.
                pass
                    
    def init_probe_phase_state(self):
        self._phase_stack_trace_state = {}
        for probe_source in self.probe_sources:
            self._phase_stack_trace_state[probe_source.phase_name] = PhaseStackTraceState(
                phase=probe_source.phase_name,
                prev_line_no=-1,
                entered=False
            )
    
    def get_probe_source_by_phase_name(self, phase_name: str) -> ProbeSource:
        for probe_source in self.probe_sources:
            if probe_source.phase_name == phase_name:
                return probe_source
        return None
    
    def get_probe_phase_state(self):
        return self._phase_stack_trace_state

    def update_trace_buffer_bindings(self, trace_buffer: TraceBuffer, num_flush: int):
        self.trace_buffer = trace_buffer
        for class_name, bindings in self._module_bindings.items():
            bindings.trace_buffer = trace_buffer.device_trace_buffer
            bindings.buffer_size = trace_buffer.buffer_size
            bindings.num_flush = num_flush
            bindings.current_buffer = trace_buffer.device_write_buffer_index_tensor
            bindings.doorbell_mapped_device_ptr = trace_buffer.doorbell_mapped_device_ptr
    

_probe_context: ProbeContext = None
def get_current_probe_context() -> ProbeContext:
    global _probe_context
    if _probe_context is None:
        raise ValueError("ProbeContext has not been set yet.")
    return _probe_context

def set_current_probe_context(context: ProbeContext) -> None:
    global _probe_context
    _probe_context = context

def _get_probe_config_by_name(name: str, framework) -> str:
    print(f"Initializing preset probe config for model name and framework: {name}, {framework}")
    if "qwen3" in name.lower():
        return f"presets/{framework}/preset_qwen3.yaml"
    elif "llama" in name.lower():
        return f"presets/{framework}/preset_llama.yaml"
    elif "deepseek" in name.lower():
        return f"presets/{framework}/preset_deepseek.yaml"
    elif "gpt" in name.lower():
        return f"presets/{framework}/preset_gpt_oss.yaml"
    else:
        raise ValueError(f"unknown model name and framework for preset probe config: {name}, {framework}")

def _load_config_from_yaml(config_path: str) -> ProbeConfig:
    import yaml
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    probe_config = ProbeConfig()
    
    # Parse basic config
    dump_mode = config_data.get("dump_mode", "doorbell")
    engine = config_data.get("llm_engine", "vllm")
    num_flush = config_data.get("num_flush", None)
    
    # Parse probes
    probes = []
    for probe in config_data.get("probes", []):
        probes.append(
            Probe(
                phase_name=probe["phase"],
                class_name=probe["class_name"],
                inner_graph=probe.get("inner_graph", False)
            )
        )
    
    probe_config.probes = probes
    probe_config.dump_mode = dump_mode
    probe_config.llm_engine = SupportedEngines(engine)
    if num_flush is not None:
        probe_config.num_flush = num_flush
    
    # Parse exporters config (support both 'exporter' and 'exporters')
    exporters_list = []
    
    # Check for single exporter (backward compatibility)
    exporter_cfg = config_data.get("exporter", None)
    if exporter_cfg is not None:
        class_name = exporter_cfg.get("class_name", None)
        if class_name is not None:
            exporter_cls = resolve_obj_by_qualname(class_name)
            if exporter_cls is None:
                raise ValueError(f"cannot resolve exporter class: {class_name}")
            params = exporter_cfg.get("params", {})
            exporter_instance = exporter_cls(**params)
            exporters_list.append(exporter_instance)
    
    # Check for multiple exporters
    exporters_cfg = config_data.get("exporters", [])
    for idx, exp_cfg in enumerate(exporters_cfg):
        class_name = exp_cfg.get("class_name", None)
        if class_name is not None:
            exporter_cls = resolve_obj_by_qualname(class_name)
            if exporter_cls is None:
                raise ValueError(f"cannot resolve exporter class: {class_name}")
            params = exp_cfg.get("params", {})
            exporter_instance = exporter_cls(**params)
            exporters_list.append(exporter_instance)
    
    probe_config.exporters = exporters_list
    
    return probe_config

def _init_preset_probe_config(llm_engine: str, model_name: str) -> ProbeConfig:
    import yaml
    with resources.files("lmtracer").joinpath(_get_probe_config_by_name(model_name, llm_engine)).open("r") as f:
        preset_config = yaml.safe_load(f)
    probe_config = ProbeConfig()
    dump_mode = preset_config.get("dump_mode", "doorbell")
    engine = preset_config.get("llm_engine", "vllm")
    probes = []
    for probe in preset_config.get("probes", []):
        probes.append(
            Probe(
                phase_name=probe["phase"],
                class_name=probe["class_name"],
                inner_graph=probe.get("inner_graph", False)
            )
        )
    probe_config.probes = probes
    probe_config.dump_mode = dump_mode
    probe_config.llm_engine = SupportedEngines(engine)
    lmtracer_log(f"Initialized lmtracer preset probe config: dump_mode={dump_mode}, llm_engine={engine}, probes={probes}")
    
    # Parse exporters (support both single and multiple)
    exporters_list = []
    exporter_cfg = preset_config.get("exporter", None)
    if exporter_cfg is not None:
        class_name = exporter_cfg.get("class_name", None)
        if class_name is not None:
            exporter_cls = resolve_obj_by_qualname(class_name)
            if exporter_cls is None:
                raise ValueError(f"cannot resolve exporter class: {class_name}")
            params = exporter_cfg.get("params", {})
            exporter_instance = exporter_cls(**params)
            exporters_list.append(exporter_instance)
    
    exporters_cfg = preset_config.get("exporters", [])
    for idx, exp_cfg in enumerate(exporters_cfg):
        class_name = exp_cfg.get("class_name", None)
        if class_name is not None:
            exporter_cls = resolve_obj_by_qualname(class_name)
            if exporter_cls is None:
                raise ValueError(f"cannot resolve exporter class: {class_name}")
            params = exp_cfg.get("params", {})
            exporter_instance = exporter_cls(**params)
            exporters_list.append(exporter_instance)
    
    probe_config.exporters = exporters_list
    return probe_config

def init_probe_context(llm_engine: str, model_name: str = None) -> ProbeContext:
    env_config = get_env_config()
    
    if env_config.config_path:
        lmtracer_log(f"Using custom config from lmtracer_CONFIG: {env_config.config_path}")
        if not os.path.exists(env_config.config_path):
            raise FileNotFoundError(f"Config file not found: {env_config.config_path}")
        probe_config = _load_config_from_yaml(env_config.config_path)
    else:
        if model_name is None:
            raise ValueError("model_name is required when lmtracer_CONFIG is not set")
        lmtracer_log(f"Using preset config for {llm_engine}/{model_name}")
        probe_config = _init_preset_probe_config(llm_engine, model_name)
    
    if probe_config.num_flush is None and env_config.num_flush is not None:
        probe_config.num_flush = env_config.num_flush
        lmtracer_log(f"Applied lmtracer_NUM_FLUSH: {env_config.num_flush}")
    
    if probe_config.dump_mode is None:
        probe_config.dump_mode = env_config.dump_mode
        lmtracer_log(f"Applied lmtracer_DUMP_MODE: {env_config.dump_mode}")
    
    set_global_probe_config(probe_config)
    context = ProbeContext(probe_config)
    set_current_probe_context(context)

    os.environ['TORCHINDUCTOR_CACHE_DIR'] = probe_config.torchinductor_dir
    return context