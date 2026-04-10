
from lmtracer.core.probe_config import SupportedEngines, get_global_probe_config


_logger = None

def lmtracer_log(log_msg: str) -> None:
    global _logger
    if _logger is not None:
        _logger(log_msg)
        return
    global_probe_config = get_global_probe_config()
    if global_probe_config is not None:
        try:
            if global_probe_config.llm_engine == SupportedEngines.VLLM:
                from vllm.logger import init_vllm_logger # type: ignore
                _logger = init_vllm_logger(__name__)
            elif global_probe_config.llm_engine == SupportedEngines.SGLANG:
                import logging # type: ignore
                _logger = logging.getLogger(__name__).info
            elif global_probe_config.llm_engine == SupportedEngines.MEGATRON:
                from megatron.training.utils import print_rank_0 # type: ignore
                _logger = print_rank_0
            else:
                raise ValueError(f"unsupported llm engine: {global_probe_config.llm_engine}")
            
            _logger(log_msg)
        except Exception:
            print(log_msg)