
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union
from vllm.platforms.cuda import CudaPlatform

import vllm.envs as envs

from lmtracer.core.probe_context import set_current_probe_context
from importlib import resources

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


class LMTracerPlatform(CudaPlatform):
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        super().check_and_update_config(vllm_config)

        if vllm_config.parallel_config.worker_cls == "vllm.v1.worker.gpu_worker.Worker":
            vllm_config.parallel_config.worker_cls = "lmtracer.plugins.vllm.v1.worker.lmtracer_gpu_worker.LMTracerWorker"
        