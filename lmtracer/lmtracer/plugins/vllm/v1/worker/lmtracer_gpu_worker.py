# from lmtracer.lmtracer_trace_context import lmtracerVllmContext
from lmtracer.plugins.vllm.v1.worker.lmtracer_gpu_model_runner import LMTracerGPUModelRunner
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.outputs import ModelRunnerOutput
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union
from torch.utils.cpp_extension import load

from vllm.logger import init_logger

logger = init_logger(__name__)

import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from unittest.mock import patch

from lmtracer.cuda import globaltimer

class LMTracerWorker(Worker):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

        gpu_runner_patcher = patch("vllm.v1.worker.gpu_model_runner.GPUModelRunner", LMTracerGPUModelRunner)
        gpu_runner_patcher.start()
        worker_patcher = patch("vllm.v1.worker.gpu_worker.GPUModelRunner", LMTracerGPUModelRunner)
        worker_patcher.start()    


    def init_device(self):
        super().init_device()
        # self._sync_worker_gpu_time()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        # self.lmtracer_trace_context.put_scheduler_output(scheduler_output)
        result = super().execute_model(scheduler_output)
        return result

    
    def compile_or_warm_up_model(self) -> None:
        super().compile_or_warm_up_model()
        self.model_runner.after_warmup = True