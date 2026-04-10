import ctypes
import inspect
import json
import platform
import threading
import time
from unittest.mock import patch
import numpy as np
from lmtracer.core.logger import lmtracer_log
from lmtracer.core.process_binder import ProcessBinder
from vllm.v1.worker.gpu_worker import Worker
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union
from contextlib import contextmanager

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)

import torch
from vllm.logger import init_logger
import logging
import os

from lmtracer.core.exporter import FileByteDataExporter
from lmtracer.core.probe_context import get_current_probe_context, _init_preset_probe_config, init_probe_context
from lmtracer.core.trace_buffer import TraceBuffer

from lmtracer.cuda import globaltimer, timer_mem

from vllm.v1.core.sched.output import SchedulerOutput

class LMTracerGPUModelRunner(GPUModelRunner):

    def __init__(self, *args, **kwargs):

        import vllm.model_executor.models.utils

        original_make_layers = vllm.model_executor.models.utils.make_layers

        def lmtracer_make_layers(layer_confs, *args, **kwargs):
            probe_context = get_current_probe_context()
            start_layer, end_layer, layers = original_make_layers(layer_confs, *args, **kwargs)

            lmtracer_log(f"make_layer function has made {len(layers)} layers.")

            for layer in layers:
                if not isinstance(layer, torch.nn.Module):
                    continue
                if not hasattr(layer, "forward"):
                    continue

                source_lines, starting_line = inspect.getsourcelines(layer.forward)
                file = inspect.getsourcefile(layer.forward)
                line_no_start = starting_line
                line_no_end = starting_line + len(source_lines) - 1

                for probe_source in probe_context.probe_sources:
                    if (probe_source.source_file == file and
                        probe_source.line_no_start == line_no_start and
                        probe_source.line_no_end == line_no_end):
                        probe_source.loop_hint = True
                        
            return start_layer, end_layer, layers

        patcher = patch("vllm.model_executor.models.utils.make_layers", lmtracer_make_layers)
        patcher.start()

        super().__init__(*args, **kwargs)

        self.process_binder = ProcessBinder("vllm", self.model_config.model)
        lmtracer_log("Initialized LMTracerGPUModelRunner with ProcessBinder.")

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        result = super().execute_model(
            scheduler_output, intermediate_tensors)
        self.process_binder.after_execution()
        return result
