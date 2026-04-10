from lmtracer.core.hooks import LMTRACER_HOOKED_ORIGINAL
from lmtracer.core.process_binder import ProcessBinder


ModelRunner = LMTRACER_HOOKED_ORIGINAL["sglang.srt.model_executor.model_runner.ModelRunner"]

class LMTracerModelRunner(ModelRunner):

    def __init__(self, *args, **kwargs):
        # Extract dp_rank from kwargs if present, otherwise None
        self.dp_rank = kwargs.get('dp_rank', None)
        self.pp_rank = kwargs.get('pp_rank', None)
        self.tp_rank = kwargs.get('tp_rank', None)
        super().__init__(*args, **kwargs)
    
    def initialize(self, *args, **kwargs):
        self.labels = {
            "model_name": self.server_args.served_model_name,
            "tp_rank": self.tp_rank,
            "pp_rank": self.pp_rank,    
        }
        if self.dp_rank is not None:
            self.labels["dp_rank"] = self.dp_rank
        self.process_binder = ProcessBinder("sglang", self.model_config.model_path, self.labels)
        super().initialize(*args, **kwargs)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        self.process_binder.after_execution()
        return output
    
    def forward_extend(self, *args, **kwargs):
        output = super().forward_extend(*args, **kwargs)
        return output
