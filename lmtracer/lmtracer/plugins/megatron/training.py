from lmtracer.core.process_binder import ProcessBinder
from lmtracer.core.hooks import LMTRACER_HOOKED_ORIGINAL

print("lmtracer Megatron training module loaded")

process_binder = None

def lmtracer_megatron_train(*args, **kwargs):
    global process_binder
    process_binder = ProcessBinder("megatron", "preset_gpt")

    # # Initialize ProcessBinder here if needed
    result = LMTRACER_HOOKED_ORIGINAL["megatron.training.training.train"](*args, **kwargs)

    process_binder.after_execution()
    
    return result

def lmtracer_megatron_train_step(*args, **kwargs):
    global process_binder

    # # Initialize ProcessBinder here if needed
    result = LMTRACER_HOOKED_ORIGINAL["megatron.training.training.train_step"](*args, **kwargs)
    process_binder.after_execution()
    
    return result
