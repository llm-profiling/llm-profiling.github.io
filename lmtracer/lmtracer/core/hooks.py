from importlib.abc import MetaPathFinder

from dataclasses import dataclass
import sys

from typing import Mapping

LMTRACER_HOOKED_ORIGINAL = {}

def patch_module_object(original_object_name=None, new_object_name=None):
    from unittest.mock import patch
    from importlib import import_module
    global LMTRACER_HOOKED_ORIGINAL

    if original_object_name is None and new_object_name is None:
        return
    
    
    if original_object_name is None and new_object_name is not None:
        import_module(new_object_name)
        return
    
    original_module_name, original_obj_name = original_object_name.rsplit(".", 1)
    original_module = import_module(original_module_name)
    original_obj = getattr(original_module, original_obj_name)
    LMTRACER_HOOKED_ORIGINAL[original_object_name] = original_obj

    new_module_name, new_obj_name = new_object_name.rsplit(".", 1)
    new_module = import_module(new_module_name)
    new_obj = getattr(new_module, new_obj_name)
    patcher = patch(original_object_name, new_obj)
    patcher.start()

@dataclass
class HookState:
    hook_keyword: str
    imported: bool = False
    being_patched: bool = False
    patched: bool = False
    hook_module_names: Mapping[str, str] = None

class LazyImportHook(MetaPathFinder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sglang_imported = False
        self._being_patched = False
        self._patched = False

        self._hook_state: Mapping[str, HookState] = {}

        self._hook_state["sglang"] = HookState(
            hook_keyword="sglang",
            imported=False,
            being_patched=False,
            patched=False,
            hook_module_names={
                "sglang.srt.model_executor.model_runner.ModelRunner": "lmtracer.plugins.sglang.srt.model_executor.model_runner.lmtracerModelRunner"
            }
        )

        self._hook_state["megatron"] = HookState(
            hook_keyword="megatron",
            imported=False,
            being_patched=False,
            patched=False,
            hook_module_names={
                "megatron.training.training.train": "lmtracer.plugins.megatron.training.lmtracer_megatron_train",
                "megatron.training.training.train_step": "lmtracer.plugins.megatron.training.lmtracer_megatron_train_step",
            }
        )


    def find_spec(self, fullname, path, target=None):

        for hook_key, hook_state in self._hook_state.items():
            if not fullname.startswith(hook_state.hook_keyword):
                continue
            
            if not hook_state.imported:
                # allow the first import to proceed normally
                hook_state.imported = True
                return None

            # subsequent imports, do the patching
            if not hook_state.patched and not hook_state.being_patched:
                hook_state.being_patched = True
                for original_class_name, new_class_name in hook_state.hook_module_names.items():
                    if new_class_name not in sys.modules:
                        patch_module_object(original_class_name, new_class_name)
                hook_state.patched = True
                hook_state.being_patched = False
            
            return None

sys.meta_path.insert(0, LazyImportHook())