import hashlib
import struct
from typing import Any
import importlib

def resolve_obj_by_qualname(qualname: str) -> Any:
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)

def hash_to_uint32(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return struct.unpack_from(">I", h)[0]