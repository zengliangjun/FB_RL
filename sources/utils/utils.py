from typing import Dict, Any, Type, TypeVar

def getattr_fix(cfg, name: str, default: Any):
    value = getattr(cfg, name)
    if value is None:
        return default
    return value

