from __future__ import annotations

from typing import Any


class SafeGPU:
    def __init__(self, mx_module: Any) -> None:
        self.mx = mx_module

    def eval(self, *arrays: Any) -> None:
        if arrays:
            self.mx.eval(*arrays)

    def item(self, array: Any) -> Any:
        self.eval(array)
        if hasattr(array, "item"):
            return array.item()
        return array
