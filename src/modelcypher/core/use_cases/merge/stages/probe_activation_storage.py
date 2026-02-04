# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Activation accumulation and paging helpers for probe stage."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _accumulate_activation(
    storage: dict[int, "Array"],
    layer_idx: int,
    act: "Array",
    backend: "Backend",
    probe_index: int,
    total_probes: int,
) -> None:
    """Accumulate activations in a fixed-size buffer per layer."""
    if not hasattr(act, "shape"):
        act = backend.array(act)

    act_shape = backend.shape(act)
    if len(act_shape) == 1:
        act = backend.reshape(act, (1, act_shape[0]))
        act_shape = backend.shape(act)

    if layer_idx not in storage:
        dim = int(act_shape[1])
        dtype = getattr(act, "dtype", None)
        storage[layer_idx] = backend.zeros((total_probes, dim), dtype=dtype)
        backend.eval(storage[layer_idx])
    else:
        existing = storage[layer_idx]
        existing_shape = backend.shape(existing)
        if int(existing_shape[0]) < total_probes:
            dim = int(existing_shape[1])
            dtype = getattr(existing, "dtype", None)
            pad_rows = total_probes - int(existing_shape[0])
            padding = backend.zeros((pad_rows, dim), dtype=dtype)
            storage[layer_idx] = backend.concatenate([existing, padding], axis=0)
    backend.eval(storage[layer_idx])

    dim = int(backend.shape(act)[1])
    indices = backend.full((1, dim), probe_index, dtype="int32")
    storage[layer_idx] = backend.put_along_axis(
        storage[layer_idx], indices, act, axis=0
    )
    backend.eval(storage[layer_idx])


def _accumulate_activation_batch(
    storage: dict[int, "Array"],
    layer_idx: int,
    acts: list["Array"],
    probe_indices: list[int],
    backend: "Backend",
    total_probes: int,
) -> None:
    if not acts:
        return
    if len(acts) != len(probe_indices):
        raise ValueError("Batch activation count does not match probe indices")

    normalized: list["Array"] = []
    for act in acts:
        if not hasattr(act, "shape"):
            act = backend.array(act)
        normalized.append(act)

    stacked = backend.stack(normalized, axis=0)
    backend.eval(stacked)

    if layer_idx not in storage:
        dim = int(backend.shape(stacked)[1])
        storage[layer_idx] = backend.zeros((total_probes, dim), dtype=stacked.dtype)
        backend.eval(storage[layer_idx])
    else:
        existing = storage[layer_idx]
        existing_shape = backend.shape(existing)
        if int(existing_shape[0]) < total_probes:
            dim = int(existing_shape[1])
            pad_rows = total_probes - int(existing_shape[0])
            padding = backend.zeros((pad_rows, dim), dtype=existing.dtype)
            storage[layer_idx] = backend.concatenate([existing, padding], axis=0)
            backend.eval(storage[layer_idx])

    dim = int(backend.shape(stacked)[1])
    idx_arr = backend.array(probe_indices, dtype="int32")
    idx_arr = backend.reshape(idx_arr, (-1, 1))
    idx_mat = backend.broadcast_to(idx_arr, (len(probe_indices), dim))
    storage[layer_idx] = backend.put_along_axis(
        storage[layer_idx], idx_mat, stacked, axis=0
    )
    backend.eval(storage[layer_idx])


def _flush_batch_activations(
    storage: dict[int, "Array"],
    acts_by_layer: dict[int, list["Array"]],
    indices_by_layer: dict[int, list[int]],
    backend: "Backend",
    total_probes: int,
) -> None:
    for layer_idx, acts in acts_by_layer.items():
        probe_indices = indices_by_layer.get(layer_idx, [])
        if not probe_indices:
            continue
        _accumulate_activation_batch(
            storage,
            layer_idx,
            acts,
            probe_indices,
            backend,
            total_probes,
        )


class PagedActivations:
    """Lazy activation loader backed by per-layer safetensors files."""

    def __init__(
        self,
        base_dir: Path,
        prefix: str,
        layer_indices: list[int],
        backend: "Backend",
        cache_size: int = 2,
    ) -> None:
        self._base_dir = base_dir
        self._prefix = prefix
        self._layers = list(layer_indices)
        self._layer_set = set(layer_indices)
        self._backend = backend
        self._cache: OrderedDict[int, "Array"] = OrderedDict()
        self._cache_size = max(1, int(cache_size))

    def _layer_path(self, layer_idx: int) -> Path:
        return self._base_dir / f"{self._prefix}_{layer_idx}.safetensors"

    def _load_layer(self, layer_idx: int) -> "Array | None":
        if layer_idx in self._cache:
            self._cache.move_to_end(layer_idx)
            return self._cache[layer_idx]

        path = self._layer_path(layer_idx)
        if not path.exists():
            return None
        loaded = self._backend.load_safetensors(str(path))
        if not loaded:
            return None
        key = f"{self._prefix}_{layer_idx}"
        arr = loaded.get(key)
        if arr is None and loaded:
            arr = next(iter(loaded.values()))
        if arr is None:
            return None

        self._cache[layer_idx] = arr
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
            if hasattr(self._backend, "clear_cache"):
                self._backend.clear_cache()
        return arr

    def __contains__(self, layer_idx: object) -> bool:
        return int(layer_idx) in self._layer_set if isinstance(layer_idx, int) else False

    def __getitem__(self, layer_idx: int) -> "Array":
        value = self._load_layer(layer_idx)
        if value is None:
            raise KeyError(layer_idx)
        return value

    def get(self, layer_idx: int, default: Any = None) -> Any:
        value = self._load_layer(layer_idx)
        return value if value is not None else default

    def keys(self) -> list[int]:
        return list(self._layers)

    def items(self) -> list[tuple[int, "Array"]]:
        return [(layer_idx, self[layer_idx]) for layer_idx in self._layers]

    def __iter__(self):
        return iter(self._layers)

    def __len__(self) -> int:
        return len(self._layers)

    def __bool__(self) -> bool:
        return bool(self._layers)

    def clear_cache(self) -> None:
        self._cache.clear()
        if hasattr(self._backend, "clear_cache"):
            self._backend.clear_cache()

    def clear(self) -> None:
        self.clear_cache()


def _page_activation_space(
    base_dir: Path,
    prefix: str,
    activations: dict[int, "Array"],
    backend: "Backend",
    cache_size: int = 2,
) -> PagedActivations:
    base_dir.mkdir(parents=True, exist_ok=True)
    layer_indices = sorted(activations.keys())
    for layer_idx, acts in activations.items():
        path = base_dir / f"{prefix}_{layer_idx}.safetensors"
        backend.save_safetensors(str(path), {f"{prefix}_{layer_idx}": acts})
    activations.clear()
    return PagedActivations(
        base_dir=base_dir,
        prefix=prefix,
        layer_indices=layer_indices,
        backend=backend,
        cache_size=cache_size,
    )
