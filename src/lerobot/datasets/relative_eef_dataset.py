#!/usr/bin/env python

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE


@dataclass(frozen=True)
class RelativeEEFChunkSpec:
    horizon: int
    step_width: int
    base_action_names: tuple[str, ...]


def _to_numpy_1d(value: torch.Tensor | np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"Expected rank-1 value, got shape={arr.shape}.")
    return arr


def _infer_relative_eef_chunk_spec(feature_info: dict[str, Any]) -> RelativeEEFChunkSpec:
    shape = feature_info.get("shape")
    chunk_meta = feature_info.get("relative_eef_chunk")
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        raise ValueError(f"Expected flat action feature for relative EEF chunk dataset, got shape={shape}.")
    if not isinstance(chunk_meta, dict):
        raise ValueError(
            "Expected `features.action.relative_eef_chunk` metadata. "
            "This wrapper is intended for postprocessed relative-eef chunk datasets."
        )

    horizon = int(chunk_meta.get("horizon", 0))
    step_width = int(chunk_meta.get("step_width", 0))
    if horizon <= 0 or step_width <= 0:
        raise ValueError(
            f"Invalid relative_eef_chunk metadata: horizon={horizon}, step_width={step_width}."
        )
    if int(shape[0]) != horizon * step_width:
        raise ValueError(
            "Action feature shape does not match relative_eef_chunk metadata: "
            f"shape={shape[0]} horizon={horizon} step_width={step_width}."
        )

    raw_names = feature_info.get("names")
    if isinstance(raw_names, list) and len(raw_names) == horizon * step_width:
        base_names = []
        for i in range(step_width):
            name = str(raw_names[i])
            prefix = "step0_"
            base_names.append(name[len(prefix) :] if name.startswith(prefix) else name)
    else:
        base_names = [f"dim_{i}" for i in range(step_width)]

    return RelativeEEFChunkSpec(
        horizon=horizon,
        step_width=step_width,
        base_action_names=tuple(base_names),
    )


def _reshape_stats_vector(values: Any, *, horizon: int, step_width: int) -> Any:
    arr = np.asarray(values)
    if arr.ndim == 1 and arr.size == horizon * step_width:
        return arr.reshape(horizon, step_width).tolist()
    return values


class RelativeEEFDataset(Dataset):
    """Wrapper for reading postprocessed relative-EEF chunk datasets during training.

    Expected dataset semantics:
    - `observation.state`: already stores previous->current relative motion
    - `action`: already stores a flattened future chunk with shape `(H * D,)`
    - `action_is_pad`: already stores the future-step padding mask with shape `(H,)`

    This wrapper:
    - reshapes `action` from `(H * D,)` to `(H, D)`
    - preserves `action_is_pad` from the stored parquet row
    - patches `meta.features` / `meta.stats` so policies see per-step action dim `D`
    """

    def __init__(
        self,
        base_dataset: LeRobotDataset,
        *,
        state_key: str = OBS_STATE,
        action_key: str = ACTION,
        action_pad_key: str = "action_is_pad",
        keep_raw_fields: bool = False,
    ) -> None:
        if not isinstance(base_dataset, LeRobotDataset):
            raise TypeError(
                "RelativeEEFDataset currently supports wrapping LeRobotDataset only. "
                f"Got {type(base_dataset)!r}."
            )
        if state_key not in base_dataset.features:
            raise KeyError(f"Missing feature `{state_key}` in base dataset.")
        if action_key not in base_dataset.features:
            raise KeyError(f"Missing feature `{action_key}` in base dataset.")

        self.base_dataset = base_dataset
        self.state_key = state_key
        self.action_key = action_key
        self.action_pad_key = action_pad_key
        self.keep_raw_fields = keep_raw_fields
        self.chunk_spec = _infer_relative_eef_chunk_spec(base_dataset.features[action_key])

        self.meta = copy.copy(base_dataset.meta)
        self.meta.info = copy.deepcopy(base_dataset.meta.info)
        self.meta.stats = copy.deepcopy(base_dataset.meta.stats)
        self._patch_meta()

    def _patch_meta(self) -> None:
        features = self.meta.info.get("features")
        if not isinstance(features, dict):
            raise ValueError("Dataset metadata missing `features`.")

        action_feature = copy.deepcopy(features[self.action_key])
        action_feature["shape"] = [self.chunk_spec.step_width]
        action_feature["names"] = list(self.chunk_spec.base_action_names)
        if isinstance(action_feature.get("relative_eef_chunk"), dict):
            action_feature["relative_eef_chunk"]["flattened"] = False
        features[self.action_key] = action_feature

        # Prevent policies/factory from misclassifying `action_is_pad` as another ACTION feature.
        features.pop(self.action_pad_key, None)

        action_stats = self.meta.stats.get(self.action_key)
        if isinstance(action_stats, dict):
            for key in ("min", "max", "mean", "std", "count"):
                if key in action_stats:
                    action_stats[key] = _reshape_stats_vector(
                        action_stats[key],
                        horizon=self.chunk_spec.horizon,
                        step_width=self.chunk_spec.step_width,
                    )

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)

    def _load_raw_row(self, idx: int) -> dict[str, Any]:
        self.base_dataset._ensure_hf_dataset_loaded()
        return self.base_dataset.hf_dataset[idx]

    def _reshape_action(self, raw_action: Any) -> np.ndarray:
        arr = _to_numpy_1d(raw_action).astype(np.float32, copy=False)
        if arr.size != self.chunk_spec.horizon * self.chunk_spec.step_width:
            raise ValueError(
                "Unexpected flattened action size for relative EEF dataset: "
                f"got={arr.size} expected={self.chunk_spec.horizon * self.chunk_spec.step_width}."
            )
        return arr.reshape(self.chunk_spec.horizon, self.chunk_spec.step_width)

    def _reshape_action_pad(self, raw_pad: Any) -> np.ndarray:
        arr = _to_numpy_1d(raw_pad).astype(bool, copy=False)
        if arr.size != self.chunk_spec.horizon:
            raise ValueError(
                f"Unexpected action pad size: got={arr.size} expected={self.chunk_spec.horizon}."
            )
        return arr

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base_dataset[idx]
        raw_row = self._load_raw_row(idx)

        raw_state = torch.as_tensor(_to_numpy_1d(raw_row[self.state_key]).astype(np.float32, copy=False))
        raw_action = torch.as_tensor(self._reshape_action(raw_row[self.action_key]))
        raw_action_is_pad = torch.as_tensor(self._reshape_action_pad(raw_row[self.action_pad_key]))

        if self.keep_raw_fields:
            item[f"raw.{self.state_key}"] = item.get(self.state_key)
            item[f"raw.{self.action_key}"] = item.get(self.action_key)
            item[f"raw.{self.action_pad_key}"] = item.get(self.action_pad_key)

        item[self.state_key] = raw_state
        item[self.action_key] = raw_action
        item[self.action_pad_key] = raw_action_is_pad
        return item
