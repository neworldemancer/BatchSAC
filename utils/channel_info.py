"""Shared helpers for discovering channel metadata in TIFF datasets."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path


_STACK_RE = re.compile(
    r"^(?P<prefix>.*?_Doc1_PMT - PMT )"
    r"(?P<ch_name>\[[^\]]+\])"
    r" _C(?P<ch_idx>\d+)_Time Time(?P<t_idx>\d+)\.ome\.tif$"
)


def parse_stack_fn(path: str):
    """Parse a BatchSAC TIFF stack filename."""
    name = os.path.basename(path)
    match = _STACK_RE.match(name)
    if not match:
        raise ValueError(f"Unrecognized stack filename: {name}")

    return {
        "ch_name": match.group("ch_name"),
        "ch_idx": int(match.group("ch_idx")),
        "t_idx": int(match.group("t_idx")),
    }


def get_dataset_channel_info(ds_path: str, glob_pattern: str = "*.tif"):
    """Return channel ids, channel names and time ids discovered in a dataset."""
    files = sorted(glob.glob(str(Path(ds_path) / glob_pattern)))
    if len(files) == 0:
        raise ValueError(f"No files matched {glob_pattern} in dataset: {ds_path}")

    ch_map = {}
    time_ids = set()
    for fn in files:
        info = parse_stack_fn(fn)
        ch_map.setdefault(info["ch_idx"], info["ch_name"])
        time_ids.add(info["t_idx"])

    ch_ids = sorted(ch_map.keys())
    t_ids = sorted(time_ids)
    return ch_map, ch_ids, t_ids


def get_dataset_channel_options(ds_path: str, glob_pattern: str = "*.tif"):
    """Return channel option dictionaries for UI selection widgets."""
    ch_map, ch_ids, t_ids = get_dataset_channel_info(ds_path, glob_pattern=glob_pattern)
    frame_counts = {ch_idx: 0 for ch_idx in ch_ids}

    files = sorted(glob.glob(str(Path(ds_path) / glob_pattern)))
    for fn in files:
        info = parse_stack_fn(fn)
        frame_counts[info["ch_idx"]] += 1

    options = []
    for ch_idx in ch_ids:
        ch_name = ch_map[ch_idx].strip("[]")
        options.append({
            "ch_idx": ch_idx,
            "ch_name": ch_name,
            "ch_tag": f"[{ch_name}] _C{ch_idx:02d}_Time",
            "n_frames": frame_counts[ch_idx],
            "n_times": len(t_ids),
        })
    return options
