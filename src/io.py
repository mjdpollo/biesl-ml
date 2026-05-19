"""Load raw biesl recordings + parse filename metadata."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

CHANNEL_INDICES = {
    "mic":  (0, 1),
    "br":   (2, 3),
    "ecg":  (4, 5),
    "temp": (6, 7),
}

FNAME_RE = re.compile(
    r"^(?P<subject>[A-Za-z]+)-(?P<month>\d+)-(?P<day>\d+)-(?P<stressor>medi|math|pla)"
    r"(?:-(?P<plank>[^\.\s\(]+))?(?:\s*\((?P<rep>\d+)\))?\.csv$"
)


@dataclass
class Recording:
    path: str
    subject: str            # mta, nvt, ...
    date: str               # "5-17"
    stressor: str           # medi | math | pla
    plank_seconds: float | None  # only for pla; parsed from filename
    rep: int | None         # for "(1)" duplicates
    channels: dict[str, np.ndarray]   # name -> (time, value) stack 2xN

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def duration_s(self) -> float:
        return max(ch[0, -1] for ch in self.channels.values())


def parse_plank_duration(token: str | None) -> float | None:
    """Parse plank duration token from filename.

    Examples:
        "1'26" -> 86.0  (1 min 26 s)
        "2"    -> 120.0
        None   -> None
    """
    if token is None:
        return None
    token = token.strip()
    if "'" in token:
        mins, secs = token.split("'")
        return float(mins) * 60 + float(secs)
    # bare number = minutes
    return float(token) * 60


def parse_filename(path: str) -> tuple[str, str, str, float | None, int | None]:
    name = os.path.basename(path)
    m = FNAME_RE.match(name)
    if not m:
        raise ValueError(f"unparseable filename: {name}")
    subject = m.group("subject")
    date = f"{int(m.group('month'))}-{int(m.group('day'))}"
    stressor = m.group("stressor")
    plank = parse_plank_duration(m.group("plank"))
    rep = int(m.group("rep")) if m.group("rep") else None
    return subject, date, stressor, plank, rep


def load_recording(path: str) -> Recording:
    """Read a tab-separated recording by column INDEX (header names vary).

    Returns 4 channels as 2xN arrays of (time, value), NaNs dropped per channel.
    """
    df = pd.read_csv(path, sep="\t", header=0)
    if df.shape[1] != 8:
        raise ValueError(f"{path}: expected 8 columns, got {df.shape[1]}")

    channels: dict[str, np.ndarray] = {}
    for name, (ti, vi) in CHANNEL_INDICES.items():
        sub = df.iloc[:, [ti, vi]].dropna()
        t = sub.iloc[:, 0].to_numpy(dtype=np.float64)
        v = sub.iloc[:, 1].to_numpy(dtype=np.float64)
        # ensure strictly increasing time
        order = np.argsort(t, kind="mergesort")
        t = t[order]
        v = v[order]
        channels[name] = np.vstack([t, v])

    subject, date, stressor, plank, rep = parse_filename(path)
    return Recording(
        path=path,
        subject=subject,
        date=date,
        stressor=stressor,
        plank_seconds=plank,
        rep=rep,
        channels=channels,
    )


def list_recordings(data_dir: str) -> list[str]:
    return sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".csv")
    )


def phase_boundaries(rec: Recording, rest_s: float = 300.0) -> dict[str, tuple[float, float]]:
    """Return (start, end) in seconds for the three phases of a recording.

    rest:  [0, rest_s)
    stress: [rest_s, rest_s + stress_dur)
    recovery: [rest_s + stress_dur, end]

    stress_dur defaults to 300 s (medi/math) or the parsed plank duration (pla).
    Anything beyond the documented recovery window is included in recovery.
    """
    end = rec.duration_s
    if rec.stressor == "pla":
        if rec.plank_seconds is None:
            raise ValueError(f"missing plank duration for pla recording: {rec.name}")
        stress_dur = rec.plank_seconds
    else:
        stress_dur = 300.0

    rest_end = rest_s
    stress_end = rest_s + stress_dur
    if stress_end > end:
        # protocol overran; clip
        stress_end = end
    return {
        "rest":     (0.0, rest_end),
        "stress":   (rest_end, stress_end),
        "recovery": (stress_end, end),
    }


def channel_fs(ch: np.ndarray) -> float:
    """Median sampling rate of a 2xN (time, value) array."""
    t = ch[0]
    return float(1.0 / np.median(np.diff(t)))


def resample_uniform(ch: np.ndarray, fs_target: float) -> np.ndarray:
    """Resample a (time, value) channel onto a uniform grid at fs_target Hz.

    Returns 1D values; the grid starts at t[0] and is at 1/fs_target spacing.
    """
    t, v = ch[0], ch[1]
    n = int(np.floor((t[-1] - t[0]) * fs_target)) + 1
    grid = t[0] + np.arange(n) / fs_target
    return np.interp(grid, t, v)
