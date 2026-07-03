# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in per-component control-loop timing for real-world teleop/rollout.

Diagnoses control-loop stalls — jerky or "lost-signal-for-~1s" teleop — by
timing each hot-path component separately (GELLO serial read, controller RPC,
camera grab) and logging any tick that blows past a threshold, plus a periodic
percentile summary. This isolates WHICH component stalls the 15 Hz loop (a late
velocity command makes the DROID controller ramp the arm down -> stutter).

Enabled by DEFAULT on this bench fork; set ``RLINF_STEP_TIMING=0`` to disable.
Tunables (env vars):
  RLINF_STEP_TIMING_THRESH_MS      per-tick "slow" threshold (default 150)
  RLINF_STEP_TIMING_SUMMARY_EVERY  steps between summaries   (default 150)

Output goes to stderr, which Ray forwards to the collect/eval driver log with
the ``(EnvGroup(rank=0) pid=...)`` prefix.
"""

import os
import sys
import time
from collections import defaultdict

_ENABLED = os.environ.get("RLINF_STEP_TIMING", "1") != "0"
_THRESH_MS = float(os.environ.get("RLINF_STEP_TIMING_THRESH_MS", "150"))
_SUMMARY_EVERY = int(os.environ.get("RLINF_STEP_TIMING_SUMMARY_EVERY", "150"))

_samples: "defaultdict[str, list]" = defaultdict(list)
_count = 0


class timed:
    """Context manager that records the wrapped block's duration under ``name``.

    ``with timed("gello_read"): ...`` — near-zero cost when disabled.
    """

    __slots__ = ("name", "_t0")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if not _ENABLED:
            return False
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        _samples[self.name].append(dt_ms)
        if dt_ms >= _THRESH_MS:
            sys.stderr.write(
                f"[STEP_TIMING] SLOW {self.name}={dt_ms:.0f}ms "
                f"(>={_THRESH_MS:.0f}ms) <- stall source this tick\n"
            )
            sys.stderr.flush()
        return False


def tick() -> None:
    """Call once per control step; emits a percentile summary every N steps."""
    global _count
    if not _ENABLED:
        return
    _count += 1
    if _count % _SUMMARY_EVERY:
        return
    import numpy as np

    parts = []
    for name in sorted(_samples):
        a = np.asarray(_samples[name][-_SUMMARY_EVERY:], dtype=float)
        if a.size == 0:
            continue
        parts.append(
            f"{name}: mean={a.mean():.0f} p95={np.percentile(a, 95):.0f} "
            f"max={a.max():.0f}ms slow={(a >= _THRESH_MS).sum()}"
        )
    sys.stderr.write(
        f"[STEP_TIMING] last {_SUMMARY_EVERY} steps | " + " | ".join(parts) + "\n"
    )
    sys.stderr.flush()
