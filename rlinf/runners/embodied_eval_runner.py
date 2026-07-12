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

import os
import time
import typing

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        rollout_handle.wait()
        env_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        # Persistent "warm policy" mode: when RLINF_EVAL_GATE_FILE is set (the
        # TASL eval dashboard sets it), keep the process — Ray cluster, env
        # workers, and the loaded model — alive across episodes and run one
        # episode per operator "Start", instead of exiting after a single
        # evaluate(). Model load + Ray init happen ONCE in init_workers(); each
        # episode is just another evaluate() (the same call training runs at
        # every val_check_interval, so repeated calls are a supported path).
        # Without the env var, behavior is unchanged: one evaluate(), then exit.
        gate_path = os.environ.get("RLINF_EVAL_GATE_FILE")
        if gate_path:
            self._run_gated(gate_path)
        else:
            self._run_once(step=0)
            self.metric_logger.finish()

    def _run_once(self, step: int):
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=step, data=eval_metrics)

    @staticmethod
    def _write_gate(path: str, state: str) -> None:
        # Sentinel FILE (not stdout): Ray buffers actor stdout, so the dashboard
        # reads this file (docker exec cat) to drive the Start button reliably.
        try:
            with open(path, "w") as f:
                f.write(state)
        except Exception:
            pass

    @staticmethod
    def _read_gate(path: str) -> typing.Optional[str]:
        try:
            with open(path) as f:
                return f.read().strip() or None
        except OSError:
            return None

    def _run_gated(self, gate_path: str):
        """Warm-policy loop: wait at the gate, run one episode per RUN, repeat.

        The dashboard writes 'RUN' to start the next episode; we flip the gate
        to 'BUSY' while the episode runs and back to 'WAIT' when it finishes and
        the arm has homed (env.evaluate resets at the start of each pass). The
        process only exits when killed (Stop / eval-stop.sh)."""
        episode = 0
        self.logger.info(
            "EVAL_READY: model loaded; warm-policy mode — waiting for Start "
            f"(gate={gate_path})"
        )
        while True:
            self._write_gate(gate_path, "WAIT")
            print("WAIT_FOR_START: press Start to run the next eval episode",
                  flush=True)
            while self._read_gate(gate_path) != "RUN":
                time.sleep(0.1)
            self._write_gate(gate_path, "BUSY")
            print(f"EPISODE_START: running eval episode {episode}", flush=True)
            self._run_once(step=episode)
            print(f"EPISODE_END: eval episode {episode} done", flush=True)
            episode += 1
