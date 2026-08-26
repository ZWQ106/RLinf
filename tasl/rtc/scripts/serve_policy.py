#!/usr/bin/env python
"""RTC-capable drop-in for openpi/scripts/serve_policy.py — identical CLI.

    ~/work/openpi/.venv/bin/python ~/RLinf/tasl/rtc/scripts/serve_policy.py \
        --port 8000 policy:checkpoint --policy.config=pi05_droid_franka_lora \
        --policy.dir=$HOME/ckpts/pi05_droid_franka_lora_10task/16000

Reuses openpi's own Args / create_policy (imported from $OPENPI_DIR/scripts/
serve_policy.py, default ~/work/openpi) and wraps the policy in RTCPolicy, so
requests without an `rtc` key are served exactly as before and requests with
one get real-time-chunking guidance (see ../rtc_policy.py).

The file is deliberately named serve_policy.py under a scripts/ directory: the
dashboard's ServeManager finds / stops the server with the patterns
"scripts/serve_policy.py" and "serve_policy.py" (pgrep -f), which keep matching.

Env: RTC_WARMUP=0 skips the start-up compile of the guided sampler.
"""

import importlib.util
import logging
import os
import pathlib
import socket
import sys

TASL_DIR = pathlib.Path(__file__).resolve().parents[2]          # …/RLinf/tasl
# Default openpi checkout: <home>/work/openpi relative to THIS repo, not $HOME —
# the dashboard spawns us under sudo where ~ is /root.
OPENPI_DIR = pathlib.Path(os.environ.get("OPENPI_DIR") or (TASL_DIR.parents[1] / "work" / "openpi"))
sys.path.insert(0, str(TASL_DIR))

import tyro  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.policies import policy as _policy  # noqa: E402
from openpi.training import config as _config  # noqa: E402

from rtc.rtc_policy import RTCPolicy  # noqa: E402


def _load_openpi_serve_module():
    path = OPENPI_DIR / "scripts" / "serve_policy.py"
    spec = importlib.util.spec_from_file_location("openpi_scripts_serve_policy", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # tyro/inspect need the module registered to read Args' source
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    openpi_serve = _load_openpi_serve_module()
    args = tyro.cli(openpi_serve.Args)

    policy = openpi_serve.create_policy(args)
    if isinstance(args.policy, openpi_serve.Checkpoint):
        config_name = args.policy.config
    else:
        config_name = openpi_serve.DEFAULT_CHECKPOINT[args.env].config
    train_config = _config.get_config(config_name)
    policy = RTCPolicy(policy, model_config=train_config.model)
    logging.info("RTC-capable policy: H=%d action_dim=%d", policy.action_horizon, policy.action_dim)
    if os.environ.get("RTC_WARMUP", "1") != "0":
        policy.warmup()
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
