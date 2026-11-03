<!-- This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
See LICENSE.txt in the project root; upstream components retain their applicable licenses. -->

# Pi05_A1Z

[中文](README.md) | English

**The original A1Z integration materials in this repository are licensed under the [Galaxea Dynamics A1Z LICENSE AGREEMENT](LICENSE.txt).** The [openpi source code](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE), π0.5 model weights, and other upstream components are governed by their respective license terms.

Built with Galaxea Dynamics A1Z

See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream licenses and the original copyright notice for collector patch context.

A1Z collection → LeRobot v2.1 → official openpi π0.5 JAX LoRA / full fine-tuning → PC inference.

**Scope: an end-to-end reference integration.** Single/dual LoRA training, checkpoint save/reload and the dual-arm online inference pipeline have been verified. Single-arm physical execution and full fine-tuning still need independent validation. Task success rates have not been systematically evaluated; results depend on data, scene and training settings.

Use `single` for one arm with head/wrist cameras, or `dual` for two arms with head/left-wrist/right-wrist cameras. Run commands in Linux Bash at the repository root. In each new terminal, enter the repository and activate the appropriate environment. Supply your own devices, task text and dataset paths.

## 1. System and clone

Use Ubuntu 22.04 x86_64. Ubuntu 22.04.5 LTS with Linux 6.8 HWE is recommended for the device host, with SocketCAN and V4L2 support. Validate drivers and simultaneous camera capture on other kernels. openpi uses a separate Python 3.11 / uv environment; collection and conversion use separate Conda environments.

| Use | Single-GPU memory guidance |
| --- | --- |
| π0.5 LoRA (JAX) | Upstream estimate: more than 22.5 GB; prepare at least 24 GB |
| π0.5 full fine-tuning (JAX) | Upstream estimate: more than 70 GB; prepare at least 80 GB |
| Inference | Upstream estimate: more than 8 GB; measure with the actual model, images and action horizon |

These are planning estimates, not guarantees. Allow at least 32 GB host RAM for LoRA or 64 GB for full fine-tuning, and disk space for data, base weights and multiple checkpoints. The pinned PyTorch implementation does not support LoRA; this project uses JAX throughout. See [openpi requirements](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md) for dependencies and drivers.

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential python3-venv ffmpeg tmux can-utils iproute2 nano
uname -r
nvidia-smi
git clone --branch main https://github.com/userguide-galaxea/a1z-pi05-deploy.git Pi05_A1Z
cd Pi05_A1Z
python3 scripts/bootstrap.py
python3 scripts/verify_workspace.py
```

Bootstrap initializes pinned `openpi`, `a1z-data-collect` and its nested `a1z-t/GALAXEA-A1Z`, then applies the collection integration patch. It does not download model weights. Do not independently update submodules to other revisions.

## 2. Install environments

Install Miniforge if Conda is unavailable; otherwise skip this block:

```bash
curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" -o /tmp/Miniforge3.sh
bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash
```

The device environment is shared by previews, collection and the robot client:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n pi05-robot python=3.11 pip -y
conda activate pi05-robot
export A1Z_PYTHON="$CONDA_PREFIX/bin/python"
bash scripts/setup_robot.sh
```

uv manages the model environment for statistics, training and policy serving:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
bash scripts/setup_model.sh
source openpi/.venv/bin/activate
python -c 'import jax; print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices())'
```

The conversion environment pins LeRobot 0.3.3, which writes v2.1, along with compatible NumPy / Arrow / Pandas versions. Do not install it into the openpi environment:

```bash
conda create -n pi05-convert python=3.11 pip -y
conda activate pi05-convert
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-convert.txt
python -c 'from importlib.metadata import version; import av; assert version("lerobot")=="0.3.3"; print(av.codec.Codec("h264","w").name)'
```

## 3. Preview, collect and inspect

data-collect / a1z-t is one supported collection method. You may use other collectors or existing data, provided the training input meets the [LeRobot v2.1 data contract](docs/data_contract.md). Existing compatible v2.1 datasets can proceed to validation in section 4 without recording or re-encoding videos.

Choose cameras that provide stable V4L2 RGB capture. Fix the head camera so it covers the operation and destination areas, and aim wrist cameras at the gripper contact area. Adjust exposure, gain and white balance to avoid blown highlights, reflections, shadows and motion blur. Keep views, resolution, lighting and object ranges consistent between collection and inference; recheck after reconnecting devices.

```bash
conda activate pi05-robot
export A1Z_PYTHON="$CONDA_PREFIX/bin/python"
export ARM_MODE=single   # Change to dual for two arms
mkdir -p "$HOME/a1z-local"
cp -n a1z-data-collect/a1z-t/teleop_app/config.yaml "$HOME/a1z-local/a1z-t.yaml"
cp -n "configs/collect_${ARM_MODE}.example.yaml" "$HOME/a1z-local/collect_${ARM_MODE}.yaml"
python scripts/check_devices.py --list-can
nano "$HOME/a1z-local/a1z-t.yaml"
nano "$HOME/a1z-local/collect_${ARM_MODE}.yaml"
export A1Z_COLLECT_CONFIG="$HOME/a1z-local/collect_${ARM_MODE}.yaml"
ip -details link show type can
ls -l /dev/serial/by-id/ /dev/v4l/by-path/
python scripts/preview_cameras.py --config "$A1Z_COLLECT_CONFIG"
# Close the preview before checking simultaneous continuous capture:
python scripts/preview_cameras.py --config "$A1Z_COLLECT_CONFIG" --check-only --duration 30
```

Fill in real task text, the output directory (prefer an absolute path), `task_name`, `num_episodes`, Leader serial ports, Follower CAN channels and camera paths. Set `leader_config_path` to the absolute path of your local a1z-t YAML, and select EVT / PVT for your hardware. Single mode uses `head` and `wrist`; dual mode keeps the template's three camera fields. Preview labels are Head / Wrist or Head / Left Wrist / Right Wrist. Dual collection requires two independent Leader serial ports.

`can0` / `can1` are interface names, not physical arm roles. With motors disabled, trace the cables and fill `hardware_checks.can_adapter_serials` from discovered adapters: `single` for single mode, or `left` / `right` for dual mode. Keep physical roles consistent between collection and inference. Empty or mismatched serials refuse startup. For hardware without a serial, explicitly use `null`, assign stable interface names and verify cabling manually. Software does not infer physical sides or swap channels automatically.

Cover each wrist lens to verify preview labels and check the actual color RGB stream. Do not identify cameras only by `/dev/videoN` or a multi-interface device's `by-id` link. Prefer a verified RGB interface's `by-path` and recheck after changing USB ports. Manage exposure with tools for your chosen camera; no camera model or exposure value is enforced.

CAN interfaces must be UP. If an interface needs configuration, first ensure no control program owns it, then use its actual name. Repeat for the second channel in dual mode:

```bash
read -r -p 'CAN interface (for example can0): ' A1Z_CAN
sudo ip link set "$A1Z_CAN" down
sudo ip link set "$A1Z_CAN" type can bitrate 1000000
sudo ip link set "$A1Z_CAN" up
```

For serial/video permission errors, `sudo usermod -aG dialout,video "$USER"` may be required; log out of the system and back in afterward. For missing interfaces or first-time Leader setup, follow the [a1z-t instructions](https://github.com/userguide-galaxea/a1z-t) for drivers, hardware versions and calibration.

Close previews and other control programs, confirm a clear workspace and an available emergency stop, then collect a few episodes before recording a larger set:

```bash
python scripts/check_devices.py --config "$A1Z_COLLECT_CONFIG"
bash scripts/collect.sh "$ARM_MODE" --check
bash scripts/collect.sh "$ARM_MODE"
```

`check_devices.py` reads CAN identity and 1 Mbit/s / UP status, then checks cameras with a deadline. It imports no arm SDK and sends no CAN frames. Collection repeats these checks before startup. Edit hardware settings in YAML; command-line hardware overrides are refused to prevent bypassing identity checks.

S starts, E saves, R discards and rerecords, and Q exits. Exit does not automatically home the arms. After saving, inspect one complete episode:

```bash
read -r -p 'Full path to a saved episode_N.hdf5: ' A1Z_EPISODE
python a1z-data-collect/tools/inspect_episode.py "$A1Z_EPISODE"
```

The inspector reads data and exports an MP4 for each camera; it does not control the robot. Confirm two streams for single or three for dual, correct view assignment, complete synchronized actions, real task text, 7D / 14D states and actions, joints in radians and grippers in `[0,1]`. Inspect early, middle and late episodes; keep failed recordings out of training.

## 4. Convert to LeRobot v2.1

This converter accepts only the project's HDF5 schema. Other HDF5, CSV, videos or newer LeRobot formats require an adapter to the [data contract](docs/data_contract.md), not a changed version label. If your v2.1 data already meets the contract, run only the final validation command.

```bash
conda activate pi05-convert
export ARM_MODE=single   # Or dual
read -r -p 'Directory of saved HDF5 episodes: ' HDF5_INPUT
read -r -p 'New v2.1 output directory: ' V21_OUTPUT
SINGLE_ARGS=(); [[ "$ARM_MODE" != single ]] || SINGLE_ARGS=(--single)
python scripts/convert_hdf5_to_v21.py --input "$HDF5_INPUT" --fps 15 --max-age 0.10 "${SINGLE_ARGS[@]}" --check-only
python scripts/convert_hdf5_to_v21.py --input "$HDF5_INPUT" --output "$V21_OUTPUT" --fps 15 --max-age 0.10 "${SINGLE_ARGS[@]}"
python scripts/check_v21_dataset.py "$V21_OUTPUT"
```

The converter validates all input timestamps and images before writing and checking the result. It refuses to overwrite directories; failed `.incomplete` outputs must not be used for training. Keep existing 15 Hz data at 15 Hz. `--max-age 0.10` is an alignment budget, not a way to hide sustained dropped frames. The π0.5 adapter reads the split v2.1 fields without changing source data.

## 5. Normalization and training

LoRA and full fine-tuning are independent choices that both start from the original JAX `pi05_base` parameters. Full fine-tuning does not require an earlier LoRA run. The first statistics pass fetches small resources such as the tokenizer; the first training run then downloads upstream base weights into `~/.cache/openpi` by default. Set `A1Z_BASE_PARAMS` to a local `params` directory if already available. G0.5 weights cannot initialize π0.5.

Use a tmux session so training continues after detaching:

```bash
tmux new -s pi05-train
# Inside the session, enter this repository and run:
source openpi/.venv/bin/activate
export ARM_MODE=single   # Or dual
export TRAIN_MODE=lora   # Change to full for full fine-tuning
read -r -p 'Validated v2.1 dataset directory: ' A1Z_DATASET
export A1Z_DATASET
bash scripts/check_data.sh "$ARM_MODE" "$TRAIN_MODE" --samples 5
bash scripts/train.sh "$ARM_MODE" "$TRAIN_MODE" --exp-name a1z_smoke --dry-run
bash scripts/norm.sh "$ARM_MODE" "$TRAIN_MODE"
bash scripts/train.sh "$ARM_MODE" "$TRAIN_MODE" --exp-name a1z_smoke --num-train-steps 10 --save-interval 10 --log-interval 1
```

`check_data.sh` reads real samples and checks image/action mappings without loading weights. Compute normalization statistics from your A1Z data, not another robot's statistics. Recompute when changing data, arm mode or delta settings. `norm.sh --max-frames N` is a quick check only; compute full statistics before a real run. Confirm finite loss, optimizer progress, successful checkpoint saving and reloading via section 7. Checkpoint directories use zero-based indices: 10 updates finish at `9/`. `--dry-run` prints configuration and does not train.

Statistics also save `pi05_a1z_norm.json`. Training checks dataset metadata, arm mode, delta settings, action horizon and the statistics file's SHA256; keep these files together. See the [data contract](docs/data_contract.md#6-统计与续训--statistics-and-resume) for changing the horizon and resuming training.

Default dataset IDs are `local/a1z_single` and `local/a1z_dual`. Before using another dataset directory for the same arm mode, set a new `A1Z_REPO_ID`, for example `export A1Z_REPO_ID=local/pick_cube_v2`, then repeat data checks and statistics. Launchers will not overwrite an existing binding to other data. Use a new `--exp-name` for another trial to preserve saved runs.

| Default | LoRA | Full fine-tuning |
| --- | --- | --- |
| Configuration | `a1z_single_lora` / `a1z_dual_lora` | `a1z_single_full` / `a1z_dual_full` |
| Updates / save interval | 20000 / 5000 | 30000 / 5000 |
| Batch size | 1 | 1 |
| Peak learning rate / warmup | 1e-4 / 1000 updates | 2.5e-5 / 1000 updates |
| Schedule / gradient clipping | Cosine / 1.0 | Cosine / 1.0 |
| Trainable parameters | Upstream Gemma LoRA configuration | All parameters, EMA 0.99 |
| LoRA rank / alpha | Vision-language model 16 / 16, action expert 32 / 32 | LoRA disabled |

These are starting points; G0.5 rank, accumulation and learning-rate settings do not transfer directly. All supplied episodes are used for training by default, with no validation split. Reserve separate evaluation data and consistent task tests. For a full run, choose one arm mode and one training method and use a new experiment name:

```bash
export EXP_NAME="a1z_${ARM_MODE}_${TRAIN_MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "logs/$EXP_NAME"
set -o pipefail
bash scripts/train.sh "$ARM_MODE" "$TRAIN_MODE" --exp-name "$EXP_NAME" 2>&1 | tee "logs/$EXP_NAME/train.log"
```

Press Ctrl+B then D to detach; return with `tmux attach -t pi05-train`. Outputs are under `checkpoints/a1z_<single|dual>_<lora|full>/<experiment>/<step>/`. Transfer the entire experiment directory, including `pi05_a1z_run.json` and the parameters / normalization assets inside each checkpoint, not only `params`.

The default action horizon is 16; the model pads inputs to 32 dimensions. Training converts joint targets to deltas relative to the current observation, and inference restores absolute targets. Grippers remain absolute `[0,1]`.

## 6. Live loss dashboard

In another terminal, read the training log:

```bash
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install flask
read -r -p 'Training log path: ' TRAIN_LOG
.venv-tools/bin/python scripts/loss_server.py "$TRAIN_LOG" --host 127.0.0.1 --port 8888
```

Open [http://127.0.0.1:8888](http://127.0.0.1:8888). It refreshes every 5 seconds; metrics absent from the log have no displayed value. For remote training, run this tunnel on your own computer and open the same address:

```bash
read -r -p 'Username@training-server: ' TRAIN_SSH
read -r -p 'SSH port (default 22): ' TRAIN_SSH_PORT
ssh -N -L 8888:127.0.0.1:8888 -p "${TRAIN_SSH_PORT:-22}" "$TRAIN_SSH"
```

Charts require browser access to a CDN; offline users can read `/data` JSON. The dashboard does not control training.

## 7. Serve a checkpoint and run inference

Stop training on the same GPU first. Match single / dual and lora / full to the saved run. Select a numeric checkpoint directory rather than an individual weight file, for example `checkpoints/a1z_single_lora/<experiment>/<step>`.

Terminal A: model environment; this process only computes actions.

```bash
source openpi/.venv/bin/activate
export ARM_MODE=single TRAIN_MODE=lora   # Match the trained configuration
read -r -p 'Saved checkpoint directory: ' MODEL_CKPT
bash scripts/server.sh "$ARM_MODE" "$TRAIN_MODE" "$MODEL_CKPT"
```

Wait until the model is loaded and listening on port 8000. Terminal B: model environment on the same PC, running the protocol bridge.

```bash
source openpi/.venv/bin/activate
export ARM_MODE=single   # Match the model
bash scripts/bridge.sh --mode "$ARM_MODE" --upstream-port 8000 --port 8765 --action-steps 16
```

Keep the default loopback address when the bridge and model share a PC. `--action-steps` must not exceed the checkpoint's prediction horizon (16 by default). The server automatically restores delta settings and the horizon from the run manifest. Terminal C: device environment.

```bash
conda activate pi05-robot
export A1Z_CLIENT_PYTHON="$CONDA_PREFIX/bin/python"
export ARM_MODE=single   # Match the model
mkdir -p "$HOME/a1z-local"
if [[ "$ARM_MODE" == single ]]; then TEMPLATE=pc-single; else TEMPLATE=pc-direct; fi
cp -n "configs/deployment/$TEMPLATE.example.json" "$HOME/a1z-local/client_${ARM_MODE}.json"
python scripts/check_devices.py --list-can
nano "$HOME/a1z-local/client_${ARM_MODE}.json"
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --check
python scripts/preview_cameras.py --config "$HOME/a1z-local/client_${ARM_MODE}.json"
# Close the preview, then check devices without enabling motors:
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --device-check
# The default client uses synthetic observations without hardware:
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json"
```

Set CAN channels, cameras, the training instruction and view assignment in the JSON. Use the actual `[width,height]` in `image_size`, the dataset FPS in `control_frequency`, and the bridge's port 8765 in `server_port`. Single mode uses `can_channel`, `camera_head` and `camera_wrist`; dual mode uses left/right CAN and three cameras. Bind CAN serials to the same physical roles used during collection, rather than copying interface numbers. Always preview before inference and match the collection setup.

Complete the default synthetic-observation inference first so initial JAX compilation and warmup finish before connecting the robot. This checks interfaces, not task performance.

With devices and the emergency stop ready, use real observations, then execute:

```bash
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --observe-only
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --execute
```

`--observe-only` starts the SDK and reads devices without sending model actions or starting the smoothing command thread. SDK initialization may still enable motors and home the gripper, so this is not a motion-free test. Use `--device-check` to avoid initializing the arms. Smoothing runs only with `--execute`. Templates initially send only 3 actions. After a supervised short test, set `max_cycles` to `0` for continuous operation and `smoothing.enabled` to `true` for direct smoothing to 250 Hz.

| Local parameter | Default and meaning |
| --- | --- |
| `max_initial_joint_step` / `max_joint_step` | 0.1 rad; first target relative to feedback / later adjacent-target limit |
| `gripper_boundary_tolerance` | 0.02; clip small excursions to `[0,1]`, reject larger excursions |
| `hardware_checks.camera_startup_timeout` / `camera_frame_timeout` | 12 / 1 seconds; camera startup and latest-frame freshness budgets |
| `smoothing.stale_timeout` | 1 second; hold without fresh actions, then investigate cameras, networking and inference latency |

Smoothing preserves reference position and velocity when targets change and brakes within the acceleration limit. An abrupt target inside the current stopping distance may briefly be passed; position and velocity are not snapped to the target. Limits apply to reference state on the integration clock, not measured joint motion. 250 Hz is the requested send rate; each step integrates at most 20 ms, so scheduler stalls are not replayed and add lag. `response_time` is not a guaranteed arrival time, and jerk is not limited. Grippers bypass the six-joint smoother. Verify actual send timing and joint feedback on hardware.

The client checks CAN identity and valid frames from every camera before initializing arms. Independent camera processes continuously retain the latest frames; startup timeouts or stale frames during operation raise an error and trigger cleanup. Ctrl+C cancels previews and `--device-check`. If a driver is blocked in the kernel, check USB cabling, power and the hub before retrying; avoid resetting shared hubs or repeatedly launching clients.

For a first-target limit error, check starting pose, role mapping, units and images first. `max_initial_joint_step` and `max_joint_step` are independent; do not repeatedly increase limits just to hide errors. Gripper `clipped` messages within tolerance indicate boundary clipping; larger excursions and nonfinite values are refused. Keep velocity, acceleration, joint and timeout protections. Smoothing to 250 Hz does not increase model observation frequency. Ctrl+C stops the client without automatic homing. For dual mode, confirm both arms stopped before stopping the bridge and model. Holding can still apply torque; disabling motors does not cut physical power. Restart the relevant processes after changing configuration or checkpoint.
