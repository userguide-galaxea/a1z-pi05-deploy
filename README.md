<!-- This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
See LICENSE.txt in the project root; upstream components retain their applicable licenses. -->

# Pi05_A1Z

中文 | [English](README.en.md)

**本仓库自有的 A1Z 集成内容采用 [Galaxea Dynamics A1Z LICENSE AGREEMENT](LICENSE.txt) 许可。** [openpi 源码](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/LICENSE)、π0.5 模型权重及其他上游组件适用各自的许可条款。

Built with Galaxea Dynamics A1Z

上游许可与采集补丁的原始版权声明见 [第三方许可说明](THIRD_PARTY_NOTICES.md)。

A1Z 采集 → LeRobot v2.1 → 官方 openpi π0.5 JAX LoRA／全量微调 → PC 推理。

**定位：端到端链路参考适配。** 已验证 single／dual LoRA 训练、checkpoint 保存与重载，以及双臂在线推理链路。单臂真机执行和全量训练仍需独立验证；尚未系统评估任务成功率，效果取决于数据、场景和训练设置。

支持 `single`（单臂、头部＋腕部相机）和 `dual`（双臂、头部＋左腕＋右腕相机）。以下命令在 Linux Bash、仓库根目录执行；新终端重新进入仓库并激活对应环境。设备路径、任务文本和数据路径需自行填写。

## 1. 系统与克隆

使用 Ubuntu 22.04 x86_64；设备端建议 Ubuntu 22.04.5 LTS／Linux 6.8 HWE，需支持 SocketCAN 和 V4L2。其他内核需验证驱动与多路相机通信。openpi 使用独立 Python 3.11／uv 环境，采集和转换分别使用 Conda 环境。

| 用途 | 单卡显存准备建议 |
| --- | --- |
| π0.5 LoRA（JAX） | 官方估计大于 22.5 GB，建议从 24 GB 起准备 |
| π0.5 全量微调（JAX） | 官方估计大于 70 GB，建议从 80 GB 起准备 |
| 推理 | 官方估计大于 8 GB，按实际模型、图像与动作窗口短测峰值 |

上述不是显存保证；主机内存建议 LoRA 32 GB、全量 64 GB 起，磁盘需同时容纳数据、基模和多个 checkpoint。固定版本的 PyTorch 路径不支持 LoRA，本项目统一使用 JAX。依赖和驱动要求见 [openpi](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md)。

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

bootstrap 初始化固定版本的 `openpi`、`a1z-data-collect` 及其内嵌 `a1z-t/GALAXEA-A1Z`，并应用采集集成补丁；不下载模型权重。不要单独将子模块更新到其他版本。

## 2. 安装环境

没有 Conda 时先安装 Miniforge；已有则跳过这一块：

```bash
curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" -o /tmp/Miniforge3.sh
bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash
```

设备环境用于预览、采集和真机客户端：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n pi05-robot python=3.11 pip -y
conda activate pi05-robot
export A1Z_PYTHON="$CONDA_PREFIX/bin/python"
bash scripts/setup_robot.sh
```

模型环境由 uv 管理，训练、统计量计算和模型服务共用：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
bash scripts/setup_model.sh
source openpi/.venv/bin/activate
python -c 'import jax; print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices())'
```

转换环境固定 LeRobot 0.3.3，其输出为 v2.1；同时固定兼容的 NumPy／Arrow／Pandas，不要装进 openpi 环境：

```bash
conda create -n pi05-convert python=3.11 pip -y
conda activate pi05-convert
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-convert.txt
python -c 'from importlib.metadata import version; import av; assert version("lerobot")=="0.3.3"; print(av.codec.Codec("h264","w").name)'
```

## 3. 预览、采集与数据检查

data-collect／a1z-t 是本项目提供的一种采集方式，也可以使用自有程序和已有数据；训练输入统一为 [LeRobot v2.1 数据契约](docs/data_contract.md)。已有合规 v2.1 数据可跳到第 4 节校验，无须重新采集或编码视频。

相机自行选型，当前入口读取 V4L2 RGB。固定机位，让头部视角覆盖操作区和目标区域，腕部能看清夹爪接触区；调整曝光、增益和白平衡，避免过曝、反光、阴影和拖影。采集与推理保持视角、分辨率、光照和目标范围一致；重插设备后复查。

```bash
conda activate pi05-robot
export A1Z_PYTHON="$CONDA_PREFIX/bin/python"
export ARM_MODE=single   # 双臂改为 dual
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
# 关闭预览后检查多路连续读取：
python scripts/preview_cameras.py --config "$A1Z_COLLECT_CONFIG" --check-only --duration 30
```

在 YAML 填写真实任务文本、保存目录（建议绝对路径）、`task_name`、`num_episodes`、Leader 串口、Follower CAN、相机路径；`leader_config_path` 填本地 a1z-t YAML 的绝对路径，按硬件选择 EVT／PVT。single 使用 `head`、`wrist`，dual 保留模板的三路字段；预览分别显示 Head／Wrist 和 Head／Left Wrist／Right Wrist。双臂需要两个独立 Leader 串口。

`can0`／`can1` 是接口名，不代表左右臂。设备未使能时，沿接线确认物理对应，将发现的串号填入 `hardware_checks.can_adapter_serials`：single 用 `single`，dual 用 `left`／`right`，必须与采集和推理的角色一致。串号为空或不匹配会拒绝启动。无串号的 CAN 硬件可显式设为 `null`，此时需自行固定接口名并核实接线。程序不会自动识别物理左右，也不会自动交换通道。

预览时遮挡腕部镜头确认标签，检查实际彩色 RGB 通道；不要只按 `/dev/videoN` 编号或多接口设备的 `by-id` 链接选择。优先使用已核对 RGB 接口的 `by-path` 路径，换 USB 端口后重查。曝光设置由所选相机的工具管理，本项目不强制某个型号或曝光值。

CAN 必须 UP；接口未配置时，确认未被控制程序占用，按实际接口执行（双臂对第二路重复）：

```bash
read -r -p 'CAN 接口（例如 can0）: ' A1Z_CAN
sudo ip link set "$A1Z_CAN" down
sudo ip link set "$A1Z_CAN" type can bitrate 1000000
sudo ip link set "$A1Z_CAN" up
```

串口或相机权限不足时可用 `sudo usermod -aG dialout,video "$USER"`，注销系统并重新登录。接口不存在或首次使用 Leader 时，先按 [a1z-t 说明](https://github.com/userguide-galaxea/a1z-t) 完成驱动、硬件版本和校准检查。

关闭预览和其他控制程序，现场确认运动范围及急停后开始；先采少量检查，再批量采集：

```bash
python scripts/check_devices.py --config "$A1Z_COLLECT_CONFIG"
bash scripts/collect.sh "$ARM_MODE" --check
bash scripts/collect.sh "$ARM_MODE"
```

`check_devices.py` 只读核对 CAN 身份、1 Mbit/s／UP 状态并限时检查相机，不导入机械臂 SDK、不发送 CAN 帧。采集启动前会重复检查；硬件设置应在 YAML 中修改，不能用命令行覆盖绕过校验。

按 S 开始、E 保存、R 丢弃重录、Q 退出；退出不自动回零。保存后抽查一条完整 episode：

```bash
read -r -p '已保存的 episode_N.hdf5 完整路径: ' A1Z_EPISODE
python a1z-data-collect/tools/inspect_episode.py "$A1Z_EPISODE"
```

该工具只读数据并导出各路 MP4，不控制机械臂。确认 single 两路／dual 三路齐全、对应正确、操作完整且同步；任务文本真实，状态／动作 7／14 维，关节弧度、夹爪 `[0,1]`。检查首批及批量数据的开头、中间、结尾；失败片段不混入训练。

## 4. 转换为 LeRobot v2.1

本转换器只接受本项目 HDF5 结构。其他 HDF5、CSV、视频或新版 LeRobot 数据需先按 [数据契约](docs/data_contract.md) 适配，不能只改版本标签。已符合契约的 v2.1 数据直接执行最后一条校验命令。

```bash
conda activate pi05-convert
export ARM_MODE=single   # 或 dual
read -r -p '已保存的 HDF5 数据目录: ' HDF5_INPUT
read -r -p '新的 v2.1 输出目录: ' V21_OUTPUT
SINGLE_ARGS=(); [[ "$ARM_MODE" != single ]] || SINGLE_ARGS=(--single)
python scripts/convert_hdf5_to_v21.py --input "$HDF5_INPUT" --fps 15 --max-age 0.10 "${SINGLE_ARGS[@]}" --check-only
python scripts/convert_hdf5_to_v21.py --input "$HDF5_INPUT" --output "$V21_OUTPUT" --fps 15 --max-age 0.10 "${SINGLE_ARGS[@]}"
python scripts/check_v21_dataset.py "$V21_OUTPUT"
```

先校验全部输入时间戳和图像，再转换并检查产物；不覆盖旧目录，失败的 `.incomplete` 目录不能训练。15 Hz 数据先保持 15 Hz；`--max-age 0.10` 是采样对齐预算，不能用来掩盖持续掉帧。π0.5 适配器直接读取 v2.1 拆分字段，不改原数据。

## 5. 统计量与训练

LoRA 和全量是独立方案，都从原始 `pi05_base` JAX 参数开始，不需要先 LoRA 再全量。首次统计会取得 tokenizer 等小资源，首次训练再下载官方基模，默认缓存 `~/.cache/openpi`；已有基模可设置 `A1Z_BASE_PARAMS` 为本地 `params` 目录。G0.5 权重不能用于 π0.5。

在 tmux 会话中执行，脱离后任务继续：

```bash
tmux new -s pi05-train
# 在会话内进入本仓库，然后执行：
source openpi/.venv/bin/activate
export ARM_MODE=single   # 或 dual
export TRAIN_MODE=lora   # 全量改为 full
read -r -p '已校验的 v2.1 数据目录: ' A1Z_DATASET
export A1Z_DATASET
bash scripts/check_data.sh "$ARM_MODE" "$TRAIN_MODE" --samples 5
bash scripts/train.sh "$ARM_MODE" "$TRAIN_MODE" --exp-name a1z_smoke --dry-run
bash scripts/norm.sh "$ARM_MODE" "$TRAIN_MODE"
bash scripts/train.sh "$ARM_MODE" "$TRAIN_MODE" --exp-name a1z_smoke --num-train-steps 10 --save-interval 10 --log-interval 1
```

`check_data.sh` 只读真实样本、检查相机和动作映射，不加载权重。统计量必须由本次 A1Z 数据计算，不能复用其他机器人的统计；更换数据、模式或增量动作设置后重新计算。`norm.sh --max-frames N` 仅用于快速检查，正式训练前计算完整统计量。试跑需确认 loss 有限、完成优化步、保存成功，并按第 7 节重载。checkpoint 目录从 0 计数：10 次更新的最终目录是 `9/`。`--dry-run` 只打印配置，不能代替试跑。

统计同时保存 `pi05_a1z_norm.json`，训练会核对数据元信息、模式、增量设置、动作窗口及统计文件 SHA256；不要分开替换统计文件。修改动作窗口和续训要求见 [数据契约](docs/data_contract.md#6-统计与续训--statistics-and-resume)。

默认数据标识为 `local/a1z_single`／`local/a1z_dual`。同一模式更换数据目录时，先设置新的 `A1Z_REPO_ID`（例如 `export A1Z_REPO_ID=local/pick_cube_v2`），再重做检查和统计；启动器不会覆盖已绑定的其他数据。重跑试验时使用新 `--exp-name`，避免覆盖已保存的训练。

| 默认设置 | LoRA | 全量 |
| --- | --- | --- |
| 配置名 | `a1z_single_lora`／`a1z_dual_lora` | `a1z_single_full`／`a1z_dual_full` |
| 更新步数／保存间隔 | 20000／5000 | 30000／5000 |
| batch size | 1 | 1 |
| 峰值学习率／warmup | 1e-4／1000 步 | 2.5e-5／1000 步 |
| 调度／梯度裁剪 | cosine／1.0 | cosine／1.0 |
| 参数更新 | 官方 Gemma LoRA 配置 | 全部参数，EMA 0.99 |
| LoRA rank／alpha | 视觉语言模型 16／16，动作专家 32／32 | 不使用 LoRA |

这些是适配起点，不能照搬 G0.5 的 rank、梯度累积或学习率参数。默认使用全部传入 episode，不划分验证集；效果需另留评估数据及相同场景的任务测试。正式训练使用新实验名，选择一种模式和一种方法：

```bash
export EXP_NAME="a1z_${ARM_MODE}_${TRAIN_MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "logs/$EXP_NAME"
set -o pipefail
bash scripts/train.sh "$ARM_MODE" "$TRAIN_MODE" --exp-name "$EXP_NAME" 2>&1 | tee "logs/$EXP_NAME/train.log"
```

Ctrl+B 后按 D 脱离；`tmux attach -t pi05-train` 返回。产物位于 `checkpoints/a1z_<single|dual>_<lora|full>/<实验名>/<步数>/`。迁移时保留整个实验目录，包含运行配置 `pi05_a1z_run.json` 以及 checkpoint 内的参数和 normalization assets，不要只拷 `params`。

默认动作窗口 16 步、模型内部补齐到 32 维；训练适配将关节转为相对当前观测的增量，推理还原绝对目标，夹爪保持绝对 `[0,1]`。

## 6. 实时 loss 图

另开终端，只读取日志：

```bash
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install flask
read -r -p '训练日志路径: ' TRAIN_LOG
.venv-tools/bin/python scripts/loss_server.py "$TRAIN_LOG" --host 127.0.0.1 --port 8888
```

打开 [http://127.0.0.1:8888](http://127.0.0.1:8888)，每 5 秒刷新；日志没有的指标不显示数值。远程机器在自己的电脑建立隧道后访问同一地址：

```bash
read -r -p '用户名@训练服务器: ' TRAIN_SSH
read -r -p 'SSH 端口（默认 22）: ' TRAIN_SSH_PORT
ssh -N -L 8888:127.0.0.1:8888 -p "${TRAIN_SSH_PORT:-22}" "$TRAIN_SSH"
```

图表需浏览器可访问 CDN；离线可读 `/data` JSON。看板不控制训练。

## 7. 启动模型与推理

先停止同卡训练。使用与训练一致的 single／dual、lora／full，选择该次保存的 checkpoint 目录，不是单个权重文件。例如 `checkpoints/a1z_single_lora/<实验名>/<步数>`。

终端 A（模型环境，只计算动作）：

```bash
source openpi/.venv/bin/activate
export ARM_MODE=single TRAIN_MODE=lora   # 按实际训练组合修改
read -r -p '已保存的 checkpoint 目录: ' MODEL_CKPT
bash scripts/server.sh "$ARM_MODE" "$TRAIN_MODE" "$MODEL_CKPT"
```

等待模型加载并监听 8000。终端 B（同一 PC 的模型环境，协议桥）：

```bash
source openpi/.venv/bin/activate
export ARM_MODE=single   # 与模型一致
bash scripts/bridge.sh --mode "$ARM_MODE" --upstream-port 8000 --port 8765 --action-steps 16
```

桥与模型在同一 PC 时保持默认本机地址；`--action-steps` 不得超过 checkpoint 的预测窗口（默认 16）。模型服务从运行配置自动恢复增量设置和动作窗口。终端 C（设备环境）：

```bash
conda activate pi05-robot
export A1Z_CLIENT_PYTHON="$CONDA_PREFIX/bin/python"
export ARM_MODE=single   # 与模型一致
mkdir -p "$HOME/a1z-local"
if [[ "$ARM_MODE" == single ]]; then TEMPLATE=pc-single; else TEMPLATE=pc-direct; fi
cp -n "configs/deployment/$TEMPLATE.example.json" "$HOME/a1z-local/client_${ARM_MODE}.json"
python scripts/check_devices.py --list-can
nano "$HOME/a1z-local/client_${ARM_MODE}.json"
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --check
python scripts/preview_cameras.py --config "$HOME/a1z-local/client_${ARM_MODE}.json"
# 关闭预览，再限时核对设备（不使能机械臂）：
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --device-check
# 默认客户端使用合成观测，不打开硬件：
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json"
```

JSON 填 CAN、相机、训练时的任务文本和视角，`image_size` 填实际 `[宽,高]`，`control_frequency` 等于数据 FPS，`server_port` 使用桥的 8765。single 用 `can_channel`、`camera_head`、`camera_wrist`；dual 用左右 CAN 和三路相机。填写与采集时物理角色相同的 CAN 串号绑定，不要照抄接口编号；推理前预览必须与采集一致。

先完成默认模式的合成观测推理，让首次 JAX 编译与预热在连接机械臂前结束；这只能检查接口，不能验证模型动作效果。

现场确认设备和急停后，再使用真实观测并执行：

```bash
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --observe-only
bash scripts/client.sh --config "$HOME/a1z-local/client_${ARM_MODE}.json" --execute
```

`--observe-only` 会启动 SDK 并读取设备，不发送模型动作，也不启动平滑发送线程；SDK 初始化仍可能使能电机、执行夹爪归零，不能当作完全不动的测试。完全不启用机械臂时使用 `--device-check`。平滑仅在 `--execute` 生效。模板先只发 3 条动作；短测通过后在本地 JSON 设置 `max_cycles: 0` 持续运行、`smoothing.enabled: true` 启用直接到 250 Hz 的平滑。

| 本地参数 | 默认与含义 |
| --- | --- |
| `max_initial_joint_step`／`max_joint_step` | 0.1 rad，首帧相对反馈／后续相邻目标的变化限制 |
| `gripper_boundary_tolerance` | 0.02，容差内裁剪到 `[0,1]`，更大越界拒绝 |
| `hardware_checks.camera_startup_timeout`／`camera_frame_timeout` | 12／1 秒，相机启动与最新帧新鲜度预算 |
| `smoothing.stale_timeout` | 1 秒，无新动作后保持；先排查相机、网络及推理延迟 |

平滑在切换目标时保留参考位置与速度，按加速度上限制动；目标突然进入当前制动距离时，可能短暂越过新目标，不强行贴位置或清零速度。速度／加速度限制针对积分时钟上的参考状态，不保证真实关节运动；250 Hz 是期望发送频率，每步最多积分 20 ms，调度卡顿后不补跑但会增加滞后。`response_time` 不是保证到达时间，也不提供 jerk 限制；夹爪不经过六关节平滑。实机需核对发送周期与关节反馈。

客户端先核对 CAN 身份并取得全部相机有效帧，再初始化机械臂。相机在独立进程持续采集最新帧；启动超时或运行中帧过期会报错并进入清理。预览和 `--device-check` 可用 Ctrl+C 中止；如果驱动卡在内核，检查 USB 线、供电及 Hub 后重试，不重置共享 Hub 或反复启动客户端。

首帧超限时先核对起始姿态、左右映射、单位和图像；`max_initial_joint_step` 与 `max_joint_step` 独立，不应只为消除报错连续调大。夹爪在容差内的 `clipped` 提示是边界截断，更大越界或非有限值仍拒绝。保留速度、加速度、关节和超时保护；250 Hz 平滑不增加模型观测频率。Ctrl+C 停止客户端，无自动回零；双臂确认两侧停止结果，再停止桥和模型服务。保持状态可能仍有力矩，电机禁用不等于切断物理电源。配置或 checkpoint 改变后重新启动相应进程。
