<!-- This file is part of Pi05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
See LICENSE.txt in the project root; upstream components retain their applicable licenses. -->

# A1Z → π0.5 数据契约 / Data contract

训练读取真正的 LeRobot v2.1，复用原始视频；数据适配器在读取时拼接状态与动作。其他采集方式或文件格式须先转换到以下结构，不能只改 `codebase_version`。

Training reads genuine LeRobot v2.1 and reuses its videos. The adapter joins state/action fields at read time. Other source formats must be converted to this contract; changing only a version label is insufficient.

## 1. 容器 / Container

```text
dataset/
├── meta/info.json                 # codebase_version: v2.1, actual fps, features
├── meta/tasks.jsonl               # real task text; task_index must resolve
├── meta/episodes.jsonl
├── meta/episodes_stats.jsonl
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/observation.images.<camera>/episode_000000.mp4
```

分块及文件名遵循 `meta/info.json` 的路径模板；包含完整 episode、连续帧索引、时间戳、正确的视频帧数。使用官方 v2.1 writer 生成元数据和统计。`INCOMPLETE` 或未写完的数据不能训练。

Follow the path templates in `meta/info.json`, with complete episodes, contiguous frame indices, timestamps and matching video frame counts. Use the official v2.1 writer for metadata/statistics. Incomplete datasets must not be used for training.

## 2. 字段与顺序 / Fields and order

| 模式 / Mode | 状态字段 / State fields | 动作字段 / Action fields | 拼接顺序 / Joined order |
| --- | --- | --- | --- |
| `single` | `observation.state.left_arm` (6), `observation.state.left_gripper` (1) | `action.left_arm` (6), `action.left_gripper` (1) | `[j1…j6, gripper]`，共 7 维 / 7D |
| `dual` | 上述字段＋ `observation.state.right_arm` (6), `observation.state.right_gripper` (1) | 上述字段＋ `action.right_arm` (6), `action.right_gripper` (1) | `[left j1…j6, left gripper, right j1…j6, right gripper]`，共 14 维 / 14D |

`single` 的 `left_*` 是固定数据字段名，不要求单臂安装在左侧。公开采集和客户端使用通用 head／wrist 命名，转换后保留上表字段；不要只改其中一端。

In `single`, `left_*` is a fixed schema name and does not require a physically left-mounted arm. Collection and client settings use generic head/wrist names; converted fields retain the schema above. Do not rename only one end of the mapping.

- 状态是从臂反馈，动作是对应时刻发送给从臂的绝对关节目标；不能把未适配的 Leader 原始舵机读数直接当作目标。State is follower feedback; action is the corresponding absolute follower target, not unconverted Leader servo readings.
- 关节为有限 float32 弧度，方向、顺序、零位与 A1Z SDK 一致。Joints are finite float32 radians with the SDK's order, signs and zero pose.
- 夹爪为绝对归一化值：`0=闭合/closed`、`1=张开/open`；不是角度或 0–100 百分数。Grippers use absolute normalized values, not angles or percentages.
- `task_index` 指向真实任务文本；不能为 `null` 或占位符。`task_index` must resolve to a real instruction, not `null` or placeholder text.

## 3. 相机 / Cameras

| 模式 / Mode | v2.1 字段 / Field | 含义 / View |
| --- | --- | --- |
| both | `observation.images.head_rgb` | 头部／全局 / Head or global |
| both | `observation.images.left_wrist_rgb` | single 腕部／dual 左腕 / Single wrist or dual left wrist |
| `dual` | `observation.images.right_wrist_rgb` | 右腕 / Right wrist |

视频须为三通道 RGB、分辨率稳定且与 metadata 一致。适配器支持读取后的 HWC uint8 `[0,255]` 或 CHW/HWC float `[0,1]`，模型变换负责尺寸处理；不要将 BGR、深度图或已二次归一化图像误当 RGB。single 缺少的第二腕视角会被遮罩，不复制另一张图充数。

Videos must be three-channel RGB with consistent dimensions matching metadata. The adapter accepts decoded HWC uint8 `[0,255]` or CHW/HWC float `[0,1]`; model transforms handle resizing. Do not pass BGR, depth or twice-normalized images. The unused wrist view in single mode is masked rather than duplicated.

## 4. 时间与动作表示 / Timing and action representation

保持数据实际 FPS，例如现有 15 Hz 数据；图像、状态和动作须按同一时间基准对齐。转换器使用过去最近样本，不读取未来图像填补掉帧。不能通过复制帧把采样频率标签改高。

Keep the actual dataset FPS, for example 15 Hz. Align images, state and action on a consistent time base. The converter selects the latest past sample instead of future images. Duplicating frames does not justify changing the FPS label.

默认训练窗口 16 步。适配器拼成 7／14 维后，将每个窗口内的关节目标减去窗口起点的当前观测关节；夹爪不作差。openpi 负责归一化和补齐到模型的 32 维。推理按相反顺序反归一化、恢复绝对关节目标并去除填充。

The default action horizon is 16. After joining 7D/14D fields, the adapter subtracts the window's current observed joints from each joint target; grippers remain absolute. openpi normalizes and pads to 32 dimensions. Inference reverses normalization and delta conversion, then removes padding.

`A1Z_DELTA_ACTIONS=0` 可切换为绝对关节训练；默认值为 `1`。统计与训练必须使用同一设置，模型服务自动从该次运行的 `pi05_a1z_run.json` 恢复，无须再手动设置。不要沿用 ALOHA 专用的关节符号和夹爪标定。

Set `A1Z_DELTA_ACTIONS=0` for absolute joint training; the default is `1`. Use the same setting for statistics and training. Serving automatically restores it from that run's `pi05_a1z_run.json`, so no manual setting is needed. Do not apply ALOHA-specific joint signs or gripper calibration.

## 5. 接入与检查 / Integration and checks

```bash
# 转换环境 / conversion environment
python scripts/check_v21_dataset.py /absolute/path/to/dataset
# 模型环境 / model environment
export A1Z_DATASET=/absolute/path/to/dataset
bash scripts/check_data.sh single lora --samples 5
bash scripts/train.sh single lora --exp-name a1z_check --dry-run   # 或 / or dual, full
bash scripts/norm.sh single lora
```

数据集路径由启动器绑定到本地缓存映射，原始数据不改动。数据、FPS、维度、相机映射或增量设置改变后重新计算统计量。统计和结构检查不能证明任务学习效果；正式长训前做少量优化步、保存和重载，并另留任务评估样本。

Launchers bind the dataset through a local cache mapping without modifying source files. Recompute statistics when data, FPS, dimensions, camera mapping or delta settings change. Structural checks and statistics do not establish task performance: first verify a short training run, checkpoint save/reload, and hold out separate task evaluation samples.

## 6. 统计与续训 / Statistics and resume

`norm.sh` 将 `norm_stats.json` 和 `pi05_a1z_norm.json` 保存于 `assets/<配置名>/<数据标识>/`。后者绑定数据元信息摘要、模式、数据标识、增量设置和动作窗口，并记录统计文件的 SHA256。训练前逐项核对；更换数据或改动预处理后重新统计，不手工修改签名。

`norm.sh` saves `norm_stats.json` and `pi05_a1z_norm.json` under `assets/<config>/<dataset-id>/`. The sidecar binds dataset metadata, mode, dataset ID, delta setting and action horizon, and records the statistics file's SHA256. Training verifies these before starting. Recompute after changing data or preprocessing; do not edit the signature manually.

默认窗口为 16，若改为 32，统计和训练必须同时修改；以下命令在模型环境、已设置 `A1Z_DATASET` 后执行。桥的 `--action-steps` 不得超过该模型窗口。

The default horizon is 16. To change it to 32, update both statistics and training. Run these in the model environment with `A1Z_DATASET` set. The bridge's `--action-steps` must not exceed the saved model horizon.

```bash
bash scripts/norm.sh single lora --action-horizon 32
bash scripts/train.sh single lora --exp-name a1z_h32 --model.action-horizon 32
```

续训添加 `--resume`，使用原实验名及原 `pi05_a1z_run.json`，并保持模型、数据、统计量、batch、seed、优化器和学习率调度等关键参数一致。可提高 `--num-train-steps`，但不会自动延长 cosine 调度；不能用续训把另一套配置接到原 checkpoint。迁移时保留整个实验目录及 checkpoint 内的 assets，服务会检查统计量并恢复已保存的模型／动作配置。

To resume, add `--resume`, retain the experiment name and its original `pi05_a1z_run.json`, and keep the model, data, statistics, batch size, seed, optimizer and learning-rate schedule unchanged. Increasing `--num-train-steps` does not automatically extend cosine decay. A different configuration cannot be attached to an existing run. Transfer the entire experiment directory and checkpoint assets; serving verifies statistics and restores the saved model/action configuration.
