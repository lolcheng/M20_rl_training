# Deep Robotics M20 — Isaac Sim 6.2 / Isaac Lab 3.0

本工程将原 `rl_training` 的 M20 任务迁移到 Lab 3.0，以原工程当前源码和 USD 为行为基准。`robot_lab` 仅用于工程架构参考。

已提供 M20 Rough/Flat、RSL-RL 训练、checkpoint 播放、TorchScript/ONNX 导出，以及环境验证工具。使用 Isaac Sim PhysX，保留原随机延迟 PD、自定义奖励、地形课程和 PPO 参数。

**在本机训练**

独立 `.venv` 和 `uv.lock` 已创建。以下命令均在新工程目录运行；NVIDIA Kit 的 EULA 已在本次迁移中获得用户确认，首次运行仍需向进程传入接受标记。

```bash
cd /home/hanaki/DeepRobotics/rl_training_lab3.0
export OMNI_KIT_ACCEPT_EULA=YES
uv sync --locked

# Rough：默认 4096 个环境、20000 次迭代
uv run python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-v0 --headless --viz none

# Flat：默认 4096 个环境、5000 次迭代
uv run python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-Deeprobotics-M20-v0 --headless --viz none
```

可用 `--num_envs`、`--max_iterations`、`--seed`、`--run_name` 覆盖运行规模。用 `--checkpoint /绝对路径/model_N.pt` 恢复本工程新格式 checkpoint。日志和模型保存在 `logs/rsl_rl/deeprobotics_m20_{rough,flat}/<时间>/`。

保留 A 的任务 ID 与 Python 包名 `rl_training`；发行包名为 `rl-training-lab3`。请使用此工程的独立环境，避免与 A 的同名模块混装。无需在此环境安装 B。

**播放与导出**

```bash
# 使用新生成的模型。将 checkpoint 替换为实际路径。
uv run python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Rough-Deeprobotics-M20-v0 \
  --checkpoint /绝对路径/model_N.pt --num_envs 4 --viz kit --real-time

# 无窗口、有限步数的播放
uv run python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Rough-Deeprobotics-M20-v0 \
  --checkpoint /绝对路径/model_N.pt --num_envs 4 \
  --headless --viz none --max_steps 100
```

播放默认使用 A 的播放配置：50 个环境、Rough 地形缩减为 5×5、关闭策略观测噪声、按 A 原事件名称移除外力项。训练随机化不被播放设置影响。

每次播放会在 checkpoint 同级的 `exported/` 中生成 `policy.pt` 和 `policy.onnx`，并核对 ONNX 与 PyTorch 输出。ONNX 元数据中的关节顺序和默认姿态来自实际环境；修复了 A 快速导出脚本中的旧默认角度。动作先输出 12 个腿部位置目标，再输出 4 个轮速目标；动作缩放、PD 参数和动作类型一并写入元数据。

`--max_steps 0` 为持续播放，Ctrl+C 可退出。首版提供标准 Kit 视图和无窗口播放；A 的键盘控制、录像和足端调试绘图工具不在本次核心任务迁移范围内。

**直接播放 SDK 的 ONNX**

已将 `/home/hanaki/DeepRobotics/sdk_deploy/src/M20S_sdk_deploy/policy/policy.onnx` 原样复制到 `policies/sdk_m20/policy.onnx`，并核对 SDK 的 57 维观测、16 维动作、默认关节角和缩放与本工程 M20 接口一致。

```bash
uv run python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Flat-Deeprobotics-M20-v0 \
  --onnx policies/sdk_m20/policy.onnx \
  --num_envs 1 --command 0.5 0 0 --viz kit --real-time
```

`--command VX VY WZ` 是机身坐标系速度指令，单位依次为 m/s、m/s、rad/s：`0.5 0 0` 前进，`0 0.3 0` 侧移，`0 0 1` 原地转向，`0 0 0` 静止。省略该参数则沿用任务的随机指令采样；固定指令仅影响本次播放，不改训练配置。

将任务改为 `Rough-Deeprobotics-M20-v0` 可在粗糙地形预览。固定 batch=1 的 SDK ONNX 也支持多台机器人，程序分别推理每台机器人的观测。推理使用 CPU ONNX Runtime，仿真仍使用 GPU PhysX。直接播放 ONNX 不会再次导出或覆盖原 ONNX，也不需要 `.pt` checkpoint；`--checkpoint xxx.onnx` 同样可被识别。

该 ONNX 没有机器人元数据，使用上述接口是根据其 SDK runner 源码核对的。目录名虽然含 M20S，本命令加载的仿真模型仍是本工程 M20，不能据接口一致断定与原训练或 M20S 实机动力学完全一致。

**验证**

```bash
uv run pytest -q tests
uv run ruff check source scripts tests

uv run python scripts/tools/smoke_test.py \
  --task Rough-Deeprobotics-M20-v0 --num_envs 4 \
  --headless --viz none --max_steps 1100 --report outputs/smoke_rough.json
```

将任务名改为 `Flat-Deeprobotics-M20-v0` 可验证 Flat。工具检查有限值、部分重置、超时重置、动作/观测维度和实际执行器参数。

已在 RTX 5090 上完成两种任务的 1100 步环境验证、各 64 环境/2 次迭代训练、各 100 步 checkpoint 播放与导出；另验证了 Rough 默认 4096 环境初始化、步进和全部重置。详细结果见 [迁移记录](docs/MIGRATION.md) 与 [验证汇总](docs/validation/summary.json)。短训练只证明运行链路可用，不代表已收敛；示例 checkpoint 仅供验证。

**依赖与基准**

当前依赖本机 `/home/hanaki/Isaac/workspaces/isaaclab3/IsaacLab` 的源码，各组件路径在 `pyproject.toml` 的 `[tool.uv.sources]` 中声明。迁移到其他机器时先修改这些路径，再重新解析依赖。执行配置以本工程为准，不要将物理后端改为 Newton。

环境使用 Python 3.12、Isaac Sim 6.2.0.0、Lab 源码版本 3.0.0、RSL-RL 5.5.1、Torch 2.11.0+cu130。组件包独立版本化，`isaaclab` 的发行包版本 28.0.0 不代表整套 Lab 主版本。

A 的 M20 模型完整复制到本工程 `deep_robotics_model/M20/`。原始源码及模型的提交、文件校验值和公式数值参照保存在 `docs/baseline/`。

本次仅迁移 M20。原 A、B 与上游 IsaacLab 源码均未修改。保留 A 的 Flat `gait_level` 初始为 0 的现有行为；其他机器人、AMP、CusRL、跨旧版本权重恢复、多 GPU 和长期收敛尚未覆盖。
