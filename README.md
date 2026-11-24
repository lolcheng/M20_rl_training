# Deep Robotics M20 — Isaac Sim 6.2 / Isaac Lab 3.0

将 M20 任务迁移到 Lab 3.0。

已提供 M20 Rough/Flat、RSL-RL 训练、checkpoint 播放、TorchScript/ONNX 导出，以及环境验证工具。使用 Isaac Sim PhysX，保留原随机延迟 PD、自定义奖励、地形课程和 PPO 参数。

**在本机训练**

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

用 `--num_envs`、`--max_iterations`、`--seed`、`--run_name` 修改运行规模。用 `--checkpoint /绝对路径/model_N.pt` 恢复 checkpoint。
日志和模型保存在 `logs/rsl_rl/deeprobotics_m20_{rough,flat}/<时间>/`。


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

播放默认使用 50 个环境、Rough 地形缩减为 5×5、关闭策略观测噪声、移除外力项。

每次播放会在 checkpoint 同级的 `exported/` 中生成 `policy.pt` 和 `policy.onnx`。

`--max_steps 0` 为持续播放，Ctrl+C 退出。

