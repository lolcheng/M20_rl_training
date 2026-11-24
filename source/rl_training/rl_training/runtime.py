# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""M20 entrypoints using Lab 3.0 launchers and A's training settings."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import warp as wp

from isaaclab_tasks.utils import resolve_task_config


def _arguments(mode: str):
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=f"{mode.title()} the original Deep Robotics M20 task on Lab 3.0.")
    parser.add_argument(
        "--task", default="Rough-Deeprobotics-M20-v0", choices=["Rough-Deeprobotics-M20-v0", "Flat-Deeprobotics-M20-v0"]
    )
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--seed", type=int)
    policy_source = parser.add_mutually_exclusive_group()
    policy_source.add_argument("--checkpoint")
    policy_source.add_argument("--onnx", help="Play an M20 ONNX model directly, without an RSL-RL checkpoint.")
    parser.add_argument(
        "--command",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "WZ"),
        help="Fixed playback velocity: m/s, m/s, rad/s. Omit to use the task command sampler.",
    )
    parser.add_argument("--run_name", default="")
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--max_steps", type=int, default=0 if mode == "play" else 1100)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--real_time", "--real-time", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Compatibility alias for --viz none.")
    add_launcher_args(parser)
    args, overrides = parser.parse_known_args()
    if mode != "play" and (args.onnx or args.command is not None):
        parser.error("--onnx and --command are playback options.")
    if mode == "play" and args.checkpoint and Path(args.checkpoint).suffix.lower() == ".onnx":
        args.onnx, args.checkpoint = args.checkpoint, None
    if args.headless and args.visualizer is None:
        args.visualizer = []
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=overrides)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    elif mode == "play":
        env_cfg.scene.num_envs = 50
    if args.seed is not None:
        agent_cfg.seed = args.seed
    env_cfg.seed = agent_cfg.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if mode == "play":
        apply_play_configuration(env_cfg)
        if args.command is not None:
            apply_fixed_command(env_cfg, args.command)
    # Explicit observation sets preserve A's asymmetric policy/critic inputs.
    agent_cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    if type(env_cfg.sim.physics).__module__.split(".")[0] != "isaaclab_physx":
        raise ValueError("This migration is validated only with Isaac Sim PhysX.")
    return args, env_cfg, agent_cfg


def apply_play_configuration(cfg):
    """Apply the original A play.py overrides, preserving its event names exactly."""
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.max_init_terrain_level = None
    cfg.observations.policy.enable_corruption = False
    for name in ("randomize_apply_external_force_torque", "push_robot"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    cfg.curriculum.command_levels = None


def apply_fixed_command(cfg, command):
    """Use exact SDK-style body velocity commands, including small commands without thresholding."""
    from isaaclab.envs.mdp import UniformVelocityCommandCfg

    if not all(torch.isfinite(torch.tensor(command))):
        raise ValueError("Velocity command must be finite.")
    vx, vy, wz = command
    cfg.commands.base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        heading_command=False,
        rel_heading_envs=0.0,
        rel_standing_envs=0.0,
        debug_vis=cfg.commands.base_velocity.debug_vis,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(vx, vx), lin_vel_y=(vy, vy), ang_vel_z=(wz, wz)),
    )


def _finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), "Non-finite simulation output"
    elif hasattr(value, "values"):
        for item in value.values():
            _finite(item)


def _write_report(path, report):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


def run(mode: str):
    """Run training, policy playback, or a bounded environment validation."""
    from isaaclab.app import launch_simulation

    wp.config.enable_backward = False
    args, cfg, agent_cfg = _arguments(mode)
    # A's exact backend flags; the upstream unified trainer changes benchmark to True.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    with launch_simulation(cfg, args):
        import gymnasium as gym

        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))

        env = gym.make(args.task, cfg=cfg)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            if mode == "train":
                suffix = f"_{args.run_name}" if args.run_name else ""
                log = (
                    Path("logs/rsl_rl")
                    / agent_cfg.experiment_name
                    / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + suffix)
                )
                log.mkdir(parents=True, exist_ok=True)
                dump_yaml(str(log / "params/env.yaml"), cfg)
                dump_yaml(str(log / "params/agent.yaml"), agent_cfg)
                runner = create_rsl_rl_runner(wrapped, agent_cfg, log_dir=str(log.resolve()))
                if args.checkpoint:
                    runner.load(args.checkpoint)
                runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
                _write_report(
                    args.report,
                    {
                        "mode": mode,
                        "task": args.task,
                        "log_dir": str(log.resolve()),
                        "iterations": agent_cfg.max_iterations,
                        "num_envs": wrapped.num_envs,
                    },
                )
            elif mode == "play":
                if args.onnx:
                    from .onnx_policy import OnnxM20Policy

                    policy = OnnxM20Policy(args.onnx)
                    policy_report = {
                        "onnx_model": str(policy.path),
                        "input_shape": policy.input.shape,
                        "output_shape": policy.output.shape,
                    }
                elif args.checkpoint:
                    runner = create_rsl_rl_runner(wrapped, agent_cfg)
                    runner.load(args.checkpoint)
                    policy = runner.get_inference_policy(device=wrapped.device)
                    output = Path(args.checkpoint).resolve().parent / "exported"
                    runner.export_policy_to_jit(str(output))
                    runner.export_policy_to_onnx(str(output))
                    from .export import attach_metadata, verify_onnx

                    attach_metadata(output / "policy.onnx", env.unwrapped, args.checkpoint)
                    export_check = verify_onnx(output / "policy.onnx", runner)
                    policy_report = {"exported": str(output), "onnx": export_check}
                else:
                    raise ValueError("--checkpoint or --onnx is required for playback.")
                obs = wrapped.get_observations()
                step = 0
                resets = 0
                with torch.inference_mode():
                    while args.max_steps == 0 or step < args.max_steps:
                        start = time.monotonic()
                        obs, reward, done, _ = wrapped.step(policy(obs))
                        policy.reset(done)
                        resets += int(done.sum())
                        _finite(obs)
                        _finite(reward)
                        step += 1
                        if args.real_time:
                            time.sleep(max(0, env.unwrapped.step_dt - (time.monotonic() - start)))
                _write_report(
                    args.report,
                    {
                        "mode": mode,
                        "task": args.task,
                        "steps": step,
                        "resets": resets,
                        "command": args.command,
                        "finite": True,
                        **policy_report,
                    },
                )
            else:
                report = validate_environment(wrapped, args.max_steps)
                report.update(task=args.task, mode=mode)
                _write_report(args.report, report)
        finally:
            env.close()


def validate_environment(wrapped, steps):
    """Exercise stateful rewards, full/partial resets, command sampling and one timeout."""
    env = wrapped.unwrapped
    robot = env.scene["robot"]
    obs = wrapped.get_observations()
    report = {
        "num_envs": env.num_envs,
        "steps": steps,
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
        "action_names": list(env.action_manager.active_terms),
        "action_dimensions": list(env.action_manager.action_term_dim),
        "physics": type(env.cfg.sim.physics).__name__,
        "dt": env.step_dt,
        "use_newton_actuators": env.cfg.sim.use_newton_actuators,
        "active_rewards": list(env.reward_manager.active_terms),
        "terminations": list(env.termination_manager.active_terms),
        "default_joint_pos": robot.data.default_joint_pos.torch[0].tolist(),
        "actuators": {},
    }
    for name, actuator in robot.actuators.items():
        report["actuators"][name] = {
            "class": type(actuator).__name__,
            "min_delay": actuator.cfg.min_delay,
            "max_delay": actuator.cfg.max_delay,
            "effort_limit": actuator.cfg.actuator_effort_limit,
            "solver_effort_limit": actuator.cfg.joint_effort_limit,
            "armature": actuator.cfg.armature,
            "stiffness_range": [float(actuator.stiffness.min()), float(actuator.stiffness.max())],
            "damping_range": [float(actuator.damping.min()), float(actuator.damping.max())],
            "delay_values": actuator.positions_delay_buffer.time_lags.unique().tolist(),
        }
        nominal_stiffness = actuator.cfg.stiffness
        nominal_damping = actuator.cfg.damping
        assert torch.all(actuator.stiffness >= nominal_stiffness * 0.85 - 1e-5)
        assert torch.all(actuator.stiffness <= nominal_stiffness * 1.15 + 1e-5)
        assert torch.all(actuator.damping >= nominal_damping * 0.85 - 1e-5)
        assert torch.all(actuator.damping <= nominal_damping * 1.15 + 1e-5)
        assert set(report["actuators"][name]["delay_values"]) <= {0, 1}
    resets = 0
    with torch.inference_mode():
        for step in range(steps):
            action = torch.zeros(env.num_envs, wrapped.num_actions, device=env.device)
            if step % 3 == 1:
                action.uniform_(-0.2, 0.2)
            elif step % 3 == 2:
                action[:, -4:] = 0.1
            obs, reward, done, _ = wrapped.step(action)
            _finite(obs)
            _finite(reward)
            _finite(robot.data.root_pos_w.torch)
            resets += int(done.sum())
            if step == min(100, steps // 2):
                env._reset_idx(torch.arange(0, env.num_envs, 2, device=env.device))
        # Force and observe the timeout path, independently of earlier falls.
        env.episode_length_buf[:] = env.max_episode_length - 1
        _, _, done, _ = wrapped.step(torch.zeros(env.num_envs, wrapped.num_actions, device=env.device))
        report["forced_timeout_resets"] = int(done.sum())
    report["observed_resets"] = resets
    report["finite"] = True
    return report
