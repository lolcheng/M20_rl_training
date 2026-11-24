# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""A's independent inertia randomization using Lab 3.0 asset writers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import RigidObject


class randomize_rigid_body_inertia(ManagerTermBase):
    """Keep the original inertia baseline and independently sample Ixx, Iyy and Izz."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        # Event terms are constructed before startup mass randomization executes.
        self.default_inertia = self.asset.data.body_inertia.torch.clone()

    def __call__(self, env, env_ids, asset_cfg, inertia_distribution_params, operation, distribution="uniform"):
        if env_ids is None:
            env_ids = slice(None)
        body_ids = asset_cfg.body_ids
        if not isinstance(body_ids, slice):
            body_ids = torch.as_tensor(body_ids, device=self.asset.device, dtype=torch.long)
        if not isinstance(env_ids, slice):
            env_ids = torch.as_tensor(env_ids, device=self.asset.device, dtype=torch.long)
        rows = env_ids[:, None] if not isinstance(env_ids, slice) and not isinstance(body_ids, slice) else env_ids
        inertias = self.default_inertia[rows, body_ids].clone()
        sample = {
            "uniform": math_utils.sample_uniform,
            "log_uniform": math_utils.sample_log_uniform,
            "gaussian": math_utils.sample_gaussian,
        }[distribution]
        for diagonal in (0, 4, 8):
            values = sample(*inertia_distribution_params, inertias.shape[:2], device=self.asset.device)
            if operation == "scale":
                inertias[:, :, diagonal] *= values
            elif operation == "add":
                inertias[:, :, diagonal] += values
            elif operation == "abs":
                inertias[:, :, diagonal] = values
            else:
                raise ValueError(f"Unknown inertia operation: {operation}")
        self.asset.set_inertias_index(inertias=inertias, body_ids=body_ids, env_ids=env_ids)


def bad_orientation_2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # type: ignore
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return (asset.data.projected_gravity_b.torch[:, 2] > 0) | (
        asset.data.projected_gravity_b.torch[:, :2].abs() > 0.7
    ).any(-1)
