# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Stateless ONNX inference for the M20 SDK's 57-observation, 16-action interface."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


class OnnxM20Policy:
    """Evaluate fixed-batch SDK exports one robot at a time, or dynamic-batch exports together."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve(strict=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "Expected one observation input and one action output; recurrent policies are unsupported."
            )
        self.input, self.output = inputs[0], outputs[0]
        for entry, width in [(self.input, 57), (self.output, 16)]:
            if entry.type != "tensor(float)" or len(entry.shape) != 2 or entry.shape[1] != width:
                raise ValueError(f"Unexpected M20 interface: {entry.name} {entry.type} {entry.shape}")
            if isinstance(entry.shape[0], int) and entry.shape[0] != 1:
                raise ValueError("Only fixed batch=1 or dynamic batch ONNX exports are supported.")

    def __call__(self, observations):
        observation = observations["policy"]
        if observation.ndim != 2 or observation.shape[1] != 57:
            raise ValueError(f"M20 ONNX expects [num_envs, 57], got {tuple(observation.shape)}")
        array = observation.detach().to(device="cpu", dtype=torch.float32).numpy()
        if not np.isfinite(array).all():
            raise ValueError("Non-finite ONNX observations")
        if self.input.shape[0] == 1:
            actions = np.concatenate(
                [self.session.run([self.output.name], {self.input.name: row[None, :]})[0] for row in array], axis=0
            )
        else:
            actions = self.session.run([self.output.name], {self.input.name: array})[0]
        if actions.shape != (array.shape[0], 16) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid ONNX action output: {actions.shape}")
        return torch.from_numpy(actions).to(device=observation.device)

    def reset(self, dones):
        """The model is stateless; environment reset clears the last-action observation."""
