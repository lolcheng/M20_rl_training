# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Export metadata from the live M20 configuration, in policy action order."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


def attach_metadata(path: Path, env, checkpoint):
    robot = env.scene["robot"]
    terms = [env.action_manager.get_term(n) for n in env.action_manager.active_terms]
    names = [n for term in terms for n in term._joint_names]
    indices = [robot.joint_names.index(n) for n in names]
    metadata = {
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "base_link_name": "base_link",
        "joint_names": ",".join(names),
        "default_joint_pos": ",".join(str(v) for v in robot.data.default_joint_pos.torch[0, indices].tolist()),
        "stiffness": ",".join(str(v) for v in ([80.0] * 12 + [0.0] * 4)),
        "damping": ",".join(str(v) for v in ([2.0] * 12 + [0.6] * 4)),
        "action_scale": ",".join(str(v) for v in ([0.125, 0.25, 0.25] * 4 + [5.0] * 4)),
        "action_types": ",".join(["position"] * 12 + ["velocity"] * 4),
    }
    # Keep A's deployment metadata names.
    metadata["joint_stiffness"] = metadata.pop("stiffness")
    metadata["joint_damping"] = metadata.pop("damping")
    model = onnx.load(str(path))
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def verify_onnx(path: Path, runner):
    model = runner.alg.get_policy().as_onnx(verbose=False).to("cpu").eval()
    inputs = model.get_dummy_inputs()
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    generator = torch.Generator().manual_seed(17)
    inputs = tuple(torch.randn(t.shape, generator=generator) for t in inputs)
    with torch.inference_mode():
        expected = model(*inputs).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    result = session.run(
        None, {entry.name: tensor.numpy() for entry, tensor in zip(session.get_inputs(), inputs, strict=True)}
    )[0]
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
    return {"max_abs_error": float(np.max(np.abs(result - expected))), "matched": True}
