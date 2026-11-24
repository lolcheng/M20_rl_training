# SDK ONNX reference

Original file: `/home/hanaki/DeepRobotics/sdk_deploy/src/M20S_sdk_deploy/policy/policy.onnx`.

Copied without alteration on 2026-11-07. SHA-256:
`4215c7513d6cb32ee3104872c96431c83feaa2fb9214b58f3bb159f92dc63eb7`.

The model has no custom metadata. Interface verification uses the sibling SDK
`run_policy/m20s_policy_runner.hpp`:

- Input `obs`: float32 `[1,57]`; output `actions`: float32 `[1,16]`.
- Observation: body angular velocity × 0.25, projected gravity, body velocity command,
  16 joint offsets with wheel positions zeroed, 16 joint velocities × 0.05, 16 previous actions.
- Joint order: FL leg, FR leg, HL leg, HR leg (hipx/hipy/knee within each), then FL/FR/HL/HR wheels.
- Front hipy/knee defaults: −0.3/+0.6 rad; hind defaults: +0.3/−0.6 rad.
- Leg position action scale: 0.125/0.25/0.25; wheel velocity action scale: 5.0.
- Nominal leg PD: 80/2; wheel PD: 0/0.6; policy period: 0.02 s.

The preview uses this project's M20 physics model. Matching the SDK interface does
not establish that the original policy was trained on this exact M20 asset.

See the project README for the direct `play.py --onnx` command.
