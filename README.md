# Holographic Quantum Neural Network (HQNN)

This repository implements a small HaPPY-inspired tensor-network encoder seeded by the `[[5,1,3]]` five-qubit stabilizer code, followed by a trainable PennyLane boundary circuit. It is HaPPY-inspired rather than a full HaPPY code, and QEC-inspired rather than a syndrome-based error-correction decoder.

## Pipeline

1. `Perfect_tensor_1.py` constructs the standard five-qubit code from four stabilizer generators. Its encoding isometry has shape `(32, 2)` and becomes a logical-first six-leg tensor with shape `(2, 2, 2, 2, 2, 2)`: axis 0 is logical and axes 1–5 are physical.
2. `Tensor_network_2.py` contracts physical legs of one central and two leaf tensors. Three logical legs and eleven boundary legs remain, producing `E : C^8 -> C^2048` with array shape `(2048, 8)`. The script verifies `E†E = I_8`.
3. `Ansatz_training_3.py` trains an eleven-qubit boundary circuit and studies the effective logical map `E† U_boundary(theta) E`. Each of three layers applies RX, RY, and RZ to every boundary qubit, followed by valid CNOT pairs at distances `[1, 2, 4, 8]`. The parameter shape is `(3, 11, 3)`, for 99 parameters. The loss is one minus average logical-state fidelity.

## Run

```bash
python Perfect_tensor_1.py
python Tensor_network_2.py
python Ansatz_training_3.py --quick
python smoke_test.py
```

Remove `--quick` for the full Stage 3 training configuration. Generated artifacts are `perfect_tensor.npy`, `network_tensor.npy`, `encoding_map.npy`, `trained_params.npy`, and `training_curve.png`. Shape validation prevents Stage 3 from loading an incompatible encoding artifact.

The installable package under `src/pennylane_holographic/` exposes the same construction through a Python API and CLI. Run its test suite with `pytest`.
