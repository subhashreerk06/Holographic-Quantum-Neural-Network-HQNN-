# Holographic-Quantum-Neural-Network-HQNN-

# HQNN

A research-stage implementation of a **Holographic Quantum Neural Network (HQNN)** pipeline built from perfect tensors, a small hyperbolic tensor network, and a trainable boundary quantum circuit.

This repository is currently an **ongoing prototype**, not a finished framework. At the moment it contains three connected stages:

1. **Perfect tensor construction** from a stabilizer-code-inspired setup
2. **Hyperbolic tensor network assembly** into an encoding map
3. **Boundary ansatz training** for quantum state reconstruction

The pipeline implemented in the current codebase is:

```text
|psi_out> = E† · U_boundary(theta) · E · |psi_in>
```

where:
- `E` is the encoding map produced by the tensor network
- `U_boundary(theta)` is a trainable boundary circuit
- `E†` decodes back to the logical space

The first training task is **state reconstruction**: the model is trained so that `|psi_out> ≈ |psi_in>`.

---

## What this repo currently does

### Stage 1 — `Perfect_tensor_1.py`
Builds a candidate perfect tensor from a `[[6,2,3]]` CSS stabilizer code construction.

It currently:
- defines Pauli operators and stabilizer generators
- checks Hermiticity and stabilizer commutation
- projects onto the code space
- extracts an isometry via SVD
- reshapes the isometry into an 8-leg tensor
- checks the perfect-tensor/isometry condition across bipartitions
- saves `perfect_tensor.npy`

### Stage 2 — `Tensor_network_2.py`
Builds a small **3-node holographic tensor network** from the tensor saved in Stage 1.

It currently:
- loads `perfect_tensor.npy`
- constructs a graph with one central tensor and two leaf tensors
- assigns index labels for contractions
- contracts the network using `opt_einsum`
- separates boundary and bulk/logical legs
- reshapes the result into an encoding map `E`
- verifies the network isometry `E†E = I`
- saves `network_tensor.npy` and `encoding_map.npy`

### Stage 3 — `Ansatz_training_3.py`
Adds a **trainable boundary quantum circuit** using PennyLane and trains it on random logical states.

It currently:
- loads `encoding_map.npy`
- generates random normalized logical states
- encodes them into the boundary Hilbert space
- defines a layered boundary ansatz with single-qubit rotations and multi-scale CNOT connectivity
- trains the circuit using Adam on the loss `1 - average fidelity`
- evaluates training and simple generalization performance
- saves `trained_params.npy` and `training_curve.png`

---


## Scientific idea

This code explores a simple holographic learning pipeline inspired by:
- **quantum error correcting codes**
- **perfect tensors**
- **holographic tensor networks / HaPPY-style ideas**
- **variational quantum circuits**



---

## How to run

Run the files in order.

### 1) Build the tensor
```bash
python Perfect_tensor_1.py
```

This produces:
- `perfect_tensor.npy`

### 2) Assemble the network
```bash
python Tensor_network_2.py
```

This produces:
- `network_tensor.npy`
- `encoding_map.npy`

### 3) Train the boundary circuit
```bash
python Ansatz_training_3.py
```

This produces:
- `trained_params.npy`
- `training_curve.png`

---




