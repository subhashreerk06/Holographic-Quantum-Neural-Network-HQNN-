"""Stage 3: train an 11-qubit boundary circuit through E† U(theta) E."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

ROOT = Path(__file__).resolve().parent
ENCODING_PATH = ROOT / "encoding_map.npy"
PARAMS_PATH = ROOT / "trained_params.npy"
CURVE_PATH = ROOT / "training_curve.png"
N_LOGICAL, N_BOUNDARY, N_LAYERS = 3, 11, 3
PARAMS_SHAPE = (N_LAYERS, N_BOUNDARY, 3)
DISTANCES = (1, 2, 4, 8)


def load_encoding(path=ENCODING_PATH, atol=1e-8):
    if not Path(path).exists():
        raise FileNotFoundError("encoding_map.npy not found; run Stage 1 and Stage 2 first")
    encoding = np.load(path)
    if encoding.shape != (2048, 8):
        raise ValueError(
            f"Expected encoding_map.npy shape (2048, 8), got {encoding.shape}. "
            "This may be an old incompatible artifact; rerun Perfect_tensor_1.py and Tensor_network_2.py."
        )
    if not np.allclose(encoding.conj().T @ encoding, np.eye(8), atol=atol):
        raise ValueError("encoding_map.npy is not an isometry; rerun Stage 1 and Stage 2")
    return encoding


def random_logical_states(count, rng):
    states = rng.normal(size=(count, 8)) + 1j * rng.normal(size=(count, 8))
    return states / np.linalg.norm(states, axis=1, keepdims=True)


def make_boundary_circuit():
    device = qml.device("default.qubit", wires=N_BOUNDARY)

    @qml.qnode(device, interface="autograd")
    def circuit(params, state):
        qml.StatePrep(state, wires=range(N_BOUNDARY))
        for layer in range(N_LAYERS):
            for qubit in range(N_BOUNDARY):
                qml.RX(params[layer, qubit, 0], wires=qubit)
                qml.RY(params[layer, qubit, 1], wires=qubit)
                qml.RZ(params[layer, qubit, 2], wires=qubit)
            for distance in DISTANCES:
                for qubit in range(N_BOUNDARY - distance):
                    qml.CNOT(wires=(qubit, qubit + distance))
        return qml.state()

    return circuit


def decoded_state(params, logical_state, encoding, circuit):
    boundary_state = encoding @ logical_state
    final_boundary_state = circuit(params, boundary_state)
    return encoding.conj().T @ final_boundary_state


def normalized_fidelity(params, logical_state, encoding, circuit):
    decoded = decoded_state(params, logical_state, encoding, circuit)
    norm = pnp.linalg.norm(decoded)
    # Projection through E† need not preserve norm when U leaves the code space.
    safe_norm = pnp.maximum(norm, 1e-12)
    normalized = decoded / safe_norm
    overlap = pnp.sum(pnp.conj(logical_state) * normalized)
    return pnp.real(pnp.abs(overlap) ** 2)


def train(encoding_np, quick=False, seed=42):
    rng = np.random.default_rng(seed)
    count, iterations = (1, 1) if quick else (20, 100)
    logical_states = pnp.array(random_logical_states(count, rng), requires_grad=False)
    encoding = pnp.array(encoding_np, requires_grad=False)
    circuit = make_boundary_circuit()
    params = pnp.array(rng.uniform(-0.1, 0.1, PARAMS_SHAPE), requires_grad=True)

    def loss(candidate):
        fidelities = [normalized_fidelity(candidate, state, encoding, circuit) for state in logical_states]
        return 1 - pnp.mean(pnp.stack(fidelities))

    initial_decoded = decoded_state(params, logical_states[0], encoding, circuit)
    if initial_decoded.shape != (8,):
        raise RuntimeError(f"Expected decoded state shape (8,), got {initial_decoded.shape}")
    initial_norm = float(pnp.linalg.norm(initial_decoded))
    if initial_norm < 1e-6:
        raise RuntimeError(f"Decoded-state norm is severely small ({initial_norm:.3e})")

    optimizer = qml.AdamOptimizer(stepsize=0.05)
    history = []
    for _ in range(iterations):
        params, cost = optimizer.step_and_cost(loss, params)
        history.append(float(cost))
    return np.asarray(params), history, initial_decoded


def save_curve(history):
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(range(1, len(history) + 1), history)
    axis.set(xlabel="Iteration", ylabel="Loss (1 - average fidelity)", title="Boundary-circuit training")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(CURVE_PATH, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use one state and one optimizer step")
    args = parser.parse_args(argv)
    encoding = load_encoding()
    params, history, decoded = train(encoding, quick=args.quick)
    np.save(PARAMS_PATH, params)
    save_curve(history)
    print(f"Encoding map shape: {encoding.shape}")
    print("Logical qubits: 3; boundary qubits: 11")
    print(f"Parameter shape: {params.shape}; trainable parameters: {params.size}")
    print(f"One decoded state shape: {decoded.shape}")
    print(f"Final loss: {history[-1]:.6f}")
    print(f"Saved {PARAMS_PATH.name} and {CURVE_PATH.name}")


if __name__ == "__main__":
    main()
