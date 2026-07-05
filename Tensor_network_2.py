"""Stage 2: contract three logical-first seed tensors into an encoding map."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TENSOR_PATH = ROOT / "perfect_tensor.npy"
NETWORK_PATH = ROOT / "network_tensor.npy"
ENCODING_PATH = ROOT / "encoding_map.npy"
EXPECTED_LOCAL_SHAPE = (2,) * 6


def assemble_network(tensor, atol=1e-8):
    if tensor.shape != EXPECTED_LOCAL_SHAPE:
        raise ValueError(
            f"Expected logical-first local tensor shape {EXPECTED_LOCAL_SHAPE}, got {tensor.shape}. "
            "Rerun Perfect_tensor_1.py to regenerate incompatible artifacts."
        )
    # Axis 0 is logical. Contract C.p0--L.p4 and C.p1--R.p4.
    central_left = np.tensordot(tensor, tensor, axes=([1], [5]))
    contracted = np.tensordot(central_left, tensor, axes=([1], [5]))
    # Raw open-axis order:
    # C.l,C.p2,C.p3,C.p4,L.l,L.p0..p3,R.l,R.p0..p3
    boundary_axes = (1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13)
    logical_axes = (0, 4, 9)
    network = np.transpose(contracted, boundary_axes + logical_axes)
    encoding = network.reshape(2**11, 2**3)
    gram = encoding.conj().T @ encoding
    scalar = np.trace(gram).real / 8
    if scalar <= 0 or not np.allclose(gram, scalar * np.eye(8), atol=atol):
        deviation = np.max(np.abs(gram - scalar * np.eye(8)))
        raise RuntimeError(f"Contracted map Gram matrix is not proportional to identity (deviation {deviation:.3e})")
    encoding = encoding / np.sqrt(scalar)
    network = encoding.reshape((2,) * 14)
    if not np.allclose(encoding.conj().T @ encoding, np.eye(8), atol=atol):
        raise RuntimeError("Normalized tensor-network map is not an isometry")
    return network, encoding, scalar


def main():
    if not TENSOR_PATH.exists():
        raise FileNotFoundError("perfect_tensor.npy not found; run Perfect_tensor_1.py first")
    tensor = np.load(TENSOR_PATH)
    network, encoding, scalar = assemble_network(tensor)
    np.save(NETWORK_PATH, network)
    np.save(ENCODING_PATH, encoding)
    print(f"Local tensor shape: {tensor.shape}")
    print("Number of logical qubits: 3")
    print("Number of boundary qubits: 11")
    print(f"Pre-normalization Gram scalar: {scalar:.6g}")
    print(f"Encoding map shape: {encoding.shape}")
    print("E^dagger E = I_8: PASS")
    print(f"Saved {NETWORK_PATH.name} and {ENCODING_PATH.name}")


if __name__ == "__main__":
    main()
