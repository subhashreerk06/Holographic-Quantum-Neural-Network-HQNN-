"""Stage 1: construct the logical-first six-leg [[5,1,3]] seed tensor."""

from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "perfect_tensor.npy"
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_all(operators):
    result = np.array([[1]], dtype=complex)
    for operator in operators:
        result = np.kron(result, operator)
    return result


def build_local_tensor(atol=1e-10):
    stabilizers = [
        kron_all([X, Z, Z, X, I]),
        kron_all([I, X, Z, Z, X]),
        kron_all([X, I, X, Z, Z]),
        kron_all([Z, X, I, X, Z]),
    ]
    identity = np.eye(32, dtype=complex)
    for index, stabilizer in enumerate(stabilizers):
        assert np.allclose(stabilizer, stabilizer.conj().T, atol=atol), f"S{index + 1} is not Hermitian"
        assert np.allclose(stabilizer @ stabilizer, identity, atol=atol), f"S{index + 1} does not square to I"
    for left, right in combinations(stabilizers, 2):
        assert np.allclose(left @ right, right @ left, atol=atol), "Stabilizers do not commute"

    projector = identity.copy()
    for stabilizer in stabilizers:
        projector = projector @ ((identity + stabilizer) / 2)
    assert np.allclose(projector, projector.conj().T, atol=atol)
    assert np.allclose(projector @ projector, projector, atol=atol)
    assert np.isclose(np.trace(projector), 2, atol=atol)
    assert np.linalg.matrix_rank(projector, tol=atol) == 2

    logical_z = kron_all([Z] * 5)
    logical_x = kron_all([X] * 5)
    values, vectors = np.linalg.eigh(projector)
    code_basis = vectors[:, values > 0.5]
    restricted_z = code_basis.conj().T @ logical_z @ code_basis
    z_values, z_vectors = np.linalg.eigh(restricted_z)
    order = np.argsort(z_values)[::-1]
    encoder = code_basis @ z_vectors[:, order]
    # Fix irrelevant column phases for deterministic artifacts.
    for column in range(2):
        pivot = np.argmax(np.abs(encoder[:, column]))
        encoder[:, column] *= np.exp(-1j * np.angle(encoder[pivot, column]))
    assert encoder.shape == (32, 2)
    assert np.allclose(encoder.conj().T @ encoder, np.eye(2), atol=atol)
    assert np.allclose(logical_z @ encoder[:, 0], encoder[:, 0], atol=atol)
    assert np.allclose(logical_z @ encoder[:, 1], -encoder[:, 1], atol=atol)
    assert np.allclose(np.abs(encoder.conj().T @ logical_x @ encoder), X, atol=atol)

    physical_first = encoder.reshape(2, 2, 2, 2, 2, 2)
    tensor = np.transpose(physical_first, (5, 0, 1, 2, 3, 4))
    local_matrix = np.transpose(tensor, (1, 2, 3, 4, 5, 0)).reshape(32, 2)
    assert tensor.shape == (2,) * 6
    assert np.allclose(local_matrix.conj().T @ local_matrix, np.eye(2), atol=atol)
    return tensor, projector


def perfect_bipartition_failures(tensor, atol=1e-8):
    failures = []
    for size in range(1, 4):
        for smaller_axes in combinations(range(6), size):
            larger_axes = tuple(axis for axis in range(6) if axis not in smaller_axes)
            matrix = np.transpose(tensor, smaller_axes + larger_axes).reshape(2**size, -1)
            gram = matrix @ matrix.conj().T
            scalar = np.trace(gram) / gram.shape[0]
            deviation = float(np.max(np.abs(gram - scalar * np.eye(gram.shape[0]))))
            if not np.allclose(gram, scalar * np.eye(gram.shape[0]), atol=atol):
                failures.append((smaller_axes, larger_axes, deviation))
    return failures


def main():
    tensor, projector = build_local_tensor()
    np.save(OUTPUT, tensor)
    failures = perfect_bipartition_failures(tensor)
    print("Stage 1: [[5,1,3]] five-qubit stabilizer code")
    print(f"Projector trace/rank: {np.trace(projector).real:.0f}/{np.linalg.matrix_rank(projector)}")
    print(f"Local tensor shape: {tensor.shape} (logical, p0, p1, p2, p3, p4)")
    print("Local encoder isometry: PASS")
    if failures:
        print("WARNING: perfect-tensor bipartition checks failed:")
        for small, large, deviation in failures:
            print(f"  {small} | {large}: max deviation {deviation:.3e}")
    else:
        print("All 41 smaller-side bipartition checks: PASS")
    print(f"Saved {OUTPUT.name}")


if __name__ == "__main__":
    main()
