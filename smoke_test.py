"""Fast end-to-end checks for the canonical [[5,1,3]] pipeline."""

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def run(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / script), *arguments], check=True, cwd=ROOT)


def main():
    run("Perfect_tensor_1.py")
    local_path = ROOT / "perfect_tensor.npy"
    assert local_path.exists()
    tensor = np.load(local_path)
    assert tensor.shape == (2,) * 6
    encoder = np.transpose(tensor, (1, 2, 3, 4, 5, 0)).reshape(32, 2)
    assert encoder.shape == (32, 2)
    assert np.allclose(encoder.conj().T @ encoder, np.eye(2), atol=1e-8)

    run("Tensor_network_2.py")
    encoding_path = ROOT / "encoding_map.npy"
    assert encoding_path.exists()
    encoding = np.load(encoding_path)
    assert encoding.shape == (2048, 8)
    assert np.allclose(encoding.conj().T @ encoding, np.eye(8), atol=1e-8)

    run("Ansatz_training_3.py", "--quick")
    params = np.load(ROOT / "trained_params.npy")
    assert params.shape == (3, 11, 3)
    from Ansatz_training_3 import load_encoding, make_boundary_circuit, decoded_state, normalized_fidelity
    logical = np.ones(8, dtype=complex) / np.sqrt(8)
    circuit = make_boundary_circuit()
    decoded = decoded_state(params, logical, load_encoding(), circuit)
    assert decoded.shape == (8,)
    loss = 1 - normalized_fidelity(params, logical, load_encoding(), circuit)
    assert np.isfinite(float(loss))
    print("SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
