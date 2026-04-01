# HQNN Stage 1: Perfect Tensor Construction
# File: Perfect_tensor_1.py
# Based on the [[6,2,3]] quantum error correcting code
#
# What this file does:
#   1. Builds 4 valid CSS stabilizer generators for the [[6,2,3]] code
#   2. Verifies they all commute with each other
#   3. Projects onto the code space (the +1 eigenspace of all stabilizers)
#   4. Extracts an isometric tensor from that code space via SVD
#   5. Verifies the perfect tensor property across ALL possible bipartitions
#   6. Saves perfect_tensor.npy to the SAME folder as this script

import numpy as np
import os
from itertools import combinations

# IMPORTANT: Save path is set to the folder containing THIS script file.
# This means perfect_tensor.npy will ALWAYS be created next to this file,
# regardless of which directory you run Python from.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH  = os.path.join(SCRIPT_DIR, "perfect_tensor.npy")

print("HQNN Stage 1: Perfect Tensor Construction")
print(f"  Script location : {SCRIPT_DIR}")
print(f"  Will save to    : {SAVE_PATH}")
print()

# SECTION 1: Pauli Matrices

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_chain(operators):
    """
    Tensor product of a list of operators.
    kron_chain([X, Z, I]) gives X⊗Z⊗I acting on 3 qubits.
    """
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result


# SECTION 2: Stabilizer Generators for [[6,2,3]]
#
# CSS code structure — X-type and Z-type generators.
# Every X-type generator overlaps every Z-type generator on an EVEN
# number of qubits, guaranteeing they all commute.
#
# 4 generators for n - k = 6 - 2 = 4.

# X-type stabilizers
S1 = kron_chain([X, X, X, X, I, I])   # X on qubits 1,2,3,4
S2 = kron_chain([X, X, I, I, X, X])   # X on qubits 1,2,5,6

# Z-type stabilizers
S3 = kron_chain([Z, Z, Z, Z, I, I])   # Z on qubits 1,2,3,4
S4 = kron_chain([Z, Z, I, I, Z, Z])   # Z on qubits 1,2,5,6

stabilizers = [S1, S2, S3, S4]

print("STEP 1: Validating stabilizers (Hermitian, S²=I)")
for i, S in enumerate(stabilizers):
    is_hermitian        = np.allclose(S, S.conj().T)
    squares_to_identity = np.allclose(S @ S, np.eye(64))
    print(f"  S{i+1}: Hermitian={is_hermitian}, S²=I: {squares_to_identity}")
print()

# SECTION 3: Verify All Stabilizers Commute

print("STEP 2: Checking all stabilizers commute")

all_commute = True
for i in range(len(stabilizers)):
    for j in range(i+1, len(stabilizers)):
        commutator = (stabilizers[i] @ stabilizers[j]
                      - stabilizers[j] @ stabilizers[i])
        commutes = np.allclose(commutator, np.zeros((64, 64)))
        status   = "✓" if commutes else "✗ FAIL"
        print(f"  [S{i+1}, S{j+1}] = 0 : {commutes} {status}")
        if not commutes:
            all_commute = False

if all_commute:
    print("  All stabilizers commute — code space is well defined ✓")
else:
    print("  WARNING: Some stabilizers do not commute!")
    print("  Code space is empty — fix the generators before continuing.")
print()

# SECTION 4: Build the Projector onto the Code Space
#
# (I + S)/2 projects onto the +1 eigenspace of each stabilizer.
# Multiplying all projectors gives the projector onto the code space.
# Each commuting stabilizer halves the remaining dimension:
#   64 → 32 → 16 → 8 → 4

print("STEP 3: Building projector onto the code space")

dim = 64
P   = np.eye(dim, dtype=complex)

for i, S in enumerate(stabilizers):
    P            = P @ ((np.eye(dim, dtype=complex) + S) / 2)
    current_rank = int(np.round(np.trace(P).real))
    print(f"  After S{i+1}: code space dimension = {current_rank}")

final_rank    = int(np.round(np.trace(P).real))
expected_rank = 4   # 2^k = 2^2 = 4 for [[6,2,3]]

print(f"  Final code space dimension = {final_rank}")
print(f"  Expected                  = {expected_rank}")

if final_rank == expected_rank:
    print("  Code space dimension correct ✓")
else:
    print("  ERROR: Wrong code space dimension — check generators!")
print()

# SECTION 5: Extract the Isometry via SVD
#
# SVD of P gives U · Σ · V†.
# Σ has exactly 4 singular values = 1 (code space) and 60 zeros.
# First 4 columns of U → isometry E : C⁴ → C⁶⁴

print("STEP 4: Extracting isometry via SVD")

U, singular_values, Vh = np.linalg.svd(P)

print(f"  Top 8 singular values: {np.round(singular_values[:8], 4)}")
print(f"  (Expect: four 1s then zeros)")

code_dim = 4
isometry = U[:, :code_dim]   # shape: (64, 4)

# verify the isometry condition E†E = I_4
EtE            = isometry.conj().T @ isometry
isometry_check = np.allclose(EtE, np.eye(code_dim), atol=1e-10)
print(f"  Isometry condition E†E = I₄ satisfied: {isometry_check}")

if isometry_check:
    print("  Isometry extracted successfully ✓")
print()

# SECTION 6: Reshape into Tensor Form
#
# isometry shape (64, 4) → tensor shape (2,2,2,2,2,2,2,2)
# Indices 0-5: physical qubits (boundary legs)
# Indices 6-7: logical qubits  (bulk legs)

print("STEP 5: Reshaping isometry into tensor form")

perfect_tensor = isometry.reshape(2, 2, 2, 2, 2, 2, 2, 2)

print(f"  Tensor shape: {perfect_tensor.shape}")
print(f"  Indices 0-5 → physical qubits (boundary legs)")
print(f"  Indices 6-7 → logical qubits  (bulk legs)")
print()

# SECTION 7: Save the Tensor before verification
#
# We save here unconditionally so that perfect_tensor.npy is ALWAYS
# created even if the perfect tensor verification below finds issues.
# Saving to SCRIPT_DIR ensures the file lands next to this .py file.

print("STEP 6: Saving tensor to disk")

np.save(SAVE_PATH, perfect_tensor)

if os.path.exists(SAVE_PATH):
    size_kb = os.path.getsize(SAVE_PATH) / 1024
    print(f"  Saved successfully: {SAVE_PATH}")
    print(f"  File size: {size_kb:.1f} KB")
else:
    print(f"  ERROR: File was not created at {SAVE_PATH}")
    print(f"  Check folder permissions.")
print()

# SECTION 8: Verify the Perfect Tensor Property
#
# For ANY bipartition of the 8 indices into sets A and B with |A| ≤ |B|,
# the reshaped matrix M_AB must be an isometry: M†M = I.
# We check all 162 bipartitions exhaustively.

print("STEP 7: Verifying perfect tensor — all bipartitions")

n_indices     = 8
T             = perfect_tensor
all_passed    = True
total_checks  = 0
failed_checks = []

for size_A in range(1, n_indices // 2 + 1):
    passes_at_this_size = 0
    fails_at_this_size  = 0

    for A in combinations(range(n_indices), size_A):
        B = tuple(i for i in range(n_indices) if i not in A)

        # rearrange tensor: A indices first, B indices last
        perm   = list(A) + list(B)
        T_perm = np.transpose(T, perm)

        # flatten into matrix
        dim_A = 2 ** len(A)
        dim_B = 2 ** len(B)
        M     = T_perm.reshape(dim_A, dim_B)

        # check isometry from smaller to larger space
        if dim_A <= dim_B:
            product     = M.conj().T @ M
            is_isometry = np.allclose(product, np.eye(dim_A), atol=1e-8)
        else:
            product     = M @ M.conj().T
            is_isometry = np.allclose(product, np.eye(dim_B), atol=1e-8)

        total_checks += 1

        if is_isometry:
            passes_at_this_size += 1
        else:
            fails_at_this_size += 1
            failed_checks.append((A, B))
            all_passed = False

    print(f"  |A|={size_A}: {passes_at_this_size} passed, "
          f"{fails_at_this_size} failed")

print(f"  Total bipartitions checked: {total_checks}")

if failed_checks:
    print(f"  First failed bipartition: A={failed_checks[0][0]}, B={failed_checks[0][1]}")
print()

if all_passed:
    print("  ✓ PERFECT TENSOR VERIFIED")
    print("  All bipartitions satisfy the isometry condition.")
    print("  Stage 1 complete — ready for Stage 2 (Tensor_network_2.py).")
else:
    print("  ✗ VERIFICATION FAILED")
    print(f"  {len(failed_checks)} bipartitions failed.")
    print("  These stabilizers do not produce a perfect tensor.")
    print("  The tensor has been saved anyway for inspection.")
    print("  A true perfect tensor requires carefully chosen generators")
    print("  — see the HaPPY code paper (Pastawski et al. 2015).")

print()
print(f"  perfect_tensor.npy saved to: {SAVE_PATH}")
print(f"  Load in Tensor_network_2.py with: T = np.load('perfect_tensor.npy')")
