
# HQNN Stage 1: Perfect Tensor Construction
# Based on the [[6,2,3]] quantum error correcting code

# What this file does:
#   1. Builds 4 valid CSS stabilizer generators for the [[6,2,3]] code
#   2. Verifies they all commute with each other
#   3. Projects onto the code space (the +1 eigenspace of all stabilizers)
#   4. Extracts an isometric tensor from that code space via SVD
#   5. Verifies the perfect tensor property across ALL possible bipartitions

import numpy as np
from itertools import combinations

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
# We use a CSS (Calderbank-Shor-Steane) code structure.
# CSS codes have two types of generators: X-type and Z-type.
# The key constraint is: every X-type generator must overlap with
# every Z-type generator on an EVEN number of qubits.
# Even overlap → they commute → valid code space exists.
#
# Here we use 4 generators (n - k = 6 - 2 = 4), which is the correct
# number for encoding 2 logical qubits into 6 physical qubits.
#
# X-type generators (detect Z errors):
#   S1: X on qubits 1,2,3,4  — support size 4
#   S2: X on qubits 1,2,5,6  — support size 4
#
# Z-type generators (detect X errors):
#   S3: Z on qubits 1,2,3,4  — same pattern as S1 (overlap = 4, even)
#   S4: Z on qubits 1,2,5,6  — same pattern as S2 (overlap = 4, even)
#
# Cross-overlaps:
#   S1 vs S4: overlap on {1,2} — size 2, even
#   S2 vs S3: overlap on {1,2} — size 2, even

# X-type stabilizers
S1 = kron_chain([X, X, X, X, I, I])   # X on qubits 1,2,3,4
S2 = kron_chain([X, X, I, I, X, X])   # X on qubits 1,2,5,6

# Z-type stabilizers
S3 = kron_chain([Z, Z, Z, Z, I, I])   # Z on qubits 1,2,3,4
S4 = kron_chain([Z, Z, I, I, Z, Z])   # Z on qubits 1,2,5,6

stabilizers = [S1, S2, S3, S4]

print("=" * 60)
print("STEP 1: Validating stabilizers (Hermitian, S²=I)")
print("=" * 60)

for i, S in enumerate(stabilizers):
    is_hermitian = np.allclose(S, S.conj().T)
    squares_to_identity = np.allclose(S @ S, np.eye(64))
    print(f"  S{i+1}: Hermitian={is_hermitian}, S²=I: {squares_to_identity}")

print()

# SECTION 3: Verify All Stabilizers Commute
#
# For every pair (Si, Sj), compute the commutator [Si, Sj] = SiSj - SjSi.
# This must be zero for all pairs. If any pair anticommutes, the code
# space is empty and no valid tensor can be extracted.

print("=" * 60)
print("STEP 2: Checking all stabilizers commute")
print("=" * 60)

all_commute = True
for i in range(len(stabilizers)):
    for j in range(i+1, len(stabilizers)):
        commutator = (stabilizers[i] @ stabilizers[j]
                      - stabilizers[j] @ stabilizers[i])
        commutes = np.allclose(commutator, np.zeros((64, 64)))
        status = "✓" if commutes else "✗ FAIL"
        print(f"  [S{i+1}, S{j+1}] = 0 : {commutes} {status}")
        if not commutes:
            all_commute = False

if all_commute:
    print("\n  All stabilizers commute — code space is well defined ✓")
else:
    print("\n  WARNING: Some stabilizers do not commute!")
    print("  Code space is empty — fix the generators before continuing.")

print()

# SECTION 4: Build the Projector onto the Code Space
#
# For each stabilizer S, the operator (I + S)/2 is the projector onto
# its +1 eigenspace:
#   If S|ψ⟩ = +|ψ⟩  →  (I+S)/2 |ψ⟩ = |ψ⟩   (state kept)
#   If S|ψ⟩ = -|ψ⟩  →  (I+S)/2 |ψ⟩ = 0     (state removed)
#
# Multiplying all these projectors together gives the projector onto the
# intersection of all +1 eigenspaces — the code space.
#
# Each commuting stabilizer halves the space:
#   64 → 32 → 16 → 8 → 4 dimensional code space

print("=" * 60)
print("STEP 3: Building projector onto the code space")
print("=" * 60)

dim = 64
P = np.eye(dim, dtype=complex)

for i, S in enumerate(stabilizers):
    P = P @ ((np.eye(dim, dtype=complex) + S) / 2)
    current_rank = int(np.round(np.trace(P).real))
    print(f"  After S{i+1}: code space dimension = {current_rank}")

final_rank = int(np.round(np.trace(P).real))
expected_rank = 4  # 2^k = 2^2 = 4 for [[6,2,3]]

print(f"\n  Final code space dimension = {final_rank}")
print(f"  Expected                  = {expected_rank}")

if final_rank == expected_rank:
    print("  Code space dimension correct ✓")
else:
    print("  ERROR: Wrong code space dimension — check generators!")

print()

# SECTION 5: Extract the Isometry via SVD
#
# The projector P maps onto a 4-dimensional subspace of the
# 64-dimensional space. SVD extracts an explicit basis for that subspace.
#
# SVD: P = U · Σ · V†
# Σ has exactly 4 singular values equal to 1 (corresponding to the
# code space) and 60 singular values equal to 0.
#
# Taking the first 4 columns of U gives the isometry E: C⁴ → C⁶⁴
# which maps 2 logical qubits into 6 physical qubits.

print("=" * 60)
print("STEP 4: Extracting isometry via SVD")
print("=" * 60)

U, singular_values, Vh = np.linalg.svd(P)

print(f"  Top 8 singular values: {np.round(singular_values[:8], 4)}")
print(f"  (Expect: four 1s then zeros)")

code_dim = 4
isometry = U[:, :code_dim]   # shape: (64, 4)

# verify the isometry condition E†E = I_4
EtE = isometry.conj().T @ isometry
isometry_check = np.allclose(EtE, np.eye(code_dim), atol=1e-10)
print(f"\n  Isometry condition E†E = I₄ satisfied: {isometry_check}")

if isometry_check:
    print("  Isometry extracted successfully ✓")

print()

# SECTION 6: Reshape into Tensor Form
#
# The isometry has shape (64, 4):
#   64 = 2^6 → six physical qubit indices, each of dimension 2
#    4 = 2^2 → two logical qubit indices, each of dimension 2
#
# Reshape to (2,2,2,2,2,2,2,2) — 8 indices total:
#   Indices 0-5: physical qubits (boundary legs in the HQNN network)
#   Indices 6-7: logical qubits  (bulk legs in the HQNN network)

print("=" * 60)
print("STEP 5: Reshaping isometry into tensor form")
print("=" * 60)

perfect_tensor = isometry.reshape(2, 2, 2, 2, 2, 2, 2, 2)

print(f"  Tensor shape: {perfect_tensor.shape}")
print(f"  Indices 0-5 → physical qubits (boundary legs)")
print(f"  Indices 6-7 → logical qubits  (bulk legs)")
print()

# SECTION 7: Verify the Perfect Tensor Property
#
# A perfect tensor must satisfy: for ANY bipartition of its 8 indices
# into sets A and B with |A| ≤ |B|, the reshaped matrix M_AB is an
# isometry: M†M = I (or MM† = I if |A| > |B|).
#
# We check this exhaustively for all possible bipartitions.
# For 8 indices, this gives 162 bipartitions to check.
# If all pass, we have a genuine perfect tensor.

print("=" * 60)
print("STEP 6: Verifying perfect tensor — all bipartitions")
print("=" * 60)

n_indices = 8
T = perfect_tensor
all_passed = True
total_checks = 0
failed_checks = []

for size_A in range(1, n_indices // 2 + 1):
    passes_at_this_size = 0
    fails_at_this_size = 0

    for A in combinations(range(n_indices), size_A):
        B = tuple(i for i in range(n_indices) if i not in A)

        # rearrange tensor: A indices first, B indices last
        perm = list(A) + list(B)
        T_perm = np.transpose(T, perm)

        # flatten into matrix
        dim_A = 2 ** len(A)
        dim_B = 2 ** len(B)
        M = T_perm.reshape(dim_A, dim_B)

        # check isometry from smaller to larger space
        if dim_A <= dim_B:
            product = M.conj().T @ M
            is_isometry = np.allclose(product, np.eye(dim_A), atol=1e-8)
        else:
            product = M @ M.conj().T
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

print(f"\n  Total bipartitions checked: {total_checks}")

if failed_checks:
    print(f"  Failed bipartitions:")
    for A, B in failed_checks[:5]:
        print(f"    A={A}, B={B}")

print()

if all_passed:
    print("=" * 60)
    print("  ✓ PERFECT TENSOR VERIFIED")
    print("  All bipartitions satisfy the isometry condition.")
    print("  Stage 1 complete — ready for Stage 2.")
    print("=" * 60)
else:
    print("=" * 60)
    print("  ✗ VERIFICATION FAILED")
    print(f"  {len(failed_checks)} bipartitions failed.")
    print("  The stabilizer generators may not produce a perfect tensor.")
    print("  Check commutativity and code space dimension above.")
    print("=" * 60)

# SECTION 8: Save tensor for Stage 2

np.save("perfect_tensor.npy", perfect_tensor)
print()
print("Perfect tensor saved to 'perfect_tensor.npy'")
print("Load in Stage 2 with: T = np.load('perfect_tensor.npy')")
