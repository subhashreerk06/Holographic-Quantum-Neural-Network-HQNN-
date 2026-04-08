# HQNN Stage 3: Boundary Ansatz Circuit and Training
# File: Ansatz_training_3.py
#
# Depends on: Tensor_network_2.py (Stage 2)
# Run Perfect_tensor_1.py and Tensor_network_2.py first.
#
# What this file does:
#   1. Loads the encoding map E from Stage 2
#   2. Generates random logical training states
#   3. Defines a parameterised boundary circuit using PennyLane
#   4. Defines the loss function (1 - fidelity)
#   5. Trains the boundary circuit using Adam optimiser
#   6. Plots and saves the training curve
#   7. Saves the trained parameters for Stage 4
#
# The full HQNN transformation is:
#
#   |ψ_out⟩ = E† · Uboundary(θ) · E · |ψ_in⟩
#
# For the first training task (quantum state reconstruction),
# we want |ψ_out⟩ ≈ |ψ_in⟩ — the circuit learns to preserve
# logical states through the encode → process → decode pipeline.
#
# This is the simplest end-to-end task that verifies the full
# HQNN pipeline works correctly before moving to harder tasks.

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ENC_PATH    = os.path.join(SCRIPT_DIR, "encoding_map.npy")
PARAMS_PATH = os.path.join(SCRIPT_DIR, "trained_params.npy")
CURVE_PATH  = os.path.join(SCRIPT_DIR, "training_curve.png")

print("HQNN Stage 3: Boundary Ansatz Circuit and Training")
print()

# SECTION 1: Load the Encoding Map from Stage 2
#
# E has shape (dim_boundary, dim_logical) = (8192, 64)
# It maps logical states (C^64) into boundary states (C^8192).
# E† maps back from boundary to logical.
#
# These dimensions come from the 3-node network in Stage 2:
#   13 boundary qubits → 2^13 = 8192
#    6 logical  qubits → 2^6  = 64

print("STEP 1: Loading encoding map from Stage 2")

try:
    E = np.load(ENC_PATH)
    print(f"  Loaded encoding map. Shape: {E.shape}")
    print(f"  Boundary dimension : 2^13 = {E.shape[0]}")
    print(f"  Logical  dimension : 2^6  = {E.shape[1]}")
except FileNotFoundError:
    print("  ERROR: encoding_map.npy not found.")
    print("  Run Tensor_network_2.py first.")
    exit()

# extract dimensions from the encoding map directly
dim_boundary = E.shape[0]    # 8192
dim_logical  = E.shape[1]    # 64
n_boundary   = int(np.round(np.log2(dim_boundary)))   # 13
n_logical    = int(np.round(np.log2(dim_logical)))    # 6

print(f"  Boundary qubits: {n_boundary}")
print(f"  Logical  qubits: {n_logical}")
print()

# SECTION 2: Generate Training States
#
# We train on a set of random logical states — vectors in C^64.
# Each state is a random unit vector (normalised complex vector).
# The circuit should learn to preserve all of them through the
# encode → boundary circuit → decode pipeline.
#
# Using random states is standard practice — if the circuit can
# reconstruct random states with high fidelity, it has learned
# a good identity-like operation in the logical space.

print("STEP 2: Generating training states")

rng     = np.random.default_rng(seed=42)   # fixed seed for reproducibility
n_train = 20                               # number of training states

# generate random complex vectors and normalise each to unit length
raw_states     = rng.standard_normal((n_train, dim_logical)) \
               + 1j * rng.standard_normal((n_train, dim_logical))
norms          = np.linalg.norm(raw_states, axis=1, keepdims=True)
logical_states = raw_states / norms   # shape: (n_train, 64)

print(f"  Generated {n_train} random logical training states")
print(f"  Each state is a unit vector in C^{dim_logical}")
print(f"  Checking norms (all should be 1.0):")
sample_norms = np.linalg.norm(logical_states[:3], axis=1)
print(f"    States 0-2 norms: {np.round(sample_norms, 6)}")
print()

# pre-encode all training states into boundary states
# boundary_state = E @ logical_state, then renormalise
# shape: (n_train, dim_boundary)
boundary_states_raw = (E @ logical_states.T).T   # shape: (n_train, 8192)
bnorms              = np.linalg.norm(boundary_states_raw, axis=1, keepdims=True)
boundary_states     = boundary_states_raw / bnorms

print(f"  Encoded {n_train} states to boundary. Shape: {boundary_states.shape}")
print()

# SECTION 3: Define the PennyLane Device and Boundary Circuit
#
# The boundary circuit acts on all 13 boundary qubits.
# It is a layered ansatz with:
#   - Single qubit rotations (Rx, Ry, Rz) on every qubit
#   - Nearest-neighbour CNOT entangling gates
#   - Multi-scale connections at distances 2, 4, 8 (reflecting
#     the hyperbolic geometry of the tensor network)
#
# Total trainable parameters per layer:
#   13 qubits × 3 angles = 39 single-qubit parameters
#   (entangling gates are fixed — no parameters)
#
# With N_LAYERS layers: total params = N_LAYERS × 39

print("STEP 3: Defining PennyLane device and boundary circuit")

N_LAYERS = 3   # number of circuit layers — increase for more expressivity

dev = qml.device("default.qubit", wires=n_boundary)

print(f"  Device: default.qubit with {n_boundary} wires")
print(f"  Circuit layers: {N_LAYERS}")
print(f"  Parameters per layer: {n_boundary} qubits × 3 angles = {n_boundary * 3}")
print(f"  Total trainable parameters: {N_LAYERS * n_boundary * 3}")
print()


@qml.qnode(dev, interface="autograd")
def boundary_circuit(params, input_state):
    """
    Parameterised boundary circuit Uboundary(θ).

    Takes an encoded boundary state as input, applies a layered
    ansatz of single-qubit rotations and entangling gates, and
    returns the final quantum state.

    Args:
        params:      shape (N_LAYERS, n_boundary, 3)
                     params[layer, qubit, 0] = Rx angle
                     params[layer, qubit, 1] = Ry angle
                     params[layer, qubit, 2] = Rz angle

        input_state: shape (dim_boundary,) = (8192,)
                     the encoded boundary state from E @ logical_state

    Returns:
        The full statevector after the circuit — shape (8192,)
    """

    # prepare the boundary state as the circuit's initial state
    # this embeds E|ψ⟩ into the quantum circuit
    qml.StatePrep(input_state, wires=range(n_boundary))

    for layer in range(N_LAYERS):

        # single qubit rotations on every boundary qubit
        # Rx, Ry, Rz together can implement any single-qubit unitary
        for qubit in range(n_boundary):
            qml.RX(params[layer, qubit, 0], wires=qubit)
            qml.RY(params[layer, qubit, 1], wires=qubit)
            qml.RZ(params[layer, qubit, 2], wires=qubit)

        # nearest-neighbour entangling gates
        # creates correlations between adjacent boundary qubits
        for qubit in range(n_boundary - 1):
            qml.CNOT(wires=[qubit, qubit + 1])

        # multi-scale connections at distances 2, 4, 8
        # reflects the hyperbolic geometry of the underlying tensor network
        # scale s=1: distance 2 (qubits 0-2, 1-3, 2-4, ...)
        # scale s=2: distance 4 (qubits 0-4, 1-5, 2-6, ...)
        # scale s=3: distance 8 (qubits 0-8, 1-9, 2-10, ...)
        for scale in range(1, 4):
            distance = 2 ** scale
            for qubit in range(n_boundary - distance):
                qml.CNOT(wires=[qubit, qubit + distance])

    # return the full statevector of the 13-qubit boundary system
    return qml.state()


# SECTION 4: Define the Loss Function
#
# For quantum state reconstruction, the loss is:
#   L(θ) = 1 - average_fidelity(θ)
#
# For each training state |ψ_i⟩:
#   1. Run boundary circuit:  |φ_i⟩ = Uboundary(θ) E|ψ_i⟩
#   2. Decode:                |ψ_i_out⟩ = E† |φ_i⟩
#   3. Fidelity:              F_i = |⟨ψ_i|ψ_i_out⟩|²
#
# Average fidelity = (1/n_train) Σ F_i
# Loss = 1 - average_fidelity  (minimising this → maximising fidelity)
#
# Perfect reconstruction: loss = 0, fidelity = 1
# Random circuit:         loss ≈ 1 - 1/dim_logical ≈ 0.98

def compute_fidelity(params, logical_state, boundary_state):
    """
    Compute the reconstruction fidelity for a single training state.

    Full pipeline:
        logical_state
            → E @ logical_state         (encode)
            → Uboundary(θ) @ boundary   (boundary circuit)
            → E† @ output               (decode)
            → fidelity with logical_state

    Returns:
        fidelity: float in [0, 1]
                  1.0 = perfect reconstruction
                  0.0 = orthogonal to target
    """

    # run the boundary circuit — returns 8192-dimensional state vector
    boundary_output = boundary_circuit(params, boundary_state)

    # decode back to logical space using E†
    # E† has shape (64, 8192), boundary_output has shape (8192,)
    logical_output = E.conj().T @ boundary_output   # shape: (64,)

    # compute fidelity |⟨ψ_target|ψ_output⟩|²
    overlap  = np.dot(logical_state.conj(), logical_output)
    fidelity = np.abs(overlap) ** 2

    return fidelity.real


def loss_function(params):
    """
    Average loss over all training states.
    Loss = 1 - mean(fidelity over all training states)
    """
    total_fidelity = 0.0

    for i in range(n_train):
        f = compute_fidelity(
            params,
            logical_states[i],
            boundary_states[i]
        )
        total_fidelity += f

    avg_fidelity = total_fidelity / n_train
    loss         = 1.0 - avg_fidelity
    return loss


# SECTION 5: Initialise Parameters and Test the Circuit
#
# Before training, verify the circuit runs correctly with random
# initial parameters. This catches any shape or interface errors
# before the training loop begins.

print("STEP 4: Initialising parameters and testing circuit")

# initialise parameters as small random angles near zero
# small initialisation avoids barren plateaus at the start of training
params = rng.uniform(-0.1, 0.1, size=(N_LAYERS, n_boundary, 3))
params = qml.numpy.array(params, requires_grad=True)

print(f"  Parameter shape: {params.shape}")
print(f"  Initial parameter range: [{params.min():.4f}, {params.max():.4f}]")

# test forward pass with one training state
print(f"  Testing circuit with training state 0...")
test_output   = boundary_circuit(params, boundary_states[0])
test_fidelity = compute_fidelity(params, logical_states[0], boundary_states[0])
initial_loss  = loss_function(params)

print(f"  Circuit output shape: {test_output.shape}")
print(f"  Test fidelity (state 0): {test_fidelity:.6f}")
print(f"  Initial average loss:    {initial_loss:.6f}")
print(f"  Initial average fidelity: {1 - initial_loss:.6f}")
print()

# SECTION 6: Training Loop
#
# We use PennyLane's built-in Adam optimiser — a gradient-based
# optimiser that adapts the learning rate for each parameter
# individually. Gradients are computed using PennyLane's
# parameter-shift rule, which is exact (not finite differences)
# and hardware-compatible.
#
# Training loop for each iteration:
#   1. Compute loss and gradients via parameter shift
#   2. Update parameters using Adam
#   3. Record loss for the training curve
#   4. Print progress every 10 iterations

print("STEP 5: Training the boundary circuit")

N_ITERATIONS  = 100     # number of training steps
LEARNING_RATE = 0.05    # Adam learning rate

optimiser    = qml.AdamOptimizer(stepsize=LEARNING_RATE)
loss_history = []

print(f"  Optimiser:       Adam")
print(f"  Learning rate:   {LEARNING_RATE}")
print(f"  Iterations:      {N_ITERATIONS}")
print(f"  Training states: {n_train}")
print()
print(f"  {'Iteration':<12} {'Loss':<12} {'Fidelity':<12}")

for iteration in range(N_ITERATIONS):

    # one Adam step — computes gradients and updates params in place
    params, current_loss = optimiser.step_and_cost(loss_function, params)

    # record for the training curve
    loss_history.append(float(current_loss))

    # print progress every 10 iterations
    if (iteration + 1) % 10 == 0 or iteration == 0:
        fidelity = 1.0 - float(current_loss)
        print(f"  {iteration+1:<12} {float(current_loss):<12.6f} {fidelity:<12.6f}")

print()

# SECTION 7: Evaluate Final Performance
#
# After training, evaluate the circuit on all training states and
# report the final reconstruction fidelity per state.
# Also test on a fresh set of unseen states to check generalisation.

print("STEP 6: Final performance evaluation")
print()
print("  Per-state fidelities after training:")
print(f"  {'State':<8} {'Fidelity':<12} {'Result':<10}")

fidelities = []
for i in range(n_train):
    f = compute_fidelity(params, logical_states[i], boundary_states[i])
    fidelities.append(float(f))
    result = "GOOD" if f > 0.9 else "POOR"
    print(f"  {i:<8} {float(f):<12.6f} {result}")

avg_fidelity = np.mean(fidelities)
min_fidelity = np.min(fidelities)
max_fidelity = np.max(fidelities)

print()
print(f"  Average fidelity: {avg_fidelity:.6f}")
print(f"  Min fidelity:     {min_fidelity:.6f}")
print(f"  Max fidelity:     {max_fidelity:.6f}")
print()

# test on unseen states to check generalisation
n_test      = 10
raw_test    = rng.standard_normal((n_test, dim_logical)) \
            + 1j * rng.standard_normal((n_test, dim_logical))
test_norms  = np.linalg.norm(raw_test, axis=1, keepdims=True)
test_states = raw_test / test_norms

test_boundary_raw = (E @ test_states.T).T
test_bnorms       = np.linalg.norm(test_boundary_raw, axis=1, keepdims=True)
test_boundary     = test_boundary_raw / test_bnorms

test_fidelities = []
for i in range(n_test):
    f = compute_fidelity(params, test_states[i], test_boundary[i])
    test_fidelities.append(float(f))

avg_test_fidelity = np.mean(test_fidelities)
print(f"  Generalisation test ({n_test} unseen states):")
print(f"  Average fidelity on unseen states: {avg_test_fidelity:.6f}")
print()

# SECTION 8: Plot and Save the Training Curve

print("STEP 7: Saving training curve")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# loss curve
ax1.plot(range(1, N_ITERATIONS + 1), loss_history,
         color='royalblue', linewidth=2)
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Loss (1 - Fidelity)', fontsize=12)
ax1.set_title('Training Loss', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# fidelity curve
fidelity_history = [1 - l for l in loss_history]
ax2.plot(range(1, N_ITERATIONS + 1), fidelity_history,
         color='seagreen', linewidth=2)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Average Fidelity', fontsize=12)
ax2.set_title('Reconstruction Fidelity', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])
ax2.axhline(y=0.9, color='red', linestyle='--',
            alpha=0.5, label='0.9 threshold')
ax2.legend()

plt.suptitle('HQNN Stage 3: Boundary Circuit Training', fontsize=14)
plt.tight_layout()
plt.savefig(CURVE_PATH, dpi=150, bbox_inches='tight')
plt.close()

print(f"  Training curve saved to: {CURVE_PATH}")
print()

# SECTION 9: Save Trained Parameters

print("STEP 8: Saving trained parameters")

np.save(PARAMS_PATH, np.array(params))

if os.path.exists(PARAMS_PATH):
    size_kb = os.path.getsize(PARAMS_PATH) / 1024
    print(f"  Saved: {PARAMS_PATH}")
    print(f"  File size: {size_kb:.2f} KB")
    print(f"  Parameter shape: {params.shape}")
else:
    print(f"  ERROR: Could not save parameters to {PARAMS_PATH}")

print()

# SUMMARY

print("STAGE 3 COMPLETE — SUMMARY")
print()
print(f"  Circuit architecture:")
print(f"    Boundary qubits : {n_boundary}")
print(f"    Layers          : {N_LAYERS}")
print(f"    Total params    : {N_LAYERS * n_boundary * 3}")
print()
print(f"  Training:")
print(f"    Iterations      : {N_ITERATIONS}")
print(f"    Training states : {n_train}")
print(f"    Initial loss    : {loss_history[0]:.6f}")
print(f"    Final loss      : {loss_history[-1]:.6f}")
print()
print(f"  Results:")
print(f"    Train fidelity  : {avg_fidelity:.6f}")
print(f"    Test fidelity   : {avg_test_fidelity:.6f}")
print()
print(f"  Files saved:")
print(f"    trained_params.npy   — optimised circuit parameters")
print(f"    training_curve.png   — loss and fidelity plots")
print()
print(f"  Load in Stage 4 with:")
print(f"    params = np.load('trained_params.npy')")
