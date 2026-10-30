import random
import copy
import numpy as np
from utils.init_state import gen_nbrs, repair_sequence_logic
from utils.topology import decode_and_synthesize, verify_circuit_logic

def ABC_run_single_experiment(max_iterations,
                rotation_cycles,
                num_neighbors,
                num_bits,
                base_trajectory,
                experiment_id,
                encoding_table,
                pop_matrix1,
                pop_matrix2,
                pop_matrix3,
                pop_matrix4,
                fitness_history_matrix,
                target_output,
                limit):

    # Initialize population (Food Sources)
    nbr1, nbr2, nbr3, nbr4 = gen_nbrs(pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, num_neighbors)

    # Initial evaluation
    circuit_solutions = decode_and_synthesize(nbr1, nbr2, nbr3, nbr4, encoding_table, num_bits, num_neighbors, base_trajectory)
    fitness = [sol[1] for sol in circuit_solutions]
    circuits = [sol[0] for sol in circuit_solutions]

    trials = [0] * num_neighbors # Record the number of trials without fitness improvement for each food source
    global_best_gate = min(fitness)
    global_best_circuit = circuits[np.argmin(fitness)]

    current_iter = 0
    while current_iter < max_iterations:
        current_iter += 1

        # Scout Bee Phase runs first and its evaluations are deducted from this iteration's
        # employed/onlooker budget, so every iteration spends exactly num_neighbors evaluations.
        scout_targets = [i for i in range(num_neighbors) if trials[i] > limit]
        nbr1, nbr2, nbr3, nbr4, fitness, circuits, trials = scout_bee_phase(
            nbr1, nbr2, nbr3, nbr4, fitness, circuits, trials, scout_targets, encoding_table, num_bits, base_trajectory)

        remaining = num_neighbors - len(scout_targets)
        emp_n = remaining // 2
        onl_n = remaining - emp_n

        # 1. Employed Bee Phase: Local search and comparison for each food source
        nbr1, nbr2, nbr3, nbr4, fitness, circuits, trials = employed_bee_phase(
            nbr1, nbr2, nbr3, nbr4, fitness, circuits, trials, encoding_table, num_bits, base_trajectory, emp_n)

        # 2. Onlooker Bee Phase: Probabilistic search reinforcement based on solution quality
        nbr1, nbr2, nbr3, nbr4, fitness, circuits, trials = onlooker_bee_phase(
            nbr1, nbr2, nbr3, nbr4, fitness, circuits, trials, encoding_table, num_bits, base_trajectory, onl_n)

        # Integrity Verification
        # Check if the synthesized circuits fulfill the logic requirements for the target output
        # valid_count = sum(verify_circuit_logic(sol, num_bits, target_output) for sol in circuits)

        # if valid_count != num_neighbors:
        #     print(f"Warning: Logic verification failed for {num_neighbors - valid_count} neighbors.")
        # Update Global Best Solution
        if min(fitness) < global_best_gate:
            global_best_gate = min(fitness)
            global_best_circuit = circuits[np.argmin(fitness)]

        fitness_history_matrix[experiment_id][current_iter - 1] = global_best_gate

    return fitness_history_matrix, global_best_gate, global_best_circuit

# --- 1. Employed Bee Phase ---
def employed_bee_phase(nbr1, nbr2, nbr3, nbr4, fitness, circuits, trials, encoding_table, num_bits, trajectories, num_bees):
    for i in range(num_bees):
        # Select a random neighbor j different from i
        j = random.choice([idx for idx in range(len(fitness)) if idx != i])

        # Generate a candidate solution (Local search on a specific layer)
        new_child = generate_abc_neighbor( (nbr1[i], nbr2[i], nbr3[i], nbr4[i]),
                                          (nbr1[j], nbr2[j], nbr3[j], nbr4[j]), num_bits )

        # Evaluation and Greedy Selection
        new_per_circuit, new_fit = evaluate_abc_fitness(new_child, encoding_table, num_bits, trajectories)

        if new_fit < fitness[i]:
            nbr1[i], nbr2[i], nbr3[i], nbr4[i] = new_child
            fitness[i] = new_fit
            circuits[i] = new_per_circuit
            trials[i] = 0
        else:
            trials[i] += 1
    return nbr1, nbr2, nbr3, nbr4, fitness, circuits, trials

# --- 2. Onlooker Bee Phase ---
def onlooker_bee_phase(nbr1, nbr2, nbr3, nbr4, fitness, circuits, trials, encoding_table, num_bits, trajectories, num_bees):

    # Calculate probabilities (Proportional to quality; lower gate count = higher probability)
    # Using reciprocal transformation to handle minimization
    weights = [1.0 / (f + 1e-6) for f in fitness]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    onlookers_dispatched = 0
    while onlookers_dispatched < num_bees:
        # Roulette Wheel Selection for food source development
        i = np.random.choice(range(len(fitness)), p=probs)
        j = random.choice([idx for idx in range(len(fitness)) if idx != i])

        new_child = generate_abc_neighbor( (nbr1[i], nbr2[i], nbr3[i], nbr4[i]),
                                          (nbr1[j], nbr2[j], nbr3[j], nbr4[j]), num_bits )

        new_per_circuit, new_fit = evaluate_abc_fitness(new_child, encoding_table, num_bits, trajectories)

        if new_fit < fitness[i]:
            nbr1[i], nbr2[i], nbr3[i], nbr4[i] = new_child
            fitness[i] = new_fit
            circuits[i] = new_per_circuit
            trials[i] = 0
        else:
            trials[i] += 1
        onlookers_dispatched += 1

    return nbr1, nbr2, nbr3, nbr4, fitness, circuits, trials

# --- 3. Scout Bee Phase ---
def scout_bee_phase(nbr1, nbr2, nbr3, nbr4, fitness, circuits, trials, scout_targets, encoding_table, num_bits, trajectories):
    for i in scout_targets:
       # Abandon food source and re-initialize randomly (Scout behavior)
        # Mutation logic used here to achieve a "random jump" in the search space

        # 1. Mutate Layer 1 and Layer 2
        for layer in [nbr1[i], nbr2[i]]:
            for r in range(len(layer)):
                for c in range(len(layer[r])):
                    if random.random() < 0.3: # High probability mutation
                        layer[r][c] = 1 - layer[r][c]

        # 2. Handle Layer 3 (4D structure: [j][k][l][m])
        for j in range(len(nbr3[i])):
            for k in range(len(nbr3[i][j])):
                for l in range(len(nbr3[i][j][k])):
                    # Skip if it is a reserved placeholder (999)

                    if nbr3[i][j][k][l] != 999:
                        for m in range(len(nbr3[i][j][k][l])):
                            if random.random() < 0.3:
                                nbr3[i][j][k][l][m] = 1 - nbr3[i][j][k][l][m]

        # 3. Handle Layer 4
        for j in range(len(nbr4[i])):
            for k in range(len(nbr4[i][j])):
                for l in range(len(nbr4[i][j][k])):
                    if random.random() < 0.3:
                        nbr4[i][j][k][l] = 1 - nbr4[i][j][k][l]
                # Apply repair logic to ensure sequence validity
                nbr4[i][j][k] = repair_sequence_logic(nbr4[i][j][k])



        # Re-evaluate fitness for the new scouted solution
        new_per_circuit, fitness[i] = evaluate_abc_fitness((nbr1[i], nbr2[i], nbr3[i], nbr4[i]), encoding_table, num_bits, trajectories)
        circuits[i] = new_per_circuit
        trials[i] = 0
    return nbr1, nbr2, nbr3, nbr4, fitness, circuits, trials

# --- Helper Function: Neighbor Generation (ABC Search Operator) ---
def generate_abc_neighbor(p1, p2, num_bits):
    """
    ABC Local Search: Performs a 'partial swap' with a neighbor on a randomly selected layer.
    """
    child = list(copy.deepcopy(p1))
    target_layer = random.randint(0, 3) # Randomly select layer 0-3 (L1-L4)

    layer_p1 = p1[target_layer]
    layer_p2 = p2[target_layer]

    if len(layer_p1) > 0:
        l1_list = layer_p1.tolist() if isinstance(layer_p1, np.ndarray) else list(layer_p1)
        l2_list = layer_p2.tolist() if isinstance(layer_p2, np.ndarray) else list(layer_p2)

        # Crossover point selection
        cp = random.randint(0, len(l1_list))

        # Structured merging (crossover)
        new_l = copy.deepcopy(l1_list[:cp]) + copy.deepcopy(l2_list[cp:])

        # Apply repair mechanism if Layer 4 (index 3) is modified
        if target_layer == 3:
            for k in range(len(new_l)):
                new_l[k] = repair_sequence_logic(new_l[k])

        child[target_layer] = new_l

    return tuple(child)

def evaluate_abc_fitness(child, encoding_table, num_bits, trajectories):
    """
    Decode and evaluate the gate count of a single candidate solution.
    """
    c1, c2, c3, c4 = child
    circuit_sols = decode_and_synthesize([c1], [c2], [c3], [c4], encoding_table, num_bits, 1, trajectories)
    return circuit_sols[0][0], len(circuit_sols[0][0])