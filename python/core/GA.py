from utils.init_state import gen_nbrs, repair_sequence_logic
from utils.topology import decode_and_synthesize, verify_circuit_logic

import random
import copy
import numpy as np

def GA_run_single_experiment(
                max_iterations,
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
                k,
                pc,
                pm
            ):
    """
    Execute a single experiment using Genetic Algorithm (GA).
    Manages the evolutionary process: initialization, evaluation, and global best tracking.
    """

    # --- Population Initialization ---
    # Generate initial neighbors (individuals) based on the input population matrices
    nbr1, nbr2, nbr3, nbr4 = gen_nbrs(pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, num_neighbors)
    
    # --- Initial Evaluation ---
    # Decode encoding layers and synthesize them into quantum circuits
    circuit_solutions = decode_and_synthesize(
        nbr1, nbr2, nbr3, nbr4, encoding_table, num_bits, num_neighbors, base_trajectory
    )
    # Calculate initial fitness (Gate Count in this context)
    # Format: [gate_count, gate_count, ...]
    local_fitness = [sol[1] for sol in circuit_solutions]
    local_indices = list(range(num_neighbors)) # Index tracking [0, 1, 2, ...]
    # Initialize global best records
    global_best_gate_count = min(local_fitness)
    best_idx = np.argmin(local_fitness)
    global_best_circuit = circuit_solutions[best_idx][0]
    local_best_circuit = circuit_solutions[best_idx][0]
    # --- Evolutionary Loop ---
    current_iter = 0
    while current_iter < max_iterations:
        current_iter += 1
        
        # Execute core GA operations: Selection, Crossover, Mutation, and Evaluation
        # Returns new generation's encoding, fitness, and synthesized circuits
        nbr1, nbr2, nbr3, nbr4, local_fitness, local_circuit = Genetic_Algorithm_Core(
            nbr1, nbr2, nbr3, nbr4, num_neighbors, num_bits, k, pc, pm,
            encoding_table, local_fitness, local_best_circuit, local_indices, base_trajectory
        )
        # --- Update Global Best Solution ---
        # Find the best individual in the current generation
        current_local_best_idx = np.argmin(local_fitness)
        current_local_best_fit = local_fitness[current_local_best_idx]
        local_best_circuit = local_circuit[current_local_best_idx]

        # Update global records if a better solution is found
        if current_local_best_fit < global_best_gate_count:

            global_best_gate_count = current_local_best_fit
            global_best_circuit = local_best_circuit

        # Log historical data for convergence curve plotting
        fitness_history_matrix[experiment_id][current_iter - 1] = global_best_gate_count

    return fitness_history_matrix, global_best_gate_count, global_best_circuit


def Genetic_Algorithm_Core(nbr1, nbr2, nbr3, nbr4, num_neighbors, num_bits, k, pc, pm,
                           encoding_table, local_fitness, local_best_circuit, local_indices, base_trajectory):
    """
    Execution of single generation GA logic: Selection -> Crossover -> Mutation -> Evaluation -> Elitist Truncation.
    Generates a full num_neighbors offspring (one fitness evaluation each) so every generation
    spends exactly num_neighbors evaluations, then merges the previous elite back in and keeps
    the best num_neighbors survivors (11-in / 10-out replacement) without re-evaluating it.
    """
    new_nbr1, new_nbr2, new_nbr3, new_nbr4, new_fit, new_circuit = [], [], [], [], [], []

    best_in_current_idx = np.argmin(local_fitness)
    elite = (
        copy.deepcopy(nbr1[best_in_current_idx]), copy.deepcopy(nbr2[best_in_current_idx]),
        copy.deepcopy(nbr3[best_in_current_idx]), copy.deepcopy(nbr4[best_in_current_idx]),
        local_fitness[best_in_current_idx], local_best_circuit
    )

    # Generate and evaluate a full new generation (num_neighbors offspring)
    while len(new_fit) < num_neighbors:
        # Parent Selection via Tournament Selection
        p1_idx = tournament_selection(local_fitness, local_indices, k)
        p2_idx = tournament_selection(local_fitness, local_indices, k)
        
        p1 = (nbr1[p1_idx], nbr2[p1_idx], nbr3[p1_idx], nbr4[p1_idx])
        p2 = (nbr1[p2_idx], nbr2[p2_idx], nbr3[p2_idx], nbr4[p2_idx])

        # Crossover Operation
        if random.random() < pc:
            child = crossover_op(p1, p2)
        else:
            child = copy.deepcopy(p1)
        # Mutation Operation
        child = mutation_op(child, pm)

        # Unpack child encoding
        c1, c2, c3, c4 = child

        # Evaluate the new offspring
        circuit_solutions = decode_and_synthesize(
        [c1], [c2], [c3], [c4], encoding_table, num_bits, 1, base_trajectory
        )
        fit = circuit_solutions[0][1]
        # Append new individual to the population
        new_nbr1.append(c1)
        new_nbr2.append(c2)
        new_nbr3.append(c3)
        new_nbr4.append(c4)
        new_circuit.append(circuit_solutions[0][0])
        new_fit.append(fit)

    # Elitist truncation: merge the previous elite back in (11 candidates) and keep the best
    # num_neighbors without spending an extra evaluation on it (its fitness is already known)
    new_nbr1.append(elite[0])
    new_nbr2.append(elite[1])
    new_nbr3.append(elite[2])
    new_nbr4.append(elite[3])
    new_fit.append(elite[4])
    new_circuit.append(elite[5])

    survivors = np.argsort(new_fit)[:num_neighbors]
    new_nbr1 = [new_nbr1[i] for i in survivors]
    new_nbr2 = [new_nbr2[i] for i in survivors]
    new_nbr3 = [new_nbr3[i] for i in survivors]
    new_nbr4 = [new_nbr4[i] for i in survivors]
    new_fit = [new_fit[i] for i in survivors]
    new_circuit = [new_circuit[i] for i in survivors]

    return new_nbr1, new_nbr2, new_nbr3, new_nbr4, new_fit, new_circuit

def tournament_selection(fitness_list, idx_list, k):
    """
    Tournament Selection:
    Randomly select k individuals and return the index of the one with the best fitness.
    Higher k increases selection pressure.
    """

    tournament_indices = random.sample(range(len(fitness_list)), k)
    best_in_tournament = min(tournament_indices, key=lambda i: fitness_list[i])
    return idx_list[best_in_tournament]

def crossover_op(p1, p2):
    """
    Single-point Crossover: Perform random cut-point exchange for each of the four encoding layers.
    """
    child = []
    for layer_p1, layer_p2 in zip(p1, p2):
        if len(layer_p1) > 0:
            # Randomly choose a crossover point
            cp = random.randint(0, len(layer_p1))
            # Handle both numpy arrays and list formats for concatenation
            l1_list = layer_p1.tolist() if isinstance(layer_p1, np.ndarray) else list(layer_p1)
            l2_list = layer_p2.tolist() if isinstance(layer_p2, np.ndarray) else list(layer_p2)
            
            new_layer = copy.deepcopy(l1_list[:cp]) + copy.deepcopy(l2_list[cp:])
        else:
            new_layer = copy.deepcopy(layer_p1)
        child.append(new_layer)
        
    return tuple(child)

def mutation_op(child, pm):
    """
    Mutation Operation: Flip bits (0 <-> 1) for each element with probability pm.
    Includes logic repair for the 4th layer sequences.
    """
    c1, c2, c3, c4 = child

    # Layer 1 & 2 Mutation (2D Matrices)
    for layer in [c1, c2]:
        for i in range(len(layer)):
            for j in range(len(layer[i])):
                if random.random() < pm:
                    layer[i][j] = 1 - layer[i][j]

    # Layer 3 Mutation (Nested 4D Structure)
    for i in range(len(c3)):
        for j in range(len(c3[i])):
            for k in range(len(c3[i][j])):
                if c3[i][j][k] != 999: # Skip padding/special markers
                    for l in range(len(c3[i][j][k])):
                        if random.random() < pm:
                            c3[i][j][k][l] = 1 - c3[i][j][k][l]

    # Layer 4 Mutation (3D Sequences) with immediate logic repair
    for i in range(len(c4)):
        for j in range(len(c4[i])):
            for k in range(len(c4[i][j])):
                if random.random() < pm:
                    c4[i][j][k] = 1 - c4[i][j][k]
            # Repair sequence logic immediately after mutation (e.g., handle redundancy)
            c4[i][j] = repair_sequence_logic(c4[i][j])
        
    return (c1, c2, c3, c4)



