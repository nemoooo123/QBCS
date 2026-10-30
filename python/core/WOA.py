import random
import copy
import numpy as np
from utils.init_state import gen_nbrs, repair_sequence_logic
from utils.topology import decode_and_synthesize, verify_circuit_logic

def WOA_run_single_experiment(max_iterations, rotation_cycles, num_neighbors, num_bits,
                                base_trajectory, experiment_id, encoding_table,
                                pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4,
                                fitness_history_matrix, target_output, b,
                                a_start=2.0, a_end=0.0, a_decay_power=1.0):
    
    """
    Executes a single experiment using the Whale Optimization Algorithm (WOA).
    Simulates whale hunting behavior to optimize quantum circuit synthesis.
    """
     # --- Step 1: Initialize Population (Whale Swarm) ---
    # nbr1~4 represent the current positions of all whales in the search space
    nbr1, nbr2, nbr3, nbr4 = gen_nbrs(pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, num_neighbors)
    
    # --- Step 2: Initial Evaluation ---
    circuit_solutions = decode_and_synthesize(nbr1, nbr2, nbr3, nbr4, encoding_table, num_bits, num_neighbors, base_trajectory)
    fitness = [sol[1] for sol in circuit_solutions]
    current_circuits = list(circuit_solutions)
    
    # Track the global leader whale (Best individual found so far)
    best_idx = np.argmin(fitness)
    global_best_gate = fitness[best_idx]
    global_best_circuit = current_circuits[best_idx]
    leader_pos = (copy.deepcopy(nbr1[best_idx]), copy.deepcopy(nbr2[best_idx]), 
                  copy.deepcopy(nbr3[best_idx]), copy.deepcopy(nbr4[best_idx]))

    # --- Step 3: Evolutionary Loop ---
    current_iter = 0
    while current_iter < max_iterations:
        current_iter += 1
        
        # Convergence factor 'a' decays from a_start to a_end over the run,
        # controlling the exploration/exploitation range of A. a_decay_power=1
        # is the classic linear schedule (default, matches the original WOA
        # paper); >1 lingers near a_start longer (more exploration up front),
        # <1 drops toward a_end sooner (more exploitation up front).
        progress = current_iter / max_iterations
        a = a_end + (a_start - a_end) * (1 - progress) ** a_decay_power
        for i in range(num_neighbors):
            p = random.random()  # Decision variable for Bubble-net (Spiral) or Encircling/Search strategy
            r = random.random()
            
            # WOA Core Parameters: A and C
            # A determines: Move toward Leader (|A|<1) or Random Exploration (|A|>=1)
            # C simulates the stochastic influence of the prey
            A = 2 * a * r - a
            C = 2 * r 
            
            if p < 0.5:
                if abs(A) < 1:
                    # [Encircling Phase] Approach the Leader whale
                    new_child = generate_woa_update((nbr1[i], nbr2[i], nbr3[i], nbr4[i]), leader_pos, C)
                else:
                    # [Search Phase] Explore by moving toward a random neighbor
                    rand_idx = random.choice([idx for idx in range(num_neighbors) if idx != i])
                    rand_pos = (nbr1[rand_idx], nbr2[rand_idx], nbr3[rand_idx], nbr4[rand_idx])
                    new_child = generate_woa_update((nbr1[i], nbr2[i], nbr3[i], nbr4[i]), rand_pos, C)
            else:
                # [Bubble-net Phase] Spiral Update (Perform local perturbation around the Leader)
                # Parameter 'b' controls the intensity of the spiral displacement
                new_child = generate_woa_spiral((nbr1[i], nbr2[i], nbr3[i], nbr4[i]), leader_pos, b)

            # --- Evaluate the new candidate solution ---
            new_per_circuit, new_fit = evaluate_woa_fitness(new_child, encoding_table, num_bits, base_trajectory)

            # --- Greedy Selection ---
            if new_fit < fitness[i]:
                nbr1[i], nbr2[i], nbr3[i], nbr4[i] = new_child
                fitness[i] = new_fit
                current_circuits[i] = new_per_circuit
                
                # Synchronously update the Global Leader whale
                if new_fit < global_best_gate:
                    global_best_gate = new_fit
                    global_best_circuit = new_per_circuit
                    leader_pos = (copy.deepcopy(nbr1[i]), copy.deepcopy(nbr2[i]), 
                                  copy.deepcopy(nbr3[i]), copy.deepcopy(nbr4[i]))
                
        # Log the historical best Gate Count for each iteration
        fitness_history_matrix[experiment_id][current_iter - 1] = global_best_gate

    return fitness_history_matrix, global_best_gate, global_best_circuit

# --- Helper Functions: Position Update Logic ---

def generate_woa_update(p1, target, C):
    """ 
    Simulates whale movement toward a target (Leader or Random):
    Determines the depth of layer updates based on parameter C and executes crossover.
    """
    child = list(copy.deepcopy(p1))
    
    # Higher C value represents stronger prey attraction, updating more layers (1~3 layers)
    num_layers = 1 if C < 0.6 else (2 if C < 1.4 else 3)
    layers = random.sample(range(4), num_layers)
    
    for l_idx in layers:
        curr_l = p1[l_idx]
        targ_l = target[l_idx]
        
        # Dimensional Protection: Ensure data is in List format for concatenation
        c_list = curr_l.tolist() if isinstance(curr_l, np.ndarray) else list(curr_l)
        t_list = targ_l.tolist() if isinstance(targ_l, np.ndarray) else list(targ_l)
        
        if len(c_list) > 0:
            # Select random crossover point
            cp = random.randint(0, len(c_list))
            new_l = copy.deepcopy(c_list[:cp]) + copy.deepcopy(t_list[cp:])
            
            # Exclusive logic repair for Layer 4 (Sequence Order)
            if l_idx == 3:
                for k in range(len(new_l)):
                    for l in range(len(new_l[k])):
                        new_l[k][l] = repair_sequence_logic(new_l[k][l])
            child[l_idx] = new_l
            
    return tuple(child)

def generate_woa_spiral(p1, leader, b):
    """ 
    Spiral Position Update:
    Executes local mutation centered around the Leader to simulate the spiral bubble-net attack path.
    Parameter 'b' defines the mutation rate scaling factor.
    """
    child = list(copy.deepcopy(leader))
    # Base mutation rate of 5%, scaled by parameter b
    mutation_rate = 0.05 * b 
    
    for l_idx in range(4): # Process all encoding layers (0, 1, 2, 3)
        layer = child[l_idx]
        
        # --- Layer 1 & 2 (2D Structure) ---
        if l_idx in [0, 1]:
            for r in range(len(layer)):
                for c in range(len(layer[r])):
                    if random.random() < mutation_rate:
                        layer[r][c] = 1 - layer[r][c]
                        
        # --- Layer 3 (Deep 4D/5D Weight Structure) ---
        elif l_idx == 2:
            # Handle reserved 999 placeholders carefully
            for j in range(len(layer)):
                for k in range(len(layer[j])):
                    for l in range(len(layer[j][k])):
                        # Perturb only if it is not a 999 marker
                        if not isinstance(layer[j][k][l], (int, float)) or layer[j][k][l] != 999:
                            if isinstance(layer[j][k][l], list):
                                for m in range(len(layer[j][k][l])):
                                    if random.random() < mutation_rate:
                                        layer[j][k][l][m] = 1 - layer[j][k][l][m]
        
        # --- Layer 4 (3D/4D Sequence Structure + Repair) ---
        elif l_idx == 3:
            for j in range(len(layer)):
                for k in range(len(layer[j])):
                    # Traverse to the sequence bit-level
                    for m in range(len(layer[j][k])):
                        if random.random() < mutation_rate:
                            layer[j][k][m] = 1 - layer[j][k][m]
                    # Immediately repair the sequence after perturbation
                    layer[j][k] = repair_sequence_logic(layer[j][k])
                    
    return tuple(child)

def evaluate_woa_fitness(child, encoding_table, num_bits, trajectories):
    """ 
    Encapsulates decoding and evaluation logic.
    Ensures synthesized circuit content and Gate Count are returned. 
    """
    c1, c2, c3, c4 = child
    # Decode single whale position (wrapped as a population of size 1)
    res = decode_and_synthesize([c1], [c2], [c3], [c4], encoding_table, num_bits, 1, trajectories)
    return res[0][0], len(res[0][0])