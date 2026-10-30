import numpy as np
import random
import copy
from utils.init_state import gen_nbrs
from utils.topology import decode_and_synthesize,verify_circuit_logic




def QEA_run_single_experiment(max_iterations, 
                                 rotation_cycles, 
                                 num_neighbors, 
                                 num_bits, 
                                 base_trajectory, 
                                 experiment_id, 
                                 encoding_table, 
                                 qindividuals1, 
                                 qindividuals2, 
                                 qindividuals3, 
                                 qindividuals4, 
                                 fitness_history_matrix, 
                                 target_output,
                                 delta_theta):
    """
    Executes a single trial of the QEA algorithm for quantum circuit synthesis.
    Iteratively evolves the quantum probability distributions (qindividuals1-4) 
    to converge toward the minimum gate count solution using global best reinforcement.
    """
    
    num_cycles = len(rotation_cycles)
    current_iter = 0
    global_best_gate_count = float('inf')
    global_best_circuit = []

    # Track the Global Best Position (gb_pos) across layers ---
    gb_pos1, gb_pos2, gb_pos3, gb_pos4 = None, None, None, None

    _init_nbr1, _init_nbr2, _init_nbr3, _init_nbr4 = gen_nbrs(
        qindividuals1, qindividuals2, qindividuals3, qindividuals4, num_neighbors
    )
    decode_and_synthesize(
        _init_nbr1, _init_nbr2, _init_nbr3, _init_nbr4, encoding_table, num_bits, num_neighbors, base_trajectory
    )

    # --- QTS Main Evolution Loop ---
    while current_iter < max_iterations:
        current_iter += 1
        
        # Step 1: Neighborhood Generation
        # Sample candidate solutions (neighbors) based on the current quantum 
        # probability amplitudes stored in qindividuals.
        nbr1, nbr2, nbr3, nbr4 = gen_nbrs(
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, num_neighbors
        )

        # Step 2: Decoding and Circuit Synthesis
        # Transform the sampled quantum neighbors into concrete reversible circuit structures.
        circuit_solutions = decode_and_synthesize(
            nbr1, nbr2, nbr3, nbr4, encoding_table, num_bits, num_neighbors, base_trajectory
        )
        # #Integrity Verification
        # # Check if the synthesized circuits fulfill the logic requirements for the target output
        # valid_count = sum(verify_circuit_logic(sol, num_bits, target_output) for sol in circuit_solutions)
        
        # if valid_count != num_neighbors:
        #     print(f"Warning: Logic verification failed for {num_neighbors - valid_count} neighbors.")

        # Step 3: Fitness Evaluation
        # Calculate the gate count for each synthesized circuit solution.
        fitness = [sol[1] for sol in circuit_solutions]
        local_best_idx = np.argmin(fitness)
        local_best_gate = fitness[local_best_idx]

        # Step 4: Update Global Best Solution and Global Best Position
        # If the current local best is superior to the historical global best, update records.
        if local_best_gate < global_best_gate_count:
            global_best_gate_count = local_best_gate
            global_best_circuit = circuit_solutions[local_best_idx]
            # Store the elite "Position Genes" for quantum rotation reinforcement
            gb_pos1 = copy.deepcopy(nbr1[local_best_idx])
            gb_pos2 = copy.deepcopy(nbr2[local_best_idx])
            gb_pos3 = copy.deepcopy(nbr3[local_best_idx])
            gb_pos4 = copy.deepcopy(nbr4[local_best_idx])

        # Step 5: QEA Quantum State Update
        # Rotate quantum states towards the Global Best Position (gb_pos) 
        # to intensify search around the most promising regions of the solution space.
        updateQ(
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, 
            gb_pos1, gb_pos2, gb_pos3, gb_pos4, 
            num_cycles, delta_theta
        )

        # Step 6: Data Recording for Convergence Analysis
        # Store the current global minimum gate count into the fitness history matrix.
        fitness_history_matrix[experiment_id][current_iter - 1] = global_best_gate_count

    # Returns the convergence history and the optimal circuit found via QEA optimization.
    return fitness_history_matrix, global_best_gate_count, global_best_circuit

def updateQ(qindividuals1, qindividuals2, qindividuals3, qindividuals4, 
                            gb_pos1, gb_pos2, gb_pos3, gb_pos4, 
                              num_cycles, delta_theta):
    """
    Updates the probability distributions of quantum populations (q1-q4).
    Shifts quantum amplitudes toward the discrete decisions made in the 
    Global Best Position (gb_pos) to guide future sampling.
    """

    # --- Update qindividuals1 (Strategy/Trajectory level) ---
    best_sol1 = [list(map(int, row)) for row in gb_pos1.tolist()]

    for i in range(len(qindividuals1)):
        for j in range(len(qindividuals1[i])):
            b_val = best_sol1[i][j]
            # Identify the alternative (worst) choice for rotation
            if b_val == 0:
                w_val = 1
            else: w_val = 0
            # Rotate toward the global best bit
            qindividuals1[i][j][b_val] += delta_theta
            qindividuals1[i][j][w_val] -= delta_theta
            
            # Boundary Correction for Quantum Probability [0, 1]
            if qindividuals1[i][j][w_val] <= 0:
                qindividuals1[i][j][b_val] = 1.0
                qindividuals1[i][j][w_val] = 0.0

    # --- Update qindividuals2 (Segment level) ---
    best_sol2 = gb_pos2

    for i in range(len(qindividuals2)):
        for j in range(len(qindividuals2[i])):
            b_val = best_sol2[i][j]
            if b_val == 0:
                w_val = 1
            else: w_val = 0

            qindividuals2[i][j][b_val] += delta_theta
            qindividuals2[i][j][w_val] -= delta_theta
            if qindividuals2[i][j][w_val] <= 0:
                qindividuals2[i][j][b_val] = 1.0
                qindividuals2[i][j][w_val] = 0.0

    # --- Update qindividuals3 (Route/Path level) ---
    best_sol3 = gb_pos3

    for i in range(len(qindividuals3)):
        for j in range(len(best_sol3[i])):
            for k in range(len(best_sol3[i][j])):
                # Respect structural constraints (999 markers indicate fixed topology)
                if best_sol3[i][j][k] != 999 :
                    num_choices = len(qindividuals3[i][j][k])
                    for l in range(num_choices):
                        b_val = best_sol3[i][j][k][l]
                        if b_val == 0:
                            w_val = 1
                        else: w_val = 0
                        qindividuals3[i][j][k][l][b_val] += delta_theta
                        qindividuals3[i][j][k][l][w_val] -= delta_theta
                        if qindividuals3[i][j][k][l][w_val] <= 0:
                            qindividuals3[i][j][k][l][b_val] = 1.0
                            qindividuals3[i][j][k][l][w_val] = 0.0

    # --- Update qindividuals4 (Gate Order level) ---
    best_sol4 = gb_pos4

    for i in range(num_cycles):
        for j in range(len(best_sol4[i])):
            # Skip trivial single-gate sequences
            if len(best_sol4[i][j]) > 1:
                for k in range(len(best_sol4[i][j])):
                    
                    b_val = best_sol4[i][j][k]
                    if b_val == 0:
                        w_val = 1
                    else: w_val = 0
                    
                    qindividuals4[i][j][k][b_val] += delta_theta
                    qindividuals4[i][j][k][w_val] -= delta_theta
                    if qindividuals4[i][j][k][w_val] <= 0:
                        qindividuals4[i][j][k][b_val] = 1.0
                        qindividuals4[i][j][k][w_val] = 0.0


          
   