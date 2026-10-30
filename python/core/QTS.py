from utils.init_state import gen_nbrs
from utils.topology import decode_and_synthesize,verify_circuit_logic




def QTS_run_single_experiment(max_iterations, 
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
    Executes a single trial of the QTS algorithm for quantum circuit synthesis.
    Iteratively evolves the quantum probability distributions (qindividuals1-4) 
    to converge toward the minimum gate count solution.
    """
    
    num_cycles = len(rotation_cycles)
    current_iter = 0
    global_best_gate_count = float('inf')
    global_best_circuit = []

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
        # Analyze the gate count of each synthesized solution to determine its quality.
        solution_metrics = [(sol[1], idx) for idx, sol in enumerate(circuit_solutions)]
        
        # Sort candidates by gate count (Ascending) to identify the elite and weak performers.
        sorted_metrics = sorted(solution_metrics, key=lambda x: x[0])
        
        local_best_gate_count = sorted_metrics[0][0]
        local_best_idx = sorted_metrics[0][1]
        local_best_circuit = circuit_solutions[local_best_idx]

        # Step 4: QTS Quantum State Update
        # Apply the 'best-vs-worst' strategy to rotate the quantum states.
        # This shifts the probability toward the elite neighbor and away from the poor one.
        updateQ(
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, 
            num_neighbors, nbr1, nbr2, nbr3, nbr4, 
            [m[1] for m in sorted_metrics], num_cycles ,delta_theta
        )

        # Step 5: Global Best Tracking
        # Retain the historical best circuit configuration found by the QTS process.
        if global_best_gate_count > local_best_gate_count:
            global_best_gate_count = local_best_gate_count
            global_best_circuit = local_best_circuit

        # Step 6: Data Recording for Convergence Analysis
        # Store the current global minimum gate count into the fitness history matrix.
        fitness_history_matrix[experiment_id][current_iter - 1] = global_best_gate_count

    # Returns the convergence history and the optimal circuit found via QTS optimization.
    return fitness_history_matrix, global_best_gate_count, global_best_circuit

def updateQ(qindividuals1, qindividuals2, qindividuals3, qindividuals4, 
                               num_neighbors, nbr1, nbr2, nbr3, nbr4, 
                               sorted_indices, num_cycles, delta_theta):
    """
    Updates the probability distributions of quantum populations (q1-q4) based on 
    the competitive 'best-vs-worst' interaction in QTS. 
    Shifts probability amplitudes to favor high-fitness discrete decisions.
    """
    
    # Step 1: Identify Best and Worst neighbor pairs from the current swarm
    best_idx = sorted_indices[0]
    worst_idx = sorted_indices[num_neighbors - 1]

    # --- Update qindividuals1 (Strategy/Trajectory level) ---
    # Shift probabilities based on the global trajectory decisions in QTS
    best_sol1 = [list(map(int, row)) for row in nbr1[best_idx].tolist()]
    worst_sol1 = [list(map(int, row)) for row in nbr1[worst_idx].tolist()]

    for i in range(len(qindividuals1)):
        for j in range(len(qindividuals1[i])):
            b_val = best_sol1[i][j]
            w_val = worst_sol1[i][j]
            if b_val != w_val:
                qindividuals1[i][j][b_val] += delta_theta
                qindividuals1[i][j][w_val] -= delta_theta
                
                # QTS Boundary Correction
                if qindividuals1[i][j][w_val] <= 0:
                    qindividuals1[i][j][b_val] = 1.0
                    qindividuals1[i][j][w_val] = 0.0

    # --- Update qindividuals2 (Segment level) ---
    best_sol2 = nbr2[best_idx]
    worst_sol2 = nbr2[worst_idx]

    for i in range(len(qindividuals2)):
        for j in range(len(qindividuals2[i])):
            if best_sol2[i][j] != worst_sol2[i][j]:
                qindividuals2[i][j][best_sol2[i][j]] += delta_theta
                qindividuals2[i][j][worst_sol2[i][j]] -= delta_theta
                if qindividuals2[i][j][worst_sol2[i][j]] <= 0:
                    qindividuals2[i][j][best_sol2[i][j]] = 1.0
                    qindividuals2[i][j][worst_sol2[i][j]] = 0.0

    # --- Update qindividuals3 (Route/Path level) ---
    # Update QTS path node probabilities while respecting 999 fixed topology
    best_sol3 = nbr3[best_idx]
    worst_sol3 = nbr3[worst_idx]

    for i in range(len(qindividuals3)):
        for j in range(len(best_sol3[i])):
            for k in range(len(best_sol3[i][j])):
                if best_sol3[i][j][k] != 999 and worst_sol3[i][j][k] != 999:
                    num_choices = len(qindividuals3[i][j][k])
                    for l in range(num_choices):
                        b_bit = best_sol3[i][j][k][l]
                        w_bit = worst_sol3[i][j][k][l]
                        if b_bit != w_bit:
                            qindividuals3[i][j][k][l][b_bit] += delta_theta
                            qindividuals3[i][j][k][l][w_bit] -= delta_theta
                            if qindividuals3[i][j][k][l][w_bit] <= 0:
                                qindividuals3[i][j][k][l][b_bit] = 1.0
                                qindividuals3[i][j][k][l][w_bit] = 0.0

    # --- Update qindividuals4 (Gate Order level) ---
    # Finalize QTS gate sequence distribution based on elite performance
    best_sol4 = nbr4[best_idx]
    worst_sol4 = nbr4[worst_idx]

    for i in range(num_cycles):
        for j in range(len(best_sol4[i])):
            if len(best_sol4[i][j]) > 1:
                for k in range(len(best_sol4[i][j])):
                    b_gate = best_sol4[i][j][k]
                    w_gate = worst_sol4[i][j][k]
                    if b_gate != w_gate:
                        qindividuals4[i][j][k][b_gate] += delta_theta
                        qindividuals4[i][j][k][w_gate] -= delta_theta
                        if qindividuals4[i][j][k][w_gate] <= 0:
                            qindividuals4[i][j][k][b_gate] = 1.0
                            qindividuals4[i][j][k][w_gate] = 0.0


          
   