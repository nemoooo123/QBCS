from utils.init_state import gen_nbrs
from utils.topology import decode_and_synthesize,verify_circuit_logic




def AE_QTS_run_single_experiment(max_iterations, 
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
    Executes a single trial of the AE-QTS algorithm.
    Iteratively updates quantum individuals (qindividuals1-4) to minimize circuit gate count.
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

    while current_iter < max_iterations:
        current_iter += 1

        # Step 1: Neighborhood Generation
        # Create candidate solutions (neighbors) based on the current quantum state of qindividuals
        nbr1, nbr2, nbr3, nbr4 = gen_nbrs(
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, num_neighbors
        )

        # Step 2: Decoding and Circuit Synthesis
        # Convert quantum neighbors into concrete reversible circuit solutions
        circuit_solutions = decode_and_synthesize(
            nbr1, nbr2, nbr3, nbr4, encoding_table, num_bits, num_neighbors, base_trajectory
        )
        # Step 3: Fitness Evaluation (Gate Count Analysis)
        # Pair each solution's gate count with its original neighborhood index
        # Lower scores indicate superior individuals in our minimization framework.
        solution_metrics = [(sol_tuple[1], idx) for idx, sol_tuple in enumerate(circuit_solutions)]
        # Sort by gate count in ascending order to identify the local optimal neighbor
        sorted_metrics = sorted(solution_metrics, key=lambda x: x[0])
        local_best_idx = sorted_metrics[0][1]
        local_best_gate_count = circuit_solutions[local_best_idx][1]
        local_best_circuit = circuit_solutions[local_best_idx][0]
        # Step 4: Quantum Population Update (Angle Adjustment)
        # Update the probability amplitudes of qindividuals based on neighbor performance
        updateQ(
            qindividuals1, qindividuals2, qindividuals3, qindividuals4, 
            num_neighbors, nbr1, nbr2, nbr3, nbr4, 
            [m[1] for m in sorted_metrics], num_cycles, delta_theta
        )

        # Step 5: Global Best Tracking
        # Update the overall best solution if a new minimum gate count is discovered
        if global_best_gate_count > local_best_gate_count:
            global_best_gate_count = local_best_gate_count
            global_best_circuit = local_best_circuit
        # Step 6: Integrity Verification
        # Check if the synthesized circuits fulfill the logic requirements for the target output
        # valid_count = sum(verify_circuit_logic(sol, num_bits, target_output) for sol in circuit_solutions)
        
        # if valid_count != num_neighbors:
        #     print(f"Warning: Logic verification failed for {num_neighbors - valid_count} neighbors.")

        # Step 7: Data Recording for Statistical Analysis
        # Record the current best gate count into the fitness history matrix (used for np.mean later)
        fitness_history_matrix[experiment_id][current_iter - 1] = global_best_gate_count


    return fitness_history_matrix, global_best_gate_count, global_best_circuit

def updateQ(qindividuals1, qindividuals2, qindividuals3, qindividuals4,
                               num_neighbors, nbr1, nbr2, nbr3, nbr4, 
                               sorted_indices, num_cycles, delta_theta):
    """
    Updates the probability distributions of quantum populations (q1-q4) based on 
    the relative fitness of their neighbors. 
    Uses a 'best-vs-worst' strategy to shift probability amplitudes.
    """
    
    # Process the top half of the population (Best half pulls away from Worst half)
    t = 0
    limit = int(num_neighbors / 2)
    
    while t < limit:
        # Step 1: Identify Best and Worst neighbor pairs
        # best_idx: index of the t-th best neighbor
        # worst_idx: index of the t-th worst neighbor
        best_idx = sorted_indices[t]
        worst_idx = sorted_indices[num_neighbors - 1 - t]

        # --- Update qindividuals1 (Strategy/Trajectory level) ---
        # best_sol1 and worst_sol1 represent the discrete decisions made by the neighbors
        # nbr1 is an int64 ndarray, so .tolist() already yields native Python ints -
        # the extra list(map(int, row)) below was re-wrapping values that were already ints.
        best_sol1 = nbr1[best_idx].tolist()
        worst_sol1 = nbr1[worst_idx].tolist()

        for i in range(len(qindividuals1)):
            for j in range(len(qindividuals1[i])):
                b_val = best_sol1[i][j]
                w_val = worst_sol1[i][j]
                if b_val != w_val:
                    # Increment probability of the 'best' decision, decrement the 'worst'
                    qindividuals1[i][j][b_val] += delta_theta/(t+1)
                    qindividuals1[i][j][w_val] -= delta_theta/(t+1)
                    
                    # Boundary Correction: Ensure probabilities stay within [0, 1]
                    if qindividuals1[i][j][w_val] <= 0:
                        qindividuals1[i][j][b_val] = 1.0
                        qindividuals1[i][j][w_val] = 0.0

        # --- Update qindividuals2 (Segment level) ---
        best_sol2 = nbr2[best_idx]
        worst_sol2 = nbr2[worst_idx]

        for i in range(len(qindividuals2)):
            for j in range(len(qindividuals2[i])):
                if best_sol2[i][j] != worst_sol2[i][j]:
                    qindividuals2[i][j][best_sol2[i][j]] += delta_theta/(t+1)
                    qindividuals2[i][j][worst_sol2[i][j]] -= delta_theta/(t+1)
                    if qindividuals2[i][j][worst_sol2[i][j]] <= 0:
                        qindividuals2[i][j][best_sol2[i][j]] = 1.0
                        qindividuals2[i][j][worst_sol2[i][j]] = 0.0

        # --- Update qindividuals3 (Route/Path level) ---
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
                                qindividuals3[i][j][k][l][b_bit] += delta_theta/(t+1)
                                qindividuals3[i][j][k][l][w_bit] -= delta_theta/(t+1)
                                if qindividuals3[i][j][k][l][w_bit] <= 0:
                                    qindividuals3[i][j][k][l][b_bit] = 1.0
                                    qindividuals3[i][j][k][l][w_bit] = 0.0

        # --- Update qindividuals4 (Gate Order level) ---
        best_sol4 = nbr4[best_idx]
        worst_sol4 = nbr4[worst_idx]

        for i in range(num_cycles):
            for j in range(len(best_sol4[i])):
                # Only update if the gate sequence has more than one bit (non-trivial)
                if len(best_sol4[i][j]) > 1:
                    for k in range(len(best_sol4[i][j])):
                        b_gate = best_sol4[i][j][k]
                        w_gate = worst_sol4[i][j][k]
                        if b_gate != w_gate:
                            qindividuals4[i][j][k][b_gate] += delta_theta/(t+1)
                            qindividuals4[i][j][k][w_gate] -= delta_theta/(t+1)
                            if qindividuals4[i][j][k][w_gate] <= 0:
                                qindividuals4[i][j][k][b_gate] = 1.0
                                qindividuals4[i][j][k][w_gate] = 0.0
        t += 1

          
   