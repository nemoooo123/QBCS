import numpy as np
import copy
from utils.init_state import  repair_sequence_logic
from utils.topology import decode_and_synthesize, verify_circuit_logic

def PSO_run_single_experiment(
    max_iterations, rotation_cycles, num_neighbors, num_bits, base_trajectory,
    experiment_id, encoding_table, pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4,
    fitness_history_matrix, target_output, w, c1, c2
):
    """
    Execute a single PSO experiment: Each neighbor is treated as a particle 
    with a continuous position and velocity vector across four encoding layers.
    """
    
    # --- Step 1: Initialization ---
    # Generate initial particle states (positions, velocities, and discrete solutions)
    (curr_nbr1, vel1, discrete_nbr1,
     curr_nbr2, vel2, discrete_nbr2,
     curr_nbr3, vel3, discrete_nbr3,
     curr_nbr4, vel4, discrete_nbr4) = gen_pso_init_states(
        pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, num_neighbors
    )

    # Initialize Personal Best (pBest) positions. 
    # Must record continuous values to maintain momentum in the velocity formula.
    pbest_n1, pbest_n2, pbest_n3, pbest_n4 = (
        copy.deepcopy(curr_nbr1), copy.deepcopy(curr_nbr2), 
        copy.deepcopy(curr_nbr3), copy.deepcopy(curr_nbr4)
    )
    
    
    # Decode and synthesize circuits to calculate initial population fitness
    circuit_solutions = decode_and_synthesize(
        discrete_nbr1, discrete_nbr2, discrete_nbr3, discrete_nbr4, 
        encoding_table, num_bits, num_neighbors, base_trajectory
    )
    # Define fitness as the gate count (Optimization goal: Minimize gate count)
    pbest_fitness = [sol[1] for sol in circuit_solutions]
    
    # Initialize Global Best (gBest)
    gbest_idx = np.argmin(pbest_fitness) # Find the index of the particle with the minimum gates
    gbest_gate_count = pbest_fitness[gbest_idx] # Record the minimum gate count found so far
    
    # Perform a deep copy of the global best position across all four layers
    gbest_n1, gbest_n2, gbest_n3, gbest_n4 = (
        copy.deepcopy(pbest_n1[gbest_idx]), copy.deepcopy(pbest_n2[gbest_idx]),
        copy.deepcopy(pbest_n3[gbest_idx]), copy.deepcopy(pbest_n4[gbest_idx])
    )
    gbest_circuit = circuit_solutions[gbest_idx] # Store the best circuit structure found
    
    current_iter = 0
    # --- Step 2: Swarm Evolution Loop ---
    while current_iter < max_iterations:
        current_iter += 1

        for i in range(num_neighbors):
            # Apply stochastic factors r1, r2 to increase search exploration
            
            # Update Layer 1 (Global Selection): Sync Pos, Vel, and Discrete bits
            vel1[i], curr_nbr1[i], discrete_nbr1[i] = pso_recursive_sync_update(
                vel1[i], curr_nbr1[i], pbest_n1[i], gbest_n1, w, c1, c2 ,True
            )   

            # Update Layer 2 (Entry Points)
            vel2[i], curr_nbr2[i], discrete_nbr2[i] = pso_recursive_sync_update(
                vel2[i], curr_nbr2[i], pbest_n2[i], gbest_n2, w, c1, c2 ,True
            )   

            # Update Layer 3 (Intermediate Path Nodes)
            vel3[i], curr_nbr3[i], discrete_nbr3[i] = pso_recursive_sync_update(
                vel3[i], curr_nbr3[i], pbest_n3[i], gbest_n3, w, c1, c2 ,True
            )   

            # Update Layer 4 (Gate Sequencing): Convert 999 markers to 0 for sequence processing
            vel4[i], curr_nbr4[i], discrete_nbr4[i] = pso_recursive_sync_update(
                vel4[i], curr_nbr4[i], pbest_n4[i], gbest_n4, w, c1, c2 ,False
            )      
            # Post-Update Constraint Satisfaction: Repair gate sequences in Layer 4
            for cycle_idx in range(len(discrete_nbr4[i])):
                for step_idx in range(len(discrete_nbr4[i][cycle_idx])):
                    # Perform logic repair on the innermost bit sequence
                    repaired_bits = repair_sequence_logic(discrete_nbr4[i][cycle_idx][step_idx])
                    discrete_nbr4[i][cycle_idx][step_idx] = repaired_bits
            
        # Synthesize updated circuits and evaluate new fitness values for the current population
        circuit_solutions = decode_and_synthesize(
            discrete_nbr1, discrete_nbr2, discrete_nbr3, discrete_nbr4, 
            encoding_table, num_bits, num_neighbors, base_trajectory
        )
        
        # # Integrity Verification
        # # Check if the synthesized circuits fulfill the logic requirements for the target output
        # valid_count = sum(verify_circuit_logic(sol, num_bits, target_output) for sol in circuit_solutions)
        
        # if valid_count != num_neighbors:
        #     print(f"Warning: Logic verification failed for {num_neighbors - valid_count} neighbors.")

        # Identify local best of the current iteration (Iteration Best)
        lbest_fitness = [sol[1] for sol in circuit_solutions]

        # Identify local best of the current iteration (Iteration Best)
        lbest_idx = np.argmin(lbest_fitness) 
        lbest_gate_count = lbest_fitness[lbest_idx] 
        
        # --- Step 3: Update Personal Best (pBest) and Global Best (gBest) ---
        for i in range(num_neighbors):
            if lbest_fitness[i] < pbest_fitness[i]:
                pbest_fitness[i] = lbest_fitness[i]  # Update gBest fitness score
                # Update gBest position data (continuous coordinates)
                pbest_n1[i], pbest_n2[i], pbest_n3[i], pbest_n4[i] = (
                    copy.deepcopy(curr_nbr1[i]), copy.deepcopy(curr_nbr2[i]),
                    copy.deepcopy(curr_nbr3[i]), copy.deepcopy(curr_nbr4[i])
                )
            
        
        # If the best particle in this iteration outperforms the global best
        if lbest_gate_count < gbest_gate_count:
            gbest_gate_count = lbest_gate_count # Update gBest fitness score
            gbest_circuit = circuit_solutions[lbest_idx]
            # Update gBest position data (continuous coordinates)
            gbest_n1, gbest_n2, gbest_n3, gbest_n4 = (
                copy.deepcopy(curr_nbr1[lbest_idx]), copy.deepcopy(curr_nbr2[lbest_idx]),
                copy.deepcopy(curr_nbr3[lbest_idx]), copy.deepcopy(curr_nbr4[lbest_idx])
            )

        # Log the global best gate count for the current iteration into the history matrix
        fitness_history_matrix[experiment_id][current_iter - 1] = gbest_gate_count
        
    # Return fitness history, final global best gate count, and the optimized circuit structure
    return fitness_history_matrix, gbest_gate_count, gbest_circuit


def pso_recursive_sync_update(v, x, pbest, gbest, w, c1, c2, keep_999):
    """
    Recursively updates PSO velocity and position, and generates discrete solutions 
    simultaneously for multi-layered nested structures.
    
    This function handles heterogeneous list depths and preserves or transforms 
    special markers (999) based on the target layer's requirements.
    """

    # Case A: Handling Nested Structures (Recursive Depth)
    if isinstance(x, list):
        new_v_list, new_x_list, new_d_list = [], [], []
        for i in range(len(x)):
            # Recursive call to traverse deeper into the nested hierarchy
            res_v, res_x, res_d = pso_recursive_sync_update(
                v[i], x[i], pbest[i], gbest[i], w, c1, c2 ,keep_999
            )
            new_v_list.append(res_v)
            new_x_list.append(res_x)
            new_d_list.append(res_d)
        return new_v_list, new_x_list, new_d_list

    # Case B: Handling Special Identifier 999 (Inertial or Fixed Points)
    if x == 999:
        # If keep_999 is True, preserve the marker for structural logic (e.g., Layer 3 skip-nodes)
        # If False, convert to 0 to satisfy bit-sequence requirements (e.g., Layer 4 sequencing)
        if keep_999:
            return 999, 999, 999
        else:
            return 999, 999, 0

    # Case C: Reached Leaf Node (Numerical values), Execute PSO Kinematics
    r1, r2 = np.random.rand(), np.random.rand()
    
    # 1. Velocity Update with Clamping within [-6, 6]
    # v_new = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    new_v = (w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x))
    new_v = max(min(new_v, 6), -6)
    
    # 2. Continuous Position Update
    # x_new = x + v_new
    new_x = x + new_v
    
    # 3. Discrete Mapping based on Sigmoid Threshold of 0.5
    # Mathematically equivalent to checking if the continuous position is positive
    new_d = 1 if new_x > 0 else 0
    
    return new_v, new_x, new_d

def gen_pso_init_states(pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, population_size):
    """
    Initialization function specifically designed for Particle Swarm Optimization (PSO).
    Returns continuous random velocity matrices with the same structure as positions (range [-6, 6]).
    """
    # Particle swarm position containers (Continuous values for optimization)
    pso_positions_l1, pso_positions_l2, pso_positions_l3, pso_positions_l4 = [], [], [], []
    
    # Particle swarm velocity containers (Movement vectors)
    pso_velocities_l1, pso_velocities_l2, pso_velocities_l3, pso_velocities_l4 = [], [], [], []
    
    # Discretized solution containers (Mapped 0/1 bits for circuit synthesis)
    discrete_solutions_l1, discrete_solutions_l2, discrete_solutions_l3, discrete_solutions_l4 = [], [], [], []
    
    for _ in range(population_size):
        # Layer 1: Global selection layer
        pos1, v1, disc1 = pso_layer_L1(pop_matrix1)
        pso_positions_l1.append(pos1)
        pso_velocities_l1.append(v1)
        discrete_solutions_l1.append(disc1)

        # Layer 2: Entry point selection layer
        pos2, v2, disc2 = pso_layer_L2(pop_matrix2)
        pso_positions_l2.append(pos2)
        pso_velocities_l2.append(v2)
        discrete_solutions_l2.append(disc2)

        # Layer 3: Intermediate path node layer
        pos3, v3, disc3 = pso_layer_L3(pop_matrix3)
        pso_positions_l3.append(pos3)
        pso_velocities_l3.append(v3)
        discrete_solutions_l3.append(disc3)

        # Layer 4: Gate sequence ordering layer
        pos4, v4, disc4 = pso_layer_L4(pop_matrix4)
        pso_positions_l4.append(pos4)
        pso_velocities_l4.append(v4)
        discrete_solutions_l4.append(disc4)
        
    # Return all hierarchical data as a tuple
    return (pso_positions_l1, pso_velocities_l1, discrete_solutions_l1,
            pso_positions_l2, pso_velocities_l2, discrete_solutions_l2,
            pso_positions_l3, pso_velocities_l3, discrete_solutions_l3,
            pso_positions_l4, pso_velocities_l4, discrete_solutions_l4)

def pso_layer_L1(population_templates): 
    """
    Initialize Particle Swarm Optimization (PSO) Layer 1 components.
    Generates:
    1. Continuous initial positions within [-6, 6].
    2. Initial velocity vectors within [-1, 1].
    3. Discrete binary solutions based on a 0.5 sigmoid threshold (x > 0).
    """
    # initial_positions: Latent coordinates for the swarm to "fly" through (Continuous)
    initial_positions = [] 
    # initial_velocities: Movement vectors for the particles
    initial_velocities = []
    # initial_discrete_solutions: The actual 0/1 bits used for circuit synthesis (Discrete)
    initial_discrete_solutions = []

    # Iterate through each cycle's encoding structure
    for cycle_idx in range(len(population_templates)): 
        pos_code = []
        vel_code = []
        discrete_code = []
        
        # Iterate through each bit position defined in the template
        for bit_idx in range(len(population_templates[cycle_idx])): 
            # 1. Initialize Continuous Position: [-6, 6]
            stochastic_pos = np.random.uniform(-6, 6)
            
            # 2. Initialize Velocity: [-1, 1]
            stochastic_vel = np.random.uniform(-1, 1)
            
            # 3. Generate Discrete Solution: 
            # Based on your requirement: If sigmoid(x) > 0.5 (which means x > 0), set to 1.
            discrete_bit = 1 if stochastic_pos > 0 else 0
            
            pos_code.append(stochastic_pos)
            vel_code.append(stochastic_vel)
            discrete_code.append(discrete_bit)
        
        initial_positions.append(pos_code)
        initial_velocities.append(vel_code)
        initial_discrete_solutions.append(discrete_code)
    
    # Return all three components for the PSO population initialization
    return initial_positions, initial_velocities, initial_discrete_solutions

def pso_layer_L2(entry_point_templates):
    """
    Initialize Particle Swarm Optimization (PSO) Layer 2 components.
    Generates:
    1. Continuous initial positions within [-6, 6] for entry points.
    2. Initial velocity vectors within [-1, 1].
    3. Discrete entry point bits based on a 0.5 sigmoid threshold (x > 0).
    """
    # initial_positions: Latent coordinates for entry point optimization
    initial_positions = []
    # initial_velocities: Movement vectors for the swarm to explore the entry node space
    initial_velocities = []
    # initial_discrete_solutions: The binary entry point bits (0/1) for circuit synthesis
    initial_discrete_solutions = []
    
    # Iterate through each cycle's entry point structure in Layer 2
    for cycle_idx in range(len(entry_point_templates)):
        pos_code = []
        vel_code = []
        discrete_code = []
        
        # Iterate through each bit defined in the entry point template 
        for _ in entry_point_templates[cycle_idx]:
            # 1. Initialize Continuous Position: Sigmoid sensitivity range [-6, 6]
            stochastic_pos = np.random.uniform(-6, 6)
            
            # 2. Initialize Velocity: Initial search momentum [-1, 1]
            stochastic_vel = np.random.uniform(-1, 1)
            
            # 3. Generate Discrete Solution: 
            # Sigmoid(x) > 0.5 is mathematically equivalent to x > 0
            discrete_bit = 1 if stochastic_pos > 0 else 0
            
            pos_code.append(stochastic_pos)
            vel_code.append(stochastic_vel)
            discrete_code.append(discrete_bit)

        initial_positions.append(pos_code)
        initial_velocities.append(vel_code)
        initial_discrete_solutions.append(discrete_code)
    
    # Return synchronized Position, Velocity, and Discrete Solution for Layer 2 Swarm
    return initial_positions, initial_velocities, initial_discrete_solutions

def pso_layer_L3(path_node_templates):
    """
    Initialize Particle Swarm Optimization (PSO) Layer 3 components.
    Generates:
    1. Continuous positions within [-6, 6] for path nodes.
    2. Initial velocity vectors within [-1, 1].
    3. Discrete node bits (0/1) based on a 0.5 sigmoid threshold (x > 0).
    Special markers (999) are preserved across all structures.
    """
    # initial_positions: Latent coordinates for the swarm to explore path combinations
    initial_positions = []
    # initial_velocities: Trajectory vectors for the path optimization
    initial_velocities = []
    # initial_discrete_solutions: Binary node representations for circuit logic evaluation
    initial_discrete_solutions = []

    # Iterate through each cycle component in Layer 3
    for cycle_idx in range(len(path_node_templates)):
        cycle_pos, cycle_vel, cycle_disc = [], [], []
        
        # Iterate through each transition step within the cycle
        for step_idx in range(len(path_node_templates[cycle_idx])):
            step_pos, step_vel, step_disc = [], [], []
            
            # Determine if this step is a direct transition (999) or requires intermediate nodes
            substep_count = len(path_node_templates[cycle_idx][step_idx])
            
            for node_template in path_node_templates[cycle_idx][step_idx]:
                if substep_count == 1:
                    # Identifier 999: Direct Hamming distance jump. 
                    # Preserve 999 in all three outputs to maintain structural alignment.
                    step_pos.append(999)
                    step_vel.append(999)
                    step_disc.append(999)
                else:
                    node_pos, node_vel, node_disc = [], [], []
                    
                    # Perform bit-wise initialization for the node
                    for _ in node_template:
                        # 1. Continuous Position: [-6, 6]
                        stochastic_pos = np.random.uniform(-6, 6)
                        # 2. Velocity: [-1, 1]
                        stochastic_vel = np.random.uniform(-1, 1)
                        # 3. Discrete Solution: x > 0 is equivalent to Sigmoid(x) > 0.5
                        discrete_bit = 1 if stochastic_pos > 0 else 0
                        
                        node_pos.append(stochastic_pos)
                        node_vel.append(stochastic_vel)
                        node_disc.append(discrete_bit)
                    
                    step_pos.append(node_pos)
                    step_vel.append(node_vel)
                    step_disc.append(node_disc)
            
            cycle_pos.append(step_pos)
            cycle_vel.append(step_vel)
            cycle_disc.append(step_disc)
            
        initial_positions.append(cycle_pos)
        initial_velocities.append(cycle_vel)
        initial_discrete_solutions.append(cycle_disc)

    # Return synchronized Continuous Position, Velocity, and Discrete Solution for Layer 3
    return initial_positions, initial_velocities, initial_discrete_solutions

def pso_layer_L4(sequence_templates):
    """
    Initialize Particle Swarm Optimization (PSO) Layer 4 components.
    Generates:
    1. Continuous initial positions within [-6, 6] for sequencing decisions.
    2. Initial velocity vectors within [-1, 1].
    3. Repaired discrete binary solutions (0/1) for valid gate ordering.
    """
    # initial_positions: Latent coordinates for optimizing gate sequences
    initial_positions = []
    # initial_velocities: Trajectory vectors for the sequencing optimization
    initial_velocities = []
    # initial_discrete_solutions: Validated (repaired) binary sequences for circuit synthesis
    initial_discrete_solutions = []
    
    # Iterate through each cycle's sequencing structure in Layer 4
    for cycle_idx in range(len(sequence_templates)):
        cycle_pos, cycle_vel, cycle_disc = [], [], []
        
        # Iterate through each transition step within the cycle
        for step_idx in range(len(sequence_templates[cycle_idx])):
            step_pos, step_vel, step_raw_disc = [], [], []
            
            # Iterate through the sequencing bits defined in the template
            for bit_template in sequence_templates[cycle_idx][step_idx]:
                # Check for the 999 fixed-sequence identifier
                if bit_template[0] == 999:
                    stochastic_pos = 999
                    stochastic_vel = 999
                    discrete_bit = 0 # Default state for fixed sequences (per original logic)
                else:
                    # 1. Initialize Position: [-6, 6]
                    stochastic_pos = np.random.uniform(-6, 6)
                    # 2. Initialize Velocity: [-1, 1]
                    stochastic_vel = np.random.uniform(-1, 1)
                    # 3. Initial Discrete Bit: Based on 0.5 threshold (x > 0)
                    discrete_bit = 1 if stochastic_pos > 0 else 0
                
                step_pos.append(stochastic_pos)
                step_vel.append(stochastic_vel)
                step_raw_disc.append(discrete_bit)
            
            # --- IMPORTANT: Execute Sequence Repair Logic ---
            # Ensure that the initialized binary sequences comply with circuit logic 
            # (e.g., maintaining the forward/reverse balance of quantum gates).
            repaired_disc = repair_sequence_logic(step_raw_disc)
            
            cycle_pos.append(step_pos)
            cycle_vel.append(step_vel)
            cycle_disc.append(repaired_disc)
            
        initial_positions.append(cycle_pos)
        initial_velocities.append(cycle_vel)
        initial_discrete_solutions.append(cycle_disc)
    
    # Return synchronized Position, Velocity, and Validated Discrete Solution for Layer 4
    return initial_positions, initial_velocities, initial_discrete_solutions