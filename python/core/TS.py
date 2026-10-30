from utils.init_state import gen_nbrs, repair_sequence_logic
from utils.topology import decode_and_synthesize, synthesize_route, verify_circuit_logic
import copy
import random

def generate_ts_neighbors(c1, c2, c3, c4, encoding_table, num_bits, num_neighbors, base_trajectory):
    """
    Neighborhood generation function: Performs bit-flips across all four layers 
    simultaneously to create 'Compound Moves.' 
    
    This approach increases search step-size and exploration depth within 
    the discrete solution space.
    """
    # Lists to store the structure of each neighbor for batch synthesis
    nbr1, nbr2, nbr3, nbr4 = [], [], [], []
    candidates = []

    for _ in range(num_neighbors):
        # List to track specific bit-flips for the Tabu identifier
        move_tags = []
        
        # Deep copy the current base state to create an independent neighbor
        nc1, nc2, nc3, nc4 = copy.deepcopy(c1), copy.deepcopy(c2), copy.deepcopy(c3), copy.deepcopy(c4)

        # --- Layer 1 Move (Global Selection) ---
        # Flip a random bit in the Cycle/Bit structure
        cy1 = random.randint(0, len(nc1) - 1)
        bi1 = random.randint(0, len(nc1[cy1]) - 1)
        nc1[cy1][bi1] = 1 - nc1[cy1][bi1]
        move_tags.append(f"L1_{cy1}_{bi1}")

        # --- Layer 2 Move (Entry Points) ---
        # Flip a random bit in the entry point selection
        cy2 = random.randint(0, len(nc2) - 1)
        bi2 = random.randint(0, len(nc2[cy2]) - 1)
        nc2[cy2][bi2] = 1 - nc2[cy2][bi2]
        move_tags.append(f"L2_{cy2}_{bi2}")

        # --- Layer 3 Move (Intermediate Path Nodes) ---
        # Navigate the 4-level nested structure: [Cycle][Step][Node][Bit]
        cy3 = random.randint(0, len(nc3) - 1)
        st3 = random.randint(0, len(nc3[cy3]) - 1)
        no3 = random.randint(0, len(nc3[cy3][st3]) - 1)
        
        # Safety check: Only perform flip if the node is not a fixed 999 marker
        if nc3[cy3][st3][no3] != 999:
            bi3 = random.randint(0, len(nc3[cy3][st3][no3]) - 1)
            nc3[cy3][st3][no3][bi3] = 1 - nc3[cy3][st3][no3][bi3]
            move_tags.append(f"L3_{cy3}_{st3}_{no3}_{bi3}")
        else:
            # Record as a Fixed Point if a 999 marker was selected
            move_tags.append(f"L3_{cy3}_{st3}_{no3}_FIX")

        # --- Layer 4 Move (Gate Sequencing) ---
        # Flip a sequencing bit and immediately apply logic repair
        cy4 = random.randint(0, len(nc4) - 1)
        st4 = random.randint(0, len(nc4[cy4]) - 1)
        bi4 = random.randint(0, len(nc4[cy4][st4]) - 1)
        nc4[cy4][st4][bi4] = 1 - nc4[cy4][st4][bi4]
        
        # Crucial: Repair the gate sequence to ensure circuit validity
        nc4[cy4][st4] = repair_sequence_logic(nc4[cy4][st4])   
        move_tags.append(f"L4_{cy4}_{st4}_{bi4}")

        # Append the new state to the neighbor lists
        nbr1.append(nc1)
        nbr2.append(nc2)
        nbr3.append(nc3)
        nbr4.append(nc4)
        
        # Combine all layer tags into a unique Tabu Move Identifier
        combined_move_id = "|".join(move_tags)
        candidates.append(((nc1, nc2, nc3, nc4), combined_move_id))
    
    # Batch synthesize all neighbors to optimize evaluation speed
    circuit_solutions = decode_and_synthesize(
            nbr1, nbr2, nbr3, nbr4, encoding_table, num_bits, num_neighbors, base_trajectory
        )
        
    return circuit_solutions, candidates

def TS_run_single_experiment(
                max_iterations, rotation_cycles, num_neighbors, num_bits, base_trajectory,
                experiment_id, encoding_table, pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4,
                fitness_history_matrix, target_output, tabu_size
                ):
    """
    Executes a single Tabu Search experiment: A local search metaheuristic 
    that uses a tabu list to prevent cycling and escape local optima.
    """
    
    # --- Step 1: Initialize current solution ---
    # Generate initial discrete states (neighbors) from population matrices
    # We only need one starting point for TS, so we take index 0
    init_1, init_2, init_3, init_4 = gen_nbrs(pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4, 1)
    curr_c1, curr_c2, curr_c3, curr_c4 = init_1[0], init_2[0], init_3[0], init_4[0]

    # Nested function to evaluate fitness for a single individual
    def evaluate_fitness(c1, c2, c3, c4):
        """Synthesizes the circuit and returns the gate count and circuit structure."""
        sol = decode_and_synthesize([c1], [c2], [c3], [c4], 
                                     encoding_table, num_bits, 1, base_trajectory)
        return len(sol[0][0]), sol[0][0]

    # Calculate initial fitness and store as the global best for this run
    best_fitness, best_circuit = evaluate_fitness(curr_c1, curr_c2, curr_c3, curr_c4)
    
    # Track the best global state found during the search
    gbest_c1, gbest_c2, gbest_c3, gbest_c4 = (
        copy.deepcopy(curr_c1), copy.deepcopy(curr_c2), 
        copy.deepcopy(curr_c3), copy.deepcopy(curr_c4)
    )
    
    tabu_list = [] # FIFO queue to store forbidden move identifiers
    current_iter = 0

    # --- Step 2: Tabu Search Iteration Loop ---
    while current_iter < max_iterations:
        current_iter += 1
        
        # Generate N neighbors and their synthesized circuits in a single batch
        circuit_solutions, candidates = generate_ts_neighbors(
            curr_c1, curr_c2, curr_c3, curr_c4, encoding_table, num_bits, num_neighbors, base_trajectory
        )
        #Integrity Verification
        # Check if the synthesized circuits fulfill the logic requirements for the target output
        # valid_count = sum(verify_circuit_logic(sol, num_bits, target_output) for sol in circuit_solutions)
        
        # if valid_count != num_neighbors:
        #     print(f"Warning: Logic verification failed for {num_neighbors - valid_count} neighbors.")

        # --- Step 3: Candidate Evaluation ---
        scored_candidates = []
        # Pair each candidate state with its move_id, fitness score, and synthesized circuit
        
        for idx in range(len(candidates)):
            state, move_id = candidates[idx]
            circuit = circuit_solutions[idx][0]
            
            # Optimization Goal: Minimize gate count
            fitness = len(circuit)
            scored_candidates.append((state, move_id, fitness, circuit))
            
        # Sort candidates by fitness in ascending order (best performing first)
        scored_candidates.sort(key=lambda x: x[2])
        
        # --- Step 4: Selection with Tabu and Aspiration Criterion ---
        found_next_move = False
        for next_state, move_id, next_fitness, next_circuit in scored_candidates:
            # Aspiration Criterion: Accept a tabu move if it yields a new global best
            is_aspiring = next_fitness < best_fitness
            
            if (move_id not in tabu_list) or is_aspiring:
                # Accept the move and update the current search position
                curr_c1, curr_c2, curr_c3, curr_c4 = next_state
                
                # Update the record-breaking global best
                if next_fitness < best_fitness:
                    best_fitness = next_fitness
                    best_circuit = next_circuit
                    # Deep copy best coordinates (Note: ensure all layers are copied if needed)
                
                # Update Tabu List (First-In, First-Out mechanism)
                tabu_list.append(move_id)
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0) 
                
                found_next_move = True
                break # Exit the candidate selection once a valid move is found
        
        # Fallback Logic: If all moves are tabu and none break the record, 
        # force accept the best available candidate to maintain search momentum.
        if not found_next_move and scored_candidates:
            best_cand = scored_candidates[0]
            curr_c1, curr_c2, curr_c3, curr_c4 = best_cand[0]
            tabu_list.append(best_cand[1])
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

        # Log the current global best gate count for statistical analysis
        fitness_history_matrix[experiment_id][current_iter - 1] = best_fitness
        
    return fitness_history_matrix, best_fitness, best_circuit