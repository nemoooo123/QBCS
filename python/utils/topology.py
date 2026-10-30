import numpy as np
import random
import copy
import time
from utils.init_state import hamming_distance


def _bits_to_int(bits):
    """
    Decode a 0/1 bit sequence (MSB-first) into its decimal value. Equivalent to
    `int(''.join(map(str, bits)), 2)` but skips building and parsing a string, since this
    runs on the order of millions of times across a full search (once per decision per
    neighbor per iteration).
    """
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def decode_and_synthesize(pop_l1, pop_l2, pop_l3, pop_l4, mapping_table, num_units, pop_size, trajectories):
    """
    Decodes hierarchical binary samples into decimal values and synthesizes
    quantum circuit routes for the entire population.

    Args:
        pop_l1-l4: Binary samples for 4 layers.
        mapping_table: Constraint table for valid cycle lengths (encode_2_bit_table).
        num_qubits: Number of qubits (n).
        pop_size: Total individuals in population (N).
        trajectories: Database of transition paths (trans).
    """
    circuit_solutions = []

    for i in range(pop_size):
        # Layer 1: Component priority decoding (Binary to Decimal)
        decoded_l1 = [_bits_to_int(bits) for bits in pop_l1[i]]

        # Layer 2: Entry point decoding with modulo constraint handling
        decoded_l2 = []
        for idx, bits in enumerate(pop_l2[i]):
            val = _bits_to_int(bits)
            limit = mapping_table[idx]
            # Ensure the starting index is within the valid range of the cycle
            decoded_l2.append(val % limit if val >= limit else val)

        # Layer 3: Intermediate node decoding (Handling special 999 bypass flag)
        decoded_l3 = []
        for cycle in pop_l3[i]:
            cycle_steps = []
            for step in cycle:
                if step[0] == 999:
                    cycle_steps.append([999])
                else:
                    # Convert each node bit-string to a decimal index
                    cycle_steps.append([_bits_to_int(node) for node in step])
            decoded_l3.append(cycle_steps)
        
        # Synthesis: Transform decoded parameters into the final circuit structure
        # Passing Layer 4 directly as it is handled within the synthesize_route logic
        
        individual_solution = synthesize_route(decoded_l1, decoded_l2, decoded_l3,
                                          pop_l4[i], num_units, trajectories)
        
        # Encapsulate synthesized circuit with its corresponding Gate Count and Path Continuity Fitness.
        # Data structure: (Circuit_Object, Integer: Gate_Count, Float: Fitness_Score)
        circuit_solutions.append((individual_solution,len(individual_solution)))
        
    return circuit_solutions

def synthesize_route(priority_weights, entry_points, mid_node_matrix, operation_sequences, num_units, trajectories):
    """
    Synthesizes the final execution route by resolving cycle sequences and inter-cycle dependencies.

    This function implements the physical instantiation phase of the synthesis framework,
    mapping optimized logical topologies into a sequence of executable state transitions.

    Args:
        priority_weights (list): Layer 1 - Heuristic processing priorities for cycle scheduling.
        entry_points (list): Layer 2 - Starting state indices for each cycle sequence.
        mid_node_matrix (list): Layer 3 - Weight matrix of candidate intermediate nodes for each transition.
        operation_sequences (list): Layer 4 - Sequence of logical operators (gates) associated with each step.
        num_units (int): System radix, representing the total number of processing elements (n-bits/qubits).
        trajectories (list): Pre-calculated database of valid state transition trajectories.

    Returns:
        circuit: The fully assembled reversible circuit description.
    """

    # Perform temporal scheduling by sorting cycles in descending order based on Layer 1 priority weights.
    # Higher priority cycles are processed first to ensure optimal path selection.
    sorted_order = [idx for idx, _ in sorted(enumerate(priority_weights), key=lambda x: x[1], reverse=True)]
    
    final_paths = []
    final_ops = []

    # Convert optimized node priorities into concrete state-space trajectories and operation sequences.
    for ind, cycle_idx in enumerate(sorted_order):
        # Retrieve the entry pointer for the current cycle sequence.
        step_ptr = entry_points[cycle_idx]
        total_steps = len(mid_node_matrix[cycle_idx])
        
        # Edge case handling for degenerate cycles (single-transition cycles) to maintain sequence integrity.
        if total_steps == 1: total_steps = 2
        
        # Iterate through all transition steps within the current cycle.
        for _ in range(total_steps - 1):
            # Manage circular buffer indexing for the cycle sequence.
            if step_ptr >= total_steps: step_ptr = 0
            
            
            # Generate the specific state trajectory based on optimized node weights and the trajectory database.
            route = generate_state_trajectory(step_ptr, mid_node_matrix[cycle_idx][step_ptr], trajectories[cycle_idx], num_units)    
            
            # Extract the corresponding logical operator (e.g., reversible gate) for the current transition.
            op_gate = operation_sequences[cycle_idx][step_ptr] 
            
            final_paths.append(route)
            final_ops.append(op_gate)
            step_ptr += 1
    # --- Optimization Metric: Adjacency Continuity Fitness ---
    # Defined as the ratio of reducible segment pairs to total adjacent transitions.
    # Formula: fitness_1 = 1 - (N_reducible / N_total_adjacent)

    # N_total_adjacent: Total number of potential coupling points between segments
    total_adjacency_count = len(final_paths) - 1
    redundant_coupling_count = 0

    for i in range(total_adjacency_count):
        # Segment pair extraction for adjacency analysis
        segment_current = final_paths[i]
        segment_next = final_paths[i+1]
        
        # CASE 1: Standard Sequential Adjacency (Intra-cycle or direct flow)
        # Check if the terminal node of segment[i] aligns with the initial node of segment[i+1]
        if segment_current[-1] == segment_next[0]:
            # Symmetric Redundancy Check: Verify if the preceding and succeeding nodes are identical
            # This implies a reversible transition that can be eliminated (Gate reduction)
            if segment_current[-2] == segment_next[1]:
                redundant_coupling_count += 1
                
        # CASE 2: Strategic Cross-Boundary Adjacency (Inter-cycle or non-linear flow)
        # Analyze four boundary alignment permutations for potential logic reduction
        else:
            # Tail-to-Head Symmetry Analysis (Offset 1)
            if segment_current[-1] == segment_next[1]:
                if segment_current[-2] == segment_next[0]:
                    redundant_coupling_count += 1
            
            # Head-to-Head Symmetry Analysis
            elif segment_current[0] == segment_next[1]:
                if segment_current[1] == segment_next[0]:
                    redundant_coupling_count += 1
            
            # Tail-to-Tail Symmetry Analysis
            elif segment_current[-1] == segment_next[-2]:
                if segment_current[-2] == segment_next[-1]:
                    redundant_coupling_count += 1
            
            # Head-to-Tail Symmetry Analysis
            elif segment_current[0] == segment_next[-2]:
                if segment_current[1] == segment_next[-1]:
                    redundant_coupling_count += 1

    # Final assembly: Map the extracted trajectories and operators into a comprehensive reversible circuit description.
    circuit = assemble_reversible_circuit(final_paths, final_ops, num_units)

    return circuit

def initialize_solution_layer(data_structure):
    """
    Resets the optimization layer to its neutral state before generating 
    a new custom solution. This clears previous search weights while 
    preserving the mandatory structural constraints.
    """
    for i in range(len(data_structure)):
        current_item = data_structure[i]
        
        if isinstance(current_item, list):
            # Check if we have reached the terminal path-selection level
            is_terminal_level = all(not isinstance(element, list) for element in current_item)
            
            if is_terminal_level:
                # 999 is a Reserved State, typically indicating a fixed or 
                # non-optimizable path. We preserve these.
                if len(current_item) == 1:
                    data_structure[i] = [999]
                else:
                    # Reset all variable path priorities to zero for the next iteration
                    data_structure[i] = [0] * len(current_item)
            else:
                # Recursive call to handle multi-dimensional solution structures
                initialize_solution_layer(current_item)
                
        else:
            # Direct reset of scalar weights, avoiding protected indices (999)
            if current_item != 999:
                data_structure[i] = 0

def analyze_bit_differences(step_idx, path_priorities, trajectory, is_flagged, num_units):
    """
    Identifies differing bit locations between two states in a trajectory.
    Returns the target bit indices, hamming distance, and state boundary nodes.
    """
    diff_locations = []
    entry_node = 0
    exit_node = 0
    
    # Check if the current step has a valid priority set (non-reserved)
    if path_priorities[0] != 999:
        # Define transition boundary nodes from the trajectory
        start_state = trajectory[step_idx]
        end_state = trajectory[step_idx + 1]
        
        # Calculate bitwise transition metrics
        h_distance = hamming_distance(start_state, end_state)
        bit_array_start = [int(b) for b in bin(start_state)[2:].zfill(num_units)]
        bit_array_end = [int(b) for b in bin(end_state)[2:].zfill(num_units)]
        
        if is_flagged:
            entry_node = start_state
            exit_node = end_state
            
        # Identify specific bit indices where transitions occur
        for i in range(len(bit_array_start)):
            if bit_array_start[i] != bit_array_end[i]:
                diff_locations.append(i)

    else:
        # RESERVED CASE: Direct transition without heuristic optimization
        start_state = trajectory[step_idx]
        end_state = trajectory[step_idx + 1]
        
        h_distance = hamming_distance(start_state, end_state)
        bit_array_start = [int(b) for b in bin(start_state)[2:].zfill(num_units)]
        bit_array_end = [int(b) for b in bin(end_state)[2:].zfill(num_units)]
        
        if is_flagged:
            entry_node = start_state
            exit_node = end_state
            
        for i in range(len(bit_array_start)):
            if bit_array_start[i] != bit_array_end[i]:
                diff_locations.append(i)

    return diff_locations, h_distance, entry_node, exit_node

def map_next_transition_bits(next_start_node, next_target_node, num_units):
    """
    Identifies differing bit indices for the first step of the next cycle.
    Used for look-ahead sequence optimization and connectivity analysis.
    """
    # Convert nodes to binary arrays representing system states
    bit_array_start = [int(b) for b in bin(next_start_node)[2:].zfill(num_units)]
    bit_array_target = [int(b) for b in bin(next_target_node)[2:].zfill(num_units)]

    transition_indices = []
    
    # Identify specific bit indices where state transitions occur
    for i in range(len(bit_array_start)):
        # Compare bits to find discrepancies (flips required)
        if bit_array_start[i] != bit_array_target[i]:
            transition_indices.append(i)
            
    return transition_indices

def reduce_sequences_standard(path_indices_list, weight_list):
    """
    Standard look-ahead reduction. Identifies and removes redundant transitions 
    by comparing current step bits with future steps.
    """
    num_steps = len(path_indices_list)
    reduction_counts = [0] * num_steps
    reduced_elements = [[] for _ in range(num_steps)]

    # Extract entry/exit metadata
    exit_node_indices = path_indices_list[-1]

    if num_steps == 1:
        return [0, 0], [[], []], exit_node_indices, []

    for i in range(num_steps - 1):
        current_step_indices = path_indices_list[i]
        matched_in_round = []
        available_weight = weight_list[i]
        
        # Adjust weight if existing reductions occupy slots
        if len(reduced_elements[i]) != 0:
            available_weight = weight_list[i] - 1 - len(reduced_elements[i])

        next_ptr = i + 1
        is_chaining = False # Tracks if we are continuing a multi-step reduction chain
        
        while next_ptr < num_steps and available_weight > 0:
            next_step_indices = path_indices_list[next_ptr]
            
            # Identify overlapping bit indices for reduction
            found_bits = [x for x in current_step_indices if x in next_step_indices and x not in matched_in_round]
            
            if found_bits:
                matched_in_round.extend(found_bits)
                temp_buffer = []
                for bit in found_bits:
                    path_indices_list[i].remove(bit)
                    path_indices_list[next_ptr].remove(bit)
                    temp_buffer.insert(0, bit)
                    reduced_elements[next_ptr].append(bit)
                    available_weight -= 1
                
                # Update current step's reduced record
                if is_chaining:
                    reduced_elements[i] = temp_buffer + reduced_elements[i]
                else:
                    reduced_elements[i].extend(temp_buffer)

            # Special Case: Preceding step synergy (Single bit flip optimization)
            if weight_list[i] == 1 and i != 0 and len(found_bits) == 1 and (reduction_counts[i-2]+1 < weight_list[i-1]):
                # Logic to check if previous step can also absorb this bit
                prev_found = [x for x in path_indices_list[i-1] if x in next_step_indices and x not in matched_in_round]
                if prev_found:
                    matched_in_round.extend(prev_found)
                    tmp_fd=[]
                    for bit in prev_found:
                        path_indices_list[i-1].remove(bit)
                        path_indices_list[next_ptr].remove(bit)
                        reduced_elements[next_ptr].append(bit)
                        
                        tmp_fd.insert(0,bit)
                        reduced_elements[i-1].extend(tmp_fd)

            # Determine if the look-ahead can proceed further
            if len(found_bits) == 1 and (next_ptr + 1) < num_steps and weight_list[next_ptr] == 1:
                next_ptr += 1
                is_chaining = True
            else:
                break
        
        reduction_counts[i] = len(matched_in_round)
        
    return reduction_counts, reduced_elements, exit_node_indices, reduced_elements[-1]

def reduce_sequences_ordered(path_indices_list, weight_list):
    """
    Performs ordered sequence reduction by tracking index offsets (ka_count).
    This ensures bit-flip consistency when integrating new reductions into existing layers.
    """
    num_steps = len(path_indices_list)
    reduction_counts = [0] * num_steps
    reduced_elements = [[] for _ in range(num_steps)]
    
    # Metadata for state tracking
    exit_boundary_indices = path_indices_list[-1]
    
    if num_steps == 1:
        reduction_counts.append(0)
        reduced_elements.append([])
        return reduction_counts, reduced_elements, exit_boundary_indices, reduced_elements[-1]

    for i in range(num_steps - 1):
        current_step_indices = path_indices_list[i]
        matched_in_round = []
        available_weight = weight_list[i]
        
        # Calculate remaining weight if the current step already contains reduced bits
        if len(reduced_elements[i]) != 0:
            available_weight = weight_list[i] - 1 - len(reduced_elements[i])

        next_ptr = i + 1
        is_chaining = False 
        # Crucial: Track the current list size to maintain correct insertion offset
        insertion_offset = len(reduced_elements[i]) 
        
        while next_ptr < num_steps and available_weight > 0:
            next_step_indices = path_indices_list[next_ptr]
            found_bits = [x for x in current_step_indices if x in next_step_indices and x not in matched_in_round]
            
            if found_bits:
                matched_in_round.extend(found_bits)
                temp_buffer = [] # This is your tmp_fd
                
                for bit in found_bits:
                    path_indices_list[i].remove(bit)
                    path_indices_list[next_ptr].remove(bit)
                    temp_buffer.insert(0, bit) # Reverse order for logic consistency
                    reduced_elements[next_ptr].append(bit)
                    available_weight -= 1
                
                # Apply the specific insertion logic (ton vs. append)
                for bit in temp_buffer:
                    if is_chaining:
                        reduced_elements[i].insert(insertion_offset, bit)
                    else:
                        reduced_elements[i].append(bit)

            # Special Case: Preceding step synergy (Look-back logic)
            # Restored: (counts[i-2] + 1 < hamm[i-1]) condition equivalent
            if weight_list[i] == 1 and i != 0 and len(found_bits) == 1:
                if (reduction_counts[i-2] + 1 < weight_list[i-1]):
                    prev_found = [x for x in path_indices_list[i-1] if x in next_step_indices and x not in matched_in_round]
                    if prev_found:
                        matched_in_round.extend(prev_found)
                        temp_buffer = []
                        for bit in prev_found:
                            path_indices_list[i-1].remove(bit)
                            path_indices_list[next_ptr].remove(bit)
                            temp_buffer.insert(0, bit)
                            reduced_elements[next_ptr].append(bit)
                            available_weight -= 1
                        for bit in temp_buffer:
                            reduced_elements[i-1].append(bit)
                
            # Look-ahead chaining condition
            if len(found_bits) == 1 and (next_ptr + 1) < num_steps and weight_list[next_ptr] == 1:
                next_ptr += 1
                is_chaining = True
            else:
                break
        
        reduction_counts[i] = len(matched_in_round)
        
    return reduction_counts, reduced_elements, exit_boundary_indices, reduced_elements[-1]

def reduce_sequences_targeted(path_indices_list, weight_list, common_value):
    """
    Targeted reduction for Leap Strategy alignment. Forces specific common_value 
    matching at the initial step (i=0) to ensure cross-cycle connectivity.
    """
    num_steps = len(path_indices_list)
    reduction_counts = [0] * num_steps
    reduced_elements = [[] for _ in range(num_steps)]
    exit_boundary_indices = path_indices_list[-1]
    
    if num_steps == 1:
        reduction_counts.append(0)
        reduced_elements.append([])
        return reduction_counts, reduced_elements, exit_boundary_indices, reduced_elements[-1]

    for i in range(num_steps - 1):
        current_step_indices = path_indices_list[i]
        matched_in_round = []
        available_weight = weight_list[i]
        
        if len(reduced_elements[i]) != 0:
            available_weight = weight_list[i] - 1 - len(reduced_elements[i])

        next_ptr = i + 1
        is_chaining = False
        insertion_offset = len(reduced_elements[i])
        
        while next_ptr < num_steps and available_weight > 0:
            next_step_indices = path_indices_list[next_ptr]
            
            # UNIQUE KB LOGIC: Target alignment at the start of cycle
            forced_bits = []
            if i == 0:
                if common_value in next_step_indices and common_value not in matched_in_round:
                    forced_bits = [common_value]
                    matched_in_round.extend(forced_bits)
            
            found_bits = [x for x in current_step_indices if x in next_step_indices and x not in matched_in_round]
            
            # Combine forced target bits with naturally found bits
            if found_bits:
                matched_in_round.extend(found_bits)
            
            if i == 0:
                found_bits = forced_bits + found_bits
            
            
            temp_buffer = []
            for bit in found_bits:
                path_indices_list[i].remove(bit)
                path_indices_list[next_ptr].remove(bit)
                temp_buffer.insert(0, bit)
                reduced_elements[next_ptr].append(bit)
                available_weight -= 1
            
            for bit in temp_buffer:
                if is_chaining:
                    reduced_elements[i].insert(insertion_offset, bit)
                else:
                    reduced_elements[i].append(bit)

            # Special Case: Look-back synergy
            if weight_list[i] == 1 and i != 0 and len(found_bits) == 1:
                if (reduction_counts[i-2] + 1 < weight_list[i-1]):
                    prev_found = [x for x in path_indices_list[i-1] if x in next_step_indices and x not in matched_in_round]
                    if prev_found:
                        matched_in_round.extend(prev_found)
                        temp_buffer = []
                        for bit in prev_found:
                            path_indices_list[i-1].remove(bit)
                            path_indices_list[next_ptr].remove(bit)
                            temp_buffer.insert(0, bit)
                            reduced_elements[next_ptr].append(bit)
                            available_weight -= 1
                        for bit in temp_buffer:
                            reduced_elements[i-1].append(bit)
                
            if len(found_bits) == 1 and (next_ptr + 1) < num_steps and weight_list[next_ptr] == 1:
                next_ptr += 1
                is_chaining = True
            else:
                break
        
        reduction_counts[i] = len(matched_in_round)
        
    return reduction_counts, reduced_elements, exit_boundary_indices, reduced_elements[-1]

def determine_transition_strategy(next_entry_node, next_second_node, 
                                 curr_entry_node, curr_exit_node, 
                                 entry_overlap_nodes, exit_overlap_nodes, 
                                 next_transition_indices):
    """
    Evaluates state connectivity between cycles to select an optimization strategy.
    Strategies 1-3 manage standard reductions, while Strategy 4 handles forced leap alignments.
    """
    
    # 'is_reduction_eligible' (formerly obe): Validation flag to ensure 
    # sequence length constraints are met for Strategies 1 & 3.
    is_reduction_eligible = False
    is_leap_enabled = 0       # Formerly ka_four (Strategy 4 flag)
    selected_strategy = 0     # Formerly inp_cycle_total (Strategy 1, 2, or 3)
    
    # Step 1: Connectivity Constraint Validation
    if len(exit_overlap_nodes) != 0:
        # Strategies 1 & 3 require multiple entry options to safely reduce
        if len(entry_overlap_nodes) > 1:
            is_reduction_eligible = True
    else:
        # If no exit overlap exists, constraints are bypassed
        is_reduction_eligible = True

    # Step 2: Distance Matrix Calculation
    # Measuring bit-flips between the boundaries of current and next cycles
    dist_next_start_curr_exit = hamming_distance(next_entry_node, curr_exit_node)
    dist_next_start_curr_start = hamming_distance(next_entry_node, curr_entry_node)
    dist_next_second_curr_exit = hamming_distance(next_second_node, curr_exit_node)
    dist_next_second_curr_start = hamming_distance(next_second_node, curr_entry_node)

    candidate_nodes = []

    # Step 3: Strategy 3 Assessment (Prioritized)
    # Check for overlapping nodes in the next transition mapping
    if len(entry_overlap_nodes) > 0:
        for node in entry_overlap_nodes:
            if node in next_transition_indices:
                candidate_nodes.append(node) 

    if (dist_next_second_curr_exit == 1 and 
        len(candidate_nodes) != 0 and 
        is_reduction_eligible):
            selected_strategy = 3 

    # Step 4: Strategy 2 Assessment
    if selected_strategy == 0 and len(exit_overlap_nodes) > 0:
        candidate_nodes = [] # Reset for Strategy 2/4 specific alignment
        
        # Verify if the primary exit node is compatible with the next transition
        if exit_overlap_nodes[0] in next_transition_indices:
            candidate_nodes.append(exit_overlap_nodes[0])
            if dist_next_start_curr_start == 1 and len(candidate_nodes) != 0:
                selected_strategy = 2

    # Step 5: Strategy 1 Assessment
    if selected_strategy == 0:
        candidate_nodes = [] # Re-scan entry overlaps
        if len(entry_overlap_nodes) > 0:
            for node in entry_overlap_nodes:
                if node in next_transition_indices:
                    candidate_nodes.append(node) 
                    
        if (dist_next_start_curr_exit == 1 and 
            len(candidate_nodes) != 0 and 
            is_reduction_eligible):
                selected_strategy = 1

    # Step 6: Strategy 4 (Leap Strategy) Assessment
    # Independent check for forced alignment when entry/exit must match
    if (dist_next_second_curr_start == 1 and 
        len(candidate_nodes) != 0 and 
        len(exit_overlap_nodes) > 0):
            is_leap_enabled = 1

    return candidate_nodes, selected_strategy, is_leap_enabled

def generate_state_trajectory(step_idx, path_priorities, trajectory, num_units):
    """
    Constructs the intermediate state sequence between two nodes.
    Flips bits based on the optimized priority sequence (path_priorities).
    """
    state_route = []
    
    # Check for non-reserved priority set
    if path_priorities[0] != 999:
        # Step 1: Sort indices based on priority values to determine flip order
        # priority_map (formerly pp) dictates which bit flips first
        sorted_elements = sorted(enumerate(path_priorities), key=lambda x: x[1], reverse=True)
        priority_map = [idx for idx, val in sorted_elements]
        
        start_node = trajectory[step_idx]
        end_node = trajectory[step_idx + 1]
        state_route.append(start_node)

        # Step 2/3: Identify the bit locations that require flipping (Hamming indices).
        # XOR marks every differing bit at once, so this needs no bin()/zfill() bit-array
        # construction (list index 0 = MSB, matching the old zfill-string convention).
        diff = start_node ^ end_node
        diff_locations = [i for i in range(num_units) if (diff >> (num_units - 1 - i)) & 1]

        # Step 4: Execute bit-flips following the optimized priority map.
        # XOR-toggling the running integer state avoids rebuilding a bit-array/string on
        # every flip (this loop runs for every flip step of every transition).
        current_state = start_node
        for flip_idx in priority_map:
            target_bit_pos = diff_locations[flip_idx]
            current_state ^= (1 << (num_units - 1 - target_bit_pos))
            state_route.append(current_state)

    else:
        # BYPASS CASE: Direct transition between start and end node
        start_node = trajectory[step_idx]
        end_node = trajectory[step_idx + 1]
        state_route.extend([start_node, end_node])

    return state_route

class _MRAccumulator:
    """
    Linear-time Deletion Rule (A*A=I) + Moving Rule (MR), replacing the old approach of
    alternating a full deletion pass and a full "bubble through commuting neighbors" pass
    to a fixed point (worst case O(passes * n^2)).

    Every gate is a hypercube edge with exactly two vertices (control pattern with its
    target bit forced to 0, and to 1). For each incoming gate, this looks up - via a stack
    of live-gate positions kept per vertex - the most recently emitted still-live gate
    that touched either of its two vertices. Every gate strictly between that lookup and
    now touched neither vertex (otherwise it, not this one, would be the "most recent"),
    which means every one of them commutes with the incoming gate (edges that share no
    vertex commute). So that lookup result is exactly the nearest gate the incoming one
    cannot freely slide past: if it's the identical gate, the two cancel (A*A=I);
    otherwise the incoming gate is simply recorded as newly live. This reaches the same
    reduced circuit as the old fixed-point loop (trace-rewriting with an independence
    relation has a confluent, order-independent normal form), but does it in a single
    amortized O(1) step per gate. Ported from an independently-implemented, verified-fast
    reference (學弟's `_MRAccumulator` in stepwise_encoding.py) into this codebase's
    gate representation (length-num_bits 0/1/3 lists, plus a target-bit-shift and a
    control-pattern integer per gate, as already used elsewhere in this module).
    """

    __slots__ = ("gates", "targets", "ctrls", "vertex_stacks", "live")

    def __init__(self):
        self.gates = []          # cancelled slots become None, positions never reused
        self.targets = []
        self.ctrls = []          # control-pattern int, target bit already cleared to 0
        self.vertex_stacks = {}  # vertex(int) -> stack of positions of live gates touching it
        self.live = 0

    def push(self, gate, target_shift, ctrl, full_mask):
        base = ctrl & full_mask & ~(1 << target_shift)
        v0 = base
        v1 = base | (1 << target_shift)

        stack0 = self.vertex_stacks.get(v0)
        if stack0 is None:
            stack0 = self.vertex_stacks[v0] = []
        stack1 = self.vertex_stacks.get(v1)
        if stack1 is None:
            stack1 = self.vertex_stacks[v1] = []

        p0 = stack0[-1] if stack0 else -1
        p1 = stack1[-1] if stack1 else -1
        p = p0 if p0 > p1 else p1

        # p shares a vertex with the incoming gate by construction. Only one edge per
        # target dimension passes through a given vertex, so sharing a vertex AND a
        # target bit forces it to be the exact same edge - no need to also compare ctrl.
        if p >= 0 and self.targets[p] == target_shift:
            self.gates[p] = None
            self.targets[p] = None
            self.ctrls[p] = None
            stack0.pop()
            stack1.pop()
            self.live -= 1
        else:
            self.gates.append(gate)
            self.targets.append(target_shift)
            self.ctrls.append(base)
            k = len(self.gates) - 1
            stack0.append(k)
            stack1.append(k)
            self.live += 1

    def result(self):
        gates, targets, ctrls = [], [], []
        for g, t, c in zip(self.gates, self.targets, self.ctrls):
            if g is not None:
                gates.append(g)
                targets.append(t)
                ctrls.append(c)
        return gates, targets, ctrls


def assemble_reversible_circuit(state_trajectories, transition_sequence_matrix, num_bits):
    """
    Synthesizes a reversible logic circuit from state trajectories and gate transition sequences.
    Implements real-time gate cancellation (Identity Law: A * A = I) to minimize circuit depth,
    followed by the Moving Rule (MR): gates are bubbled through commuting neighbors (hypercube
    edges that don't touch) so cancellation can reach pairs a plain adjacent-only pass misses.

    Args:
        state_trajectories (list): List of decimal state sequences for each transition.
        transition_sequence_matrix (list): Matrix defining the order of bit-flips (0: head-start, 1: tail-start).
        num_bits (int): Total number of bits in the system.

    Returns:
        tuple: (optimized_gate_list, total_raw_transitions)
    """
    raw_step_transitions = []
    
    # Step 1: Decompose state trajectories into discrete bit-flip transitions
    for i in range(len(state_trajectories)):
        head_ptr = 0
        tail_ptr = len(state_trajectories[i]) - 1
        
        total_steps = len(transition_sequence_matrix[i])
        mid_point = int(total_steps / 2)
        
        # Assemble the first half of the transition path
        # 0: Represents a flip starting from the current state (head)
        # 1: Represents a flip starting from the target state (tail)
        for j in range(mid_point):
            if transition_sequence_matrix[i][j] == 0:
                raw_step_transitions.append([state_trajectories[i][head_ptr], state_trajectories[i][head_ptr + 1]])
                head_ptr += 1
            elif transition_sequence_matrix[i][j] == 1:
                raw_step_transitions.append([state_trajectories[i][tail_ptr - 1], state_trajectories[i][tail_ptr]])
                tail_ptr -= 1
        
        # Central Pivot: The primary transition linking the head and tail paths
        if tail_ptr - head_ptr == 1 or mid_point == 0:
            raw_step_transitions.append([state_trajectories[i][head_ptr], state_trajectories[i][tail_ptr]])
            # Flip pointers to assemble the returning path correctly
            head_ptr, tail_ptr = tail_ptr, head_ptr 
        else:
            raise ValueError("Synchronization Error: Path pointers failed to meet at central pivot.")

        # Assemble the second half (return path)
        effective_limit = total_steps if mid_point != 0 else 0
        for j in range(mid_point, effective_limit):
            if transition_sequence_matrix[i][j] == 0:
                raw_step_transitions.append([state_trajectories[i][head_ptr], state_trajectories[i][head_ptr + 1]])
                head_ptr += 1
            elif transition_sequence_matrix[i][j] == 1:
                raw_step_transitions.append([state_trajectories[i][tail_ptr - 1], state_trajectories[i][tail_ptr]])
                tail_ptr -= 1

    # Step 2: Map Bit Transitions to Reversible Gates and Apply Optimization
    # Gate Encoding: 0/1 = Control Bits, 3 = Target Bit (Flip)
    # Step 3/4: Cancellation (A*A=I) + Moving Rule, done in one O(1)-amortized step per
    # gate via the vertex-indexed accumulator instead of a separate peephole pass plus
    # repeated O(n) fixed-point rescans.
    full_mask = (1 << num_bits) - 1
    acc = _MRAccumulator()

    for transition in raw_step_transitions:
        state_start, state_end = transition[0], transition[1]

        # Convert decimal states to binary bit arrays
        bits_start = [int(b) for b in bin(state_start)[2:].zfill(num_bits)]
        bits_end = [int(b) for b in bin(state_end)[2:].zfill(num_bits)]

        current_gate = []
        target_shift = -1
        for bit_idx in range(num_bits):
            # Identical bits denote a Control condition; differing bits denote the Target
            if bits_start[bit_idx] == bits_end[bit_idx]:
                current_gate.append(bits_start[bit_idx])
            else:
                current_gate.append(3)
                # zfill()/bin() strings are MSB-first, so list position bit_idx corresponds
                # to integer bit weight 2**(num_bits-1-bit_idx), not 2**bit_idx.
                target_shift = num_bits - 1 - bit_idx

        # state_start's binary digits already agree with this gate's 0/1 controls
        # everywhere except the target bit, which _MRAccumulator masks out anyway.
        acc.push(current_gate, target_shift, state_start, full_mask)

    optimized_gate_list, gate_targets, gate_ctrls = acc.result()

    return optimized_gate_list

def verify_circuit_logic(optimized_gates, num_bits, target_truth_table):
    """
    Validates the functional correctness of the synthesized reversible circuit.
    Simulates the circuit's operation across all possible input states (2^n).
    
    Args:
        optimized_gates (list): The list of synthesized reversible gates (Toffoli-like).
        num_bits (int): Total number of bits in the system.
        target_truth_table (list): The reference output sequence to compare against.
        
    Returns:
        int: 1 if the circuit logic matches the target truth table, 0 otherwise.
    """
    simulated_outputs = []
    
    # Generate all possible input states from 0 to 2^n - 1
    input_space = list(range(2**num_bits))
    
    for input_state in input_space:
        # Step 1: Initialize the current state bit array
        current_bits = [int(b) for b in bin(input_state)[2:].zfill(num_bits)]
        
        # Step 2: Pass the input through each logic gate in the sequence
        for gate in optimized_gates:
            target_pos = -1
            is_control_satisfied = True

            # Check Control Bit conditions (0, 1) and identify the Target Bit (3)
            for bit_idx in range(num_bits):
                gate_value = gate[bit_idx]
                
                # If the gate bit is a control (0 or 1), it must match the current state
                if (gate_value == 0 or gate_value == 1):
                    if gate_value != current_bits[bit_idx]:
                        is_control_satisfied = False
                        break
                # Identification of the target bit to be flipped (NOT operation)
                elif gate_value == 3:
                    target_pos = bit_idx
            
            # Step 3: Execute Bit-Flip (NOT) if all control conditions are met
            if target_pos != -1 and is_control_satisfied:
                current_bits[target_pos] = 1 - current_bits[target_pos]
        
        # Step 4: Convert final bit array back to decimal state
        binary_result = ''.join(map(str, current_bits))
        decimal_output = int(binary_result, 2)
        simulated_outputs.append(decimal_output)
    
    # Step 5: Final Integrity Comparison
    # Check if the entire simulated output set matches the target truth table
    if simulated_outputs == target_truth_table:
        return 1
    else:
        return 0
    
  