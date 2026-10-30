import numpy as np
import time
import pandas as pd
import os
import copy
from utils.data_loader import DataLoader
from utils.init_state import find_cycles, build_encode
from core.AE_QTS import AE_QTS_run_single_experiment
from core.DE import DE_run_single_experiment
from core.PSO import PSO_run_single_experiment
from core.TS import TS_run_single_experiment
from core.QTS import QTS_run_single_experiment
from core.GA import GA_run_single_experiment
from core.ABC import ABC_run_single_experiment
from core.WOA import WOA_run_single_experiment
from core.QEA import QEA_run_single_experiment
def main():
    """
    Main execution entry point for Reversible Circuit Synthesis experiments.
    Handles data loading, algorithm selection, and statistical aggregation.
    """
    # Initialize data loader and show available problem sets
    loader = DataLoader()
    loader.get_info()
    # User Input Section
    try:
        num_bits = int(input("Enter number of bits (n): "))
        problem_idx = int(input("Enter problem index (0 = load a custom sparse truth table file): "))
        algo_choice = int(input("Select Algorithm (1: AE-QTS, 2: QTS, 3: QEA, 4: GA, 5: DE, 6: TS, 7: PSO, 8: WOA, 9: ABC): "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Load the target truth table (output)
    if problem_idx == 0:
        # Sparse format: only the positions that differ from identity need to be listed
        # (see DataLoader.parse_sparse_table) - avoids hand-writing/pasting a full 2^n table.
        sparse_path = input("Enter path to sparse truth table file: ").strip()
        try:
            target_output = DataLoader.read_sparse_table_file(sparse_path)
        except (OSError, ValueError) as e:
            print(f"Error: {e}")
            return
        if len(target_output) != 2 ** num_bits:
            print(f"Error: sparse file declares a table of size {len(target_output)}, "
                  f"which doesn't match the {num_bits} bits you entered.")
            return
    else:
        target_output = loader.get_output(num_bits, problem_idx)
        if not target_output:
            print("Error: Targeted problem not found. Please restart.")
            return
    
    print(f"\nTarget Truth Table: {target_output}")

    # Experimental Configuration
    num_experiments = 30
    max_iterations = 10000
    num_neighbors = 50  # Population size (N)

    print(f"Starting {num_experiments} experiments, each with {max_iterations} iterations...")

    # Pre-allocate matrix for convergence analysis (Rows: Experiments, Cols: Generations)
    # Using float('inf') as default to easily track minimization progress
    fitness_history_matrix = np.full((num_experiments, max_iterations), float('inf'))
    best_scores_per_experiment = []
    best_circuit_per_experiment = []
    execution_times = []

    # Step 1: Analyze permutation cycles in the truth table (deterministic given
    # target_output, so this only needs to run once, not once per experiment).
    # check_zero_gate: Boolean to verify if the circuit requires any gates at all
    cycles, is_gate_required = find_cycles(target_output, check_zero_gate=True)

    if is_gate_required:
        # Step 2: Build the encoding framework once. encoding_table/trajectory_base/cycles
        # are only ever read during a run (never mutated), so they're safe to share across
        # experiments as-is. qindividuals1-4 ARE mutated in place by updateQ() during a run
        # (for AE-QTS/QTS/QEA), so each experiment needs its own deep copy of the pristine
        # template below - a shallow .copy() would leave the inner numpy arrays shared,
        # letting probability drift leak between experiments and breaking the i.i.d.
        # assumption behind averaging over num_experiments trials.
        (qindividuals1_template, qindividuals2_template, qindividuals3_template,
         qindividuals4_template, encoding_table, trajectory_base) = build_encode(cycles)

    for r in range(num_experiments):
        # Track execution time for the current experiment trial
        experiment_start_time = time.time()

        if not is_gate_required:
            print(f"Experiment {r+1}: Identity mapping detected (Zero gates required).")
            best_scores_per_experiment.append(0)
            fitness_history_matrix[r, :] = 0
            continue

        if algo_choice == 1:  # AE-QTS (Amplitude-Ensemble Quantum-Inspired Tabu Search)
            # Fresh deep copy per experiment: updateQ() mutates these in place, so reusing
            # the template without copying (or shallow-copying it) would carry the previous
            # experiment's learned probabilities into this one.
            qindividuals1 = copy.deepcopy(qindividuals1_template)
            qindividuals2 = copy.deepcopy(qindividuals2_template)
            qindividuals3 = copy.deepcopy(qindividuals3_template)
            qindividuals4 = copy.deepcopy(qindividuals4_template)

            # Step 3: Run the core search algorithm for a single experiment
            fitness_history_matrix, final_best_gate, best_circuit_this_run = AE_QTS_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                qindividuals1 = qindividuals1,
                qindividuals2 = qindividuals2,
                qindividuals3 = qindividuals3,
                qindividuals4 = qindividuals4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output,
                delta_theta = 0.005
            )

        elif algo_choice == 2: # QTS (Quantum-Inspired Tabu Search)
            # Fresh deep copy per experiment (updateQ() mutates these in place - see note above)
            qindividuals1 = copy.deepcopy(qindividuals1_template)
            qindividuals2 = copy.deepcopy(qindividuals2_template)
            qindividuals3 = copy.deepcopy(qindividuals3_template)
            qindividuals4 = copy.deepcopy(qindividuals4_template)

            # Step 3: Run the core search algorithm for a single experiment
            fitness_history_matrix, final_best_gate, best_circuit_this_run = QTS_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                qindividuals1 = qindividuals1,
                qindividuals2 = qindividuals2,
                qindividuals3 = qindividuals3,
                qindividuals4 = qindividuals4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output,
                delta_theta = 0.01
            )

        elif algo_choice == 3: # QEA (Quantum Evolutionary Algorithm - Global Best Guided)
            # Fresh deep copy per experiment (updateQ() mutates these in place - see note above)
            qindividuals1 = copy.deepcopy(qindividuals1_template)
            qindividuals2 = copy.deepcopy(qindividuals2_template)
            qindividuals3 = copy.deepcopy(qindividuals3_template)
            qindividuals4 = copy.deepcopy(qindividuals4_template)

            # Step 2: Execute QEA Core Evolution
            # This algorithm uses a 'Global Best' oriented rotation strategy to shift
            # quantum probability amplitudes toward the historical optimal circuit configuration.
            fitness_history_matrix, final_best_gate, best_circuit_this_run = QEA_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                qindividuals1 = qindividuals1,
                qindividuals2 = qindividuals2,
                qindividuals3 = qindividuals3,
                qindividuals4 = qindividuals4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output,
                delta_theta = 0.0005  # Rotation step size for QEA state updates
            )

        elif algo_choice == 4: # GA (Genetic Algorithm)

            # GA never mutates its seed population matrices in place (they're only sampled
            # once via gen_nbrs to build the initial generation), so the shared template can
            # be reused directly across experiments without copying.
            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4 = (
                qindividuals1_template, qindividuals2_template, qindividuals3_template, qindividuals4_template
            )

            fitness_history_matrix, final_best_gate, best_circuit_this_run = GA_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                pop_matrix1 = pop_matrix1,
                pop_matrix2 = pop_matrix2,
                pop_matrix3 = pop_matrix3,
                pop_matrix4 = pop_matrix4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output,
                k=5,
                pc=0.9, # crossover
                pm=0.001 # mutation

            )
        elif algo_choice == 5: # DE (Differential Evolution - A population-based stochastic global optimizer)
            # Never mutated in place - safe to reuse the shared template (see GA note above).
            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4 = (
                qindividuals1_template, qindividuals2_template, qindividuals3_template, qindividuals4_template
            )

            fitness_history_matrix, final_best_gate , best_circuit_this_run = DE_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                pop_matrix1 = pop_matrix1,
                pop_matrix2 = pop_matrix2,
                pop_matrix3 = pop_matrix3,
                pop_matrix4 = pop_matrix4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output,
                CR = 0.3,
            )
        
        elif algo_choice == 6: # TS (Tubu Search)
            # Never mutated in place - safe to reuse the shared template (see GA note above).
            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4 = (
                qindividuals1_template, qindividuals2_template, qindividuals3_template, qindividuals4_template
            )

            fitness_history_matrix, final_best_gate, best_circuit_this_run = TS_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                pop_matrix1 = pop_matrix1,
                pop_matrix2 = pop_matrix2,
                pop_matrix3 = pop_matrix3,
                pop_matrix4 = pop_matrix4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output,
                tabu_size = 150
            )

        elif algo_choice == 7: # PSO (Particle Swarm Optimization)
            # Never mutated in place - safe to reuse the shared template (see GA note above).
            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4 = (
                qindividuals1_template, qindividuals2_template, qindividuals3_template, qindividuals4_template
            )

            fitness_history_matrix, final_best_gate, best_circuit_this_run = PSO_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                pop_matrix1 = pop_matrix1,
                pop_matrix2 = pop_matrix2,
                pop_matrix3 = pop_matrix3,
                pop_matrix4 = pop_matrix4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output,
                w=0.6,c1=2.5,c2=1.0
            )

        elif algo_choice == 8: # WOA (Whale Optimization Algorithm)
            # Never mutated in place - safe to reuse the shared template (see GA note above).
            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4 = (
                qindividuals1_template, qindividuals2_template, qindividuals3_template, qindividuals4_template
            )

            fitness_history_matrix, final_best_gate, best_circuit_this_run = WOA_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                pop_matrix1 = pop_matrix1,
                pop_matrix2 = pop_matrix2,
                pop_matrix3 = pop_matrix3,
                pop_matrix4 = pop_matrix4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output,
                b = 0.1  # Spiral Constant
            )

        elif algo_choice == 9: # ABC(Artificial Bee Colony Algorithm)

            # Never mutated in place - safe to reuse the shared template (see GA note above).
            pop_matrix1, pop_matrix2, pop_matrix3, pop_matrix4 = (
                qindividuals1_template, qindividuals2_template, qindividuals3_template, qindividuals4_template
            )

            fitness_history_matrix, final_best_gate, best_circuit_this_run = ABC_run_single_experiment(
                max_iterations = max_iterations,
                rotation_cycles = cycles,
                num_neighbors = num_neighbors,
                num_bits = num_bits,
                base_trajectory = trajectory_base,
                experiment_id = r,
                encoding_table = encoding_table,
                pop_matrix1 = pop_matrix1,
                pop_matrix2 = pop_matrix2,
                pop_matrix3 = pop_matrix3,
                pop_matrix4 = pop_matrix4,
                fitness_history_matrix = fitness_history_matrix,
                target_output = target_output,
                limit = 40
            )

        

        experiment_end_time = time.time()
        elapsed_time = experiment_end_time - experiment_start_time
        execution_times.append(elapsed_time)
            
        best_circuit_per_experiment.append(best_circuit_this_run)
        best_scores_per_experiment.append(final_best_gate)

        # Progress reporting
        print(f"--- Experiment {r+1} Finished | Best Gate: {final_best_gate} | Time: {elapsed_time:.2f}s ---")
    # Statistical Analysis and Result Aggregation
    # Calculate the average gate count across all experiments for each generation
    average_convergence_curve = np.mean(fitness_history_matrix, axis=0)
    std_convergence_curve = np.std(fitness_history_matrix, axis=0)
    
    global_min_gate = min(best_scores_per_experiment)
    best_exp_index = best_scores_per_experiment.index(global_min_gate) # Identify the most successful trial
    absolute_best_circuit = best_circuit_per_experiment[best_exp_index]
    total_end_time = time.time()
    
    final_avg = average_convergence_curve[-1]
    final_std = std_convergence_curve[-1]

# --- Automatic Directory and Filename Creation ---
    # Algorithm mapping: mapping choice ID to descriptive names
    algo_names = {
        1: "AE-QTS", 2: "QTS", 3: "QEA", 4: "GA", 5: "DE", 
        6: "TS", 7: "PSO", 8: "WOA", 9: "ABC"
    }
    algo_name = algo_names.get(algo_choice, "Other")
    
    # Establish directory path structure (e.g., exp/13_bit/QEA_Results/)
    save_dir = os.path.join("exp", f"{num_bits}_bit", f"{algo_name}_Results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define file path foundations
    base_filename = f"{algo_name}_{num_bits}_{problem_idx}"
    txt_path = os.path.join(save_dir, f"{base_filename}.txt")
    xlsx_path = os.path.join(save_dir, f"{base_filename}.xlsx")

    # --- Generate TXT Summary Report ---
    # This file provides a quick overview of the best results and system performance
    with open(txt_path, "w", encoding="utf-16") as f:
        f.write("========================================\n")
        f.write("      FINAL EXPERIMENTAL STATISTICS      \n")
        f.write("========================================\n")
        f.write(f"Best Gate Counts per Trial: {best_scores_per_experiment}\n")
        f.write(f"Global Minimum Gate Count:  {global_min_gate}\n")
        f.write(f"Final Result (Gen {max_iterations}): {final_avg:.2f} ± {final_std:.2f}\n")
        f.write(f"Average Time per Experiment: {sum(execution_times)/len(execution_times):.2f}s\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Circuit Structure:\n{absolute_best_circuit}\n")
        f.write("========================================\n")

    # --- Generate Detailed Excel Data (Pandas) ---
    # This file stores the full convergence matrix for statistical plotting and verification
    
    try:
        # 1. Construct the basic Fitness DataFrame (Rows: Trials, Columns: Generations)
        df = pd.DataFrame(fitness_history_matrix)
        df.index = [f"Trial_{i+1}" for i in range(df.shape[0])]
        df.columns = [f"Gen_{i+1}" for i in range(df.shape[1])]
        
        # 2. Append the Execution Time column
        # Ensure 'execution_times' is a list of floats/ints with the same length as num_experiments
        df['Execution_Time(s)'] = execution_times
        
        # 3. Calculate statistical rows (Mean and Std Deviation)
        # CRITICAL: We only calculate stats for 'Gen_x' columns, excluding 'Execution_Time(s)'
        fitness_cols = [c for c in df.columns if c.startswith('Gen_')]
        
        # Create a temporary view for calculation
        stats_view = df[fitness_cols]
        
        # Append Average and Std Dev rows
        df.loc['Average_Convergence'] = stats_view.mean()
        df.loc['Std_Deviation'] = stats_view.std()
        
        # 4. Manually fill the Execution Time average for the summary row
        # Calculate the mean execution time across all successful trials
        trial_execution_times = df['Execution_Time(s)'].iloc[:-2] 
        df.at['Average_Convergence', 'Execution_Time(s)'] = trial_execution_times.mean()
        
        # 5. EXPORT TO FILE (Make sure this line is present!)
        df.to_excel(xlsx_path)
        
        # Console Feedback
        print(f"\n[System] Summary report saved to: {txt_path}")
        print(f"[System] Full convergence history with execution times saved to: {xlsx_path}")

    except Exception as e:
        print(f"\n[Error] Failed to generate Excel file: {e}")
        # If Excel fails, try saving as CSV as a backup
        csv_path = xlsx_path.replace(".xlsx", ".csv")
        df.to_csv(csv_path)
        print(f"[Backup] Data saved to CSV instead: {csv_path}")

    


if __name__ == "__main__":
    main()
    