"""
FDD Analysis Example for Students
=================================

This example shows how to use the FDD utility for operational modal analysis.
Students can adapt this code for their own structural health monitoring projects.

Course: Sensing Techniques and Data Analytics
Instructor: Mohammad Talebi-Kalaleh, University of Alberta
"""

import numpy as np
import json
from fdd_utility import FDD_Analysis

def example_bridge_analysis():
    """
    Example: Analyzing bridge vibration data using FDD
    """
    print("Bridge FDD Analysis Example")
    print("=" * 40)
    
    # Step 1: Load your data
    # Replace this section with your own data loading
    with open('bridge_vibration_data.json', 'r') as f:
        data = json.load(f)
    
    # Extract ambient test data (intact scenario, test 1)
    scenario_data = data['scenarios']['intact']
    ambient_test = scenario_data['ambient_tests']['test_1']
    sampling_frequency = scenario_data['metadata']['sampling_frequency_Hz']
    
    # Prepare acceleration data matrix (samples x sensors)
    sensor_names = [f'sensor_{i}' for i in range(1, 25)]  # 24 sensors
    acceleration_data = []
    
    for sensor_name in sensor_names:
        sensor_data = np.array(ambient_test[sensor_name])
        acceleration_data.append(sensor_data)
    
    # Convert to proper format: (time_samples, n_sensors)
    acceleration_matrix = np.array(acceleration_data).T
    
    print(f"Data loaded: {acceleration_matrix.shape}")
    print(f"Sampling frequency: {sampling_frequency} Hz")
    print(f"Duration: {acceleration_matrix.shape[0]/sampling_frequency:.1f} seconds")
    
    # Step 2: Create FDD analyzer
    fdd = FDD_Analysis(sampling_frequency)
    
    # Step 3: Perform FDD analysis
    results = fdd.analyze(
        acceleration_data=acceleration_matrix,
        n_modes=4,                    # Number of modes to identify
        freq_range=(0.5, 10),        # Frequency search range (Hz)
        peak_prominence=None         # Auto-detect peak prominence
    )
    
    # Step 4: Display results
    fdd.print_results(results)
    
    # Step 5: Generate plots
    # Define sensor positions (bridge example: 24 sensors at 5m intervals)
    sensor_positions = np.array([2.5 + i*5 for i in range(24)])
    
    # Plot singular values with identified peaks
    fdd.plot_singular_values(
        results, 
        save_filename='bridge_fdd_singular_values.png',
        show_plot=True
    )
    
    # Plot identified mode shapes
    fdd.plot_mode_shapes(
        results, 
        sensor_positions,
        save_filename='bridge_fdd_mode_shapes.png',
        show_plot=True,
        bridge_spans=[20, 60]  # Mark span boundaries for 3-span bridge
    )
    
    # Step 6: Access results for further processing
    modal_frequencies = results['modal_frequencies']
    mode_shapes = results['mode_shapes']
    
    print("\\nExtracted Results:")
    print(f"Modal frequencies: {modal_frequencies}")
    print(f"Mode shapes matrix: {mode_shapes.shape}")
    
    return results

def compare_with_known_results():
    """
    Example: Comparing FDD results with known modal properties
    """
    print("\\nFDD Validation Example")
    print("=" * 40)
    
    # Load bridge data and known modal properties
    with open('bridge_vibration_data.json', 'r') as f:
        data = json.load(f)
    
    # Load true modal properties
    with open('mode_shapes/intact_mode_data.json', 'r') as f:
        true_mode_data = json.load(f)
    
    true_frequencies = true_mode_data['natural_frequencies_Hz'][:4]
    print(f"True frequencies: {[f'{f:.3f}' for f in true_frequencies]} Hz")
    
    # Perform FDD analysis (as in previous example)
    scenario_data = data['scenarios']['intact']
    ambient_test = scenario_data['ambient_tests']['test_1']
    sampling_frequency = scenario_data['metadata']['sampling_frequency_Hz']
    
    sensor_names = [f'sensor_{i}' for i in range(1, 25)]
    acceleration_data = []
    for sensor_name in sensor_names:
        acceleration_data.append(np.array(ambient_test[sensor_name]))
    acceleration_matrix = np.array(acceleration_data).T
    
    fdd = FDD_Analysis(sampling_frequency)
    results = fdd.analyze(acceleration_matrix, n_modes=4, freq_range=(0.5, 10))
    
    # Compare results
    identified_frequencies = results['modal_frequencies']
    
    print("\\nComparison Results:")
    print("-" * 30)
    for i in range(min(len(true_frequencies), len(identified_frequencies))):
        error = abs(identified_frequencies[i] - true_frequencies[i]) / true_frequencies[i] * 100
        print(f"Mode {i+1}:")
        print(f"  True: {true_frequencies[i]:.3f} Hz")
        print(f"  FDD:  {identified_frequencies[i]:.3f} Hz")
        print(f"  Error: {error:.2f}%")
    
    # Calculate MAC if mode shapes are available
    if 'mode_shapes' in results and len(results['mode_shapes']) > 0:
        true_mode_shapes = []
        for i in range(4):
            mode_key = f'mode_{i+1}'
            if mode_key in true_mode_data['mode_shapes']:
                true_mode_shapes.append(true_mode_data['mode_shapes'][mode_key]['sensor_amplitudes'])
        
        if len(true_mode_shapes) > 0:
            true_modes_matrix = np.array(true_mode_shapes).T
            identified_modes = results['mode_shapes']
            
            print("\\nModal Assurance Criterion (MAC):")
            print("-" * 30)
            for i in range(min(identified_modes.shape[1], true_modes_matrix.shape[1])):
                # Align signs if necessary
                if np.dot(identified_modes[:, i], true_modes_matrix[:, i]) < 0:
                    identified_modes[:, i] *= -1
                
                mac = fdd.calculate_mac(true_modes_matrix[:, i], identified_modes[:, i])
                print(f"Mode {i+1} MAC: {mac:.4f}")
    
    return results

if __name__ == "__main__":
    print("FDD Analysis Examples for Students")
    print("=" * 50)
    
    # Run examples
    try:
        # Example 1: Bridge analysis
        results1 = example_bridge_analysis()
        
        # Example 2: Validation
        results2 = compare_with_known_results()
        
        print("\\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the generated PNG files for plots.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure the required data files are available.")