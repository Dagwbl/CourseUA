"""
FDD Utility for Structural Health Monitoring
============================================

Simple and clean Frequency Domain Decomposition (FDD) tool for students to perform 
operational modal analysis on ambient vibration data.

Course: Sensing Techniques and Data Analytics
Instructor: Mohammad Talebi-Kalaleh, University of Alberta

Usage Example:
--------------
```python
import numpy as np
from fdd_utility import FDD_Analysis

# Load your acceleration data (samples x sensors)
acceleration_data = np.array(...)  # Shape: (time_samples, n_sensors)
sampling_frequency = 200  # Hz

# Create FDD analyzer
fdd = FDD_Analysis(sampling_frequency)

# Perform analysis
results = fdd.analyze(acceleration_data, n_modes=4)

# Display results
fdd.print_results(results)

# Plot singular values
fdd.plot_singular_values(results)

# Plot mode shapes (if sensor positions provided)
sensor_positions = np.array([2.5, 7.5, 12.5, ...])  # positions in meters
fdd.plot_mode_shapes(results, sensor_positions)
```
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import csd as cpsd
from scipy.linalg import svd
from scipy.signal import find_peaks
import warnings

class FDD_Analysis:
    """
    Frequency Domain Decomposition (FDD) for Operational Modal Analysis
    
    A clean and simple implementation for students to identify modal parameters
    from ambient vibration data.
    """
    
    def __init__(self, sampling_frequency):
        """
        Initialize FDD analyzer
        
        Parameters:
        -----------
        sampling_frequency : float
            Sampling frequency in Hz
        """
        self.fs = sampling_frequency
        
    def compute_psd_matrix(self, acceleration_data):
        """
        Compute Cross-Power Spectral Density (CPSD) matrix
        
        Parameters:
        -----------
        acceleration_data : numpy.ndarray
            Acceleration data matrix (samples x sensors)
            
        Returns:
        --------
        frequencies : numpy.ndarray
            Frequency vector
        psd_matrix : numpy.ndarray
            PSD matrix (frequency x sensors x sensors)
        """
        n_samples, n_sensors = acceleration_data.shape
        
        # Choose appropriate segment length for good frequency resolution
        nperseg = min(4096, n_samples//8)
        noverlap = nperseg // 2
        
        print(f"Computing PSD matrix for {n_sensors} sensors...")
        print(f"Segment length: {nperseg}, Overlap: {noverlap}")
        
        # Initialize PSD matrix
        frequencies = None
        psd_matrix = None
        
        for i in range(n_sensors):
            for j in range(n_sensors):
                # Compute cross-power spectral density
                freqs, psd_ij = cpsd(acceleration_data[:, i], acceleration_data[:, j],
                                   nperseg=nperseg, noverlap=noverlap, 
                                   nfft=nperseg, fs=self.fs, window='hann')
                
                # Initialize arrays on first iteration
                if frequencies is None:
                    frequencies = freqs
                    n_freq = len(frequencies)
                    psd_matrix = np.zeros((n_freq, n_sensors, n_sensors), dtype=complex)
                
                psd_matrix[:, i, j] = psd_ij
        
        return frequencies, psd_matrix
    
    def perform_svd_analysis(self, psd_matrix):
        """
        Perform Singular Value Decomposition on PSD matrix
        
        Parameters:
        -----------
        psd_matrix : numpy.ndarray
            PSD matrix (frequency x sensors x sensors)
            
        Returns:
        --------
        singular_values : numpy.ndarray
            First singular values at each frequency
        mode_shapes : numpy.ndarray
            Mode shape vectors (sensors x frequency)
        """
        n_freq, n_sensors, _ = psd_matrix.shape
        
        print("Performing SVD analysis...")
        
        singular_values = np.zeros(n_freq)
        mode_shapes = np.zeros((n_sensors, n_freq), dtype=complex)
        
        for k in range(n_freq):
            # SVD of PSD matrix at frequency k
            u, s, vh = svd(psd_matrix[k, :, :])
            
            # Store first singular value and corresponding mode shape
            singular_values[k] = s[0]
            mode_shapes[:, k] = u[:, 0]
        
        return singular_values, mode_shapes
    
    def identify_modal_peaks(self, frequencies, singular_values, n_modes=4, 
                           freq_range=(0.5, 10), peak_prominence=None):
        """
        Identify modal frequencies from singular value peaks
        
        Parameters:
        -----------
        frequencies : numpy.ndarray
            Frequency vector
        singular_values : numpy.ndarray
            First singular values
        n_modes : int
            Number of modes to identify
        freq_range : tuple
            Frequency range to search (min_freq, max_freq) in Hz
        peak_prominence : float, optional
            Minimum prominence for peak detection
            
        Returns:
        --------
        modal_frequencies : list
            Identified modal frequencies
        peak_indices : list
            Indices of identified peaks in frequency vector
        """
        print(f"Identifying {n_modes} modal peaks in range {freq_range} Hz...")
        
        # Convert to dB scale for better peak detection
        sv_db = 10 * np.log10(singular_values + 1e-12)
        
        # Limit search to specified frequency range
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        freq_limited = frequencies[freq_mask]
        sv_limited = sv_db[freq_mask]
        
        # Automatic prominence if not specified
        if peak_prominence is None:
            peak_prominence = (np.max(sv_limited) - np.min(sv_limited)) * 0.1
        
        # Find peaks
        peaks_idx, properties = find_peaks(sv_limited, prominence=peak_prominence, 
                                         distance=len(sv_limited)//20)
        
        if len(peaks_idx) == 0:
            print("Warning: No peaks found. Try adjusting peak_prominence or freq_range.")
            return [], []
        
        # Sort peaks by prominence (highest first)
        prominences = properties['prominences']
        sorted_indices = np.argsort(prominences)[::-1]
        
        # Select top n_modes peaks
        selected_peaks = peaks_idx[sorted_indices[:min(n_modes, len(peaks_idx))]]
        
        # Convert back to original frequency indices
        original_indices = np.where(freq_mask)[0][selected_peaks]
        modal_frequencies = frequencies[original_indices]
        
        # Sort by frequency
        sort_order = np.argsort(modal_frequencies)
        modal_frequencies = modal_frequencies[sort_order]
        peak_indices = original_indices[sort_order]
        
        print(f"Identified frequencies: {[f'{f:.3f}' for f in modal_frequencies]} Hz")
        
        return modal_frequencies.tolist(), peak_indices.tolist()
    
    def extract_mode_shapes(self, mode_shapes_complex, peak_indices):
        """
        Extract and process mode shapes at identified frequencies
        
        Parameters:
        -----------
        mode_shapes_complex : numpy.ndarray
            Complex mode shapes (sensors x frequency)
        peak_indices : list
            Indices of identified modal peaks
            
        Returns:
        --------
        mode_shapes_real : numpy.ndarray
            Real-valued mode shapes (sensors x modes)
        """
        n_sensors = mode_shapes_complex.shape[0]
        n_modes = len(peak_indices)
        
        mode_shapes_real = np.zeros((n_sensors, n_modes))
        
        for i, peak_idx in enumerate(peak_indices):
            # Get complex mode shape at peak frequency
            complex_mode = mode_shapes_complex[:, peak_idx]
            
            # Convert to real by phase alignment
            # Use the sensor with maximum amplitude as reference
            max_idx = np.argmax(np.abs(complex_mode))
            reference_phase = np.angle(complex_mode[max_idx])
            
            # Align all components to reference phase
            aligned_mode = complex_mode * np.exp(-1j * reference_phase)
            real_mode = np.real(aligned_mode)
            
            # Normalize (unit maximum amplitude)
            max_val = np.max(np.abs(real_mode))
            if max_val > 0:
                real_mode = real_mode / max_val
            
            mode_shapes_real[:, i] = real_mode
        
        return mode_shapes_real
    
    def analyze(self, acceleration_data, n_modes=4, freq_range=(0.5, 10), 
                peak_prominence=None):
        """
        Complete FDD analysis pipeline
        
        Parameters:
        -----------
        acceleration_data : numpy.ndarray
            Acceleration data matrix (samples x sensors)
        n_modes : int
            Number of modes to identify
        freq_range : tuple
            Frequency range to search (min_freq, max_freq) in Hz
        peak_prominence : float, optional
            Minimum prominence for peak detection
            
        Returns:
        --------
        results : dict
            Dictionary containing all analysis results:
            - 'frequencies': Frequency vector
            - 'singular_values': First singular values
            - 'modal_frequencies': Identified modal frequencies
            - 'mode_shapes': Identified mode shapes
            - 'peak_indices': Indices of modal peaks
            - 'psd_matrix': Cross-power spectral density matrix
        """
        print("="*50)
        print("FREQUENCY DOMAIN DECOMPOSITION (FDD) ANALYSIS")
        print("="*50)
        
        # Remove mean from each sensor (remove gravity/DC component)
        acc_data = acceleration_data - np.mean(acceleration_data, axis=0)
        
        print(f"Data shape: {acc_data.shape}")
        print(f"Duration: {acc_data.shape[0]/self.fs:.1f} seconds")
        print(f"Sampling frequency: {self.fs} Hz")
        
        # Step 1: Compute PSD matrix
        frequencies, psd_matrix = self.compute_psd_matrix(acc_data)
        
        # Step 2: Perform SVD analysis
        singular_values, mode_shapes_complex = self.perform_svd_analysis(psd_matrix)
        
        # Step 3: Identify modal peaks
        modal_frequencies, peak_indices = self.identify_modal_peaks(
            frequencies, singular_values, n_modes, freq_range, peak_prominence)
        
        # Step 4: Extract mode shapes
        if len(peak_indices) > 0:
            mode_shapes = self.extract_mode_shapes(mode_shapes_complex, peak_indices)
        else:
            mode_shapes = np.array([])
        
        # Compile results
        results = {
            'frequencies': frequencies,
            'singular_values': singular_values,
            'modal_frequencies': modal_frequencies,
            'mode_shapes': mode_shapes,
            'peak_indices': peak_indices,
            'psd_matrix': psd_matrix,
            'sampling_frequency': self.fs,
            'n_sensors': acc_data.shape[1],
            'duration': acc_data.shape[0] / self.fs
        }
        
        print("="*50)
        print("FDD ANALYSIS COMPLETE")
        print("="*50)
        
        return results
    
    def print_results(self, results):
        """
        Print formatted analysis results
        
        Parameters:
        -----------
        results : dict
            Results dictionary from analyze() method
        """
        print("\\nFDD Analysis Results:")
        print("-" * 30)
        print(f"Number of sensors: {results['n_sensors']}")
        print(f"Data duration: {results['duration']:.1f} seconds")
        print(f"Frequency resolution: {results['frequencies'][1]:.4f} Hz")
        print(f"Number of modes identified: {len(results['modal_frequencies'])}")
        
        if len(results['modal_frequencies']) > 0:
            print("\\nIdentified Modal Frequencies:")
            for i, freq in enumerate(results['modal_frequencies']):
                print(f"  Mode {i+1}: {freq:.3f} Hz")
    
    def plot_singular_values(self, results, save_filename=None, show_plot=True):
        """
        Plot singular values with identified peaks
        
        Parameters:
        -----------
        results : dict
            Results dictionary from analyze() method
        save_filename : str, optional
            Filename to save plot
        show_plot : bool
            Whether to display the plot
        """
        frequencies = results['frequencies']
        singular_values = results['singular_values']
        modal_frequencies = results['modal_frequencies']
        peak_indices = results['peak_indices']
        
        # Convert to dB
        sv_db = 10 * np.log10(singular_values + 1e-12)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot singular values
        ax.plot(frequencies, sv_db, 'k-', linewidth=2, label='1st Singular Value')
        
        # Mark identified peaks
        for i, (freq, peak_idx) in enumerate(zip(modal_frequencies, peak_indices)):
            ax.plot(freq, sv_db[peak_idx], 'ro', markersize=8, markerfacecolor='red')
            ax.axvline(x=freq, color='red', linestyle='--', alpha=0.7)
            ax.annotate(f'f₍{i+1}₎ = {freq:.3f} Hz',
                       xy=(freq, sv_db[peak_idx]),
                       xytext=(10, 20), textcoords='offset points',
                       arrowprops=dict(arrowstyle="->", color='red'),
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
        ax.set_ylabel('1st Singular Values (dB)', fontsize=14, fontweight='bold')
        ax.set_title('FDD Analysis - Singular Value Decomposition', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.2)
        
        # Set reasonable limits
        if len(modal_frequencies) > 0:
            ax.set_xlim([0, max(10, modal_frequencies[-1] * 2)])
        else:
            ax.set_xlim([0, 10])
        
        plt.tight_layout()
        
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"Singular value plot saved: {save_filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_mode_shapes(self, results, sensor_positions, save_filename=None, 
                        show_plot=True, bridge_spans=None):
        """
        Plot identified mode shapes
        
        Parameters:
        -----------
        results : dict
            Results dictionary from analyze() method
        sensor_positions : numpy.ndarray
            Sensor positions along the structure (in meters)
        save_filename : str, optional
            Filename to save plot
        show_plot : bool
            Whether to display the plot
        bridge_spans : list, optional
            Span boundaries for vertical lines (e.g., [20, 60] for 3-span bridge)
        """
        modal_frequencies = results['modal_frequencies']
        mode_shapes = results['mode_shapes']
        
        if len(modal_frequencies) == 0:
            print("No modes to plot")
            return
        
        n_modes = len(modal_frequencies)
        
        # Create subplots
        fig, axes = plt.subplots(n_modes, 1, figsize=(14, 4*n_modes))
        if n_modes == 1:
            axes = [axes]
        
        for i in range(n_modes):
            mode_shape = mode_shapes[:, i]
            freq = modal_frequencies[i]
            
            # Plot mode shape
            axes[i].plot(sensor_positions, mode_shape, 'b-', linewidth=3, 
                        marker='o', markersize=6, label=f'Mode {i+1}')
            axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Add span boundaries if provided
            if bridge_spans:
                for span_boundary in bridge_spans:
                    axes[i].axvline(x=span_boundary, color='gray', 
                                  linestyle=':', alpha=0.7, linewidth=2)
            
            # Formatting
            axes[i].set_xlabel('Position along Structure (m)', fontsize=12)
            axes[i].set_ylabel('Normalized Amplitude', fontsize=12)
            axes[i].set_title(f'Mode {i+1}: f = {freq:.3f} Hz', 
                            fontsize=14, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=11)
            
            # Set limits
            axes[i].set_xlim([sensor_positions[0], sensor_positions[-1]])
            y_max = max(abs(mode_shape.min()), abs(mode_shape.max()))
            axes[i].set_ylim([-y_max*1.1, y_max*1.1])
        
        plt.tight_layout()
        
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"Mode shapes plot saved: {save_filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def calculate_mac(self, mode1, mode2):
        """
        Calculate Modal Assurance Criterion between two mode shapes
        
        Parameters:
        -----------
        mode1, mode2 : numpy.ndarray
            Mode shape vectors
            
        Returns:
        --------
        mac : float
            MAC value (0 to 1)
        """
        if len(mode1) != len(mode2):
            return 0.0
        
        numerator = (np.dot(mode1, mode2))**2
        denominator = np.dot(mode1, mode1) * np.dot(mode2, mode2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


# Example usage and demo function
def demo_fdd_analysis():
    """
    Demonstration of FDD utility using the bridge dataset
    """
    import json
    import os
    
    print("FDD Utility Demonstration")
    print("=" * 50)
    
    # Try to load bridge dataset
    try:
        with open('bridge_vibration_data.json', 'r') as f:
            data = json.load(f)
        
        # Extract intact scenario ambient test data
        scenario_data = data['scenarios']['intact']
        ambient_test = scenario_data['ambient_tests']['test_1']
        fs = scenario_data['metadata']['sampling_frequency_Hz']
        
        # Prepare acceleration data
        sensor_names = [f'sensor_{i}' for i in range(1, 25)]
        acc_data = []
        
        for sensor_name in sensor_names:
            if sensor_name in ambient_test:
                sensor_data = np.array(ambient_test[sensor_name])
                acc_data.append(sensor_data)
        
        # Convert to numpy array (samples x sensors)
        acceleration_matrix = np.array(acc_data).T
        
        # Sensor positions (bridge sensors at 5m intervals starting at 2.5m)
        sensor_positions = np.array([2.5 + i*5 for i in range(24)])
        
        print(f"Loaded data: {acceleration_matrix.shape}")
        print(f"Sampling frequency: {fs} Hz")
        
        # Create FDD analyzer
        fdd = FDD_Analysis(fs)
        
        # Perform analysis
        results = fdd.analyze(acceleration_matrix, n_modes=4, freq_range=(0.5, 10))
        
        # Display results
        fdd.print_results(results)
        
        # Create plots
        fdd.plot_singular_values(results, save_filename='demo_singular_values.png')
        fdd.plot_mode_shapes(results, sensor_positions, 
                           save_filename='demo_mode_shapes.png',
                           bridge_spans=[20, 60])  # 3-span bridge
        
        print("\\nDemo completed successfully!")
        print("Generated files: demo_singular_values.png, demo_mode_shapes.png")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure bridge_vibration_data.json is available")


if __name__ == "__main__":
    demo_fdd_analysis()