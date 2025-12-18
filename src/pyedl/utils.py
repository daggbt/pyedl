"""
Utility functions for the electrochemical capacitance package.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from .models import StericModel, ElectrochemicalSystem


def polarizability_angstrom_to_si(alpha_angstrom):
    """
    Convert polarizability from cubic angstroms (Å³) to SI units (C·m²/V).

    Parameters
    ----------
    alpha_angstrom : float
        Polarizability in cubic angstroms (Å³).

    Returns
    -------
    float
        Polarizability in SI units (C·m²/V).
    """
    # Conversion factor: 1 Å³ = 1.11265 × 10^-40 C·m²/V
    conversion_factor = 1.11265e-40

    # Convert to SI units
    alpha_si = alpha_angstrom * conversion_factor

    return alpha_si


def plot_capacitance_vs_potential(system, expt_cap=None,
                                 potential_range=(-1, 1), num_points=101, 
                                 steric_model='cs', save_path=None, show_plot=True):
    """
    Plots analytical capacitance over potential for an electrochemical system.
    
    Parameters:
    -----------
    system : ElectrochemicalSystem
        The electrochemical system to analyze
    expt_cap : pandas.DataFrame or None
        Experimental capacitance data with columns 'pot' and 'cap'
    potential_range : tuple
        Range of potentials to plot (min, max) in V
    num_points : int
        Number of points to calculate
    steric_model : str
        Steric model type ('cs' for Carnahan-Starling or 'liu' for Liu model)
    save_path : str or None
        Path to save the plot (if None, plot is not saved)
    show_plot : bool
        Whether to display the plot
    
    Returns:
    --------
    tuple: (potentials, capacitances)
        The calculated potential and capacitance values
    """
    # Create model
    model = StericModel(system, steric_model=steric_model)
    
    # Define potentials
    potentials = np.linspace(potential_range[0], potential_range[1], num_points)
    
    # Calculate capacitance for each potential
    capacitances = []
    
    for pot in potentials:
        try:
            # Use analytical capacitance
            capacitance = model.analytical_capacitance(pot)
            capacitances.append(capacitance)
        except Exception as e:
            print(f"Error at potential {pot}V: {str(e)}")
            capacitances.append(np.nan)  # Use NaN for error points
    
    # Convert to numpy array
    capacitances = np.array(capacitances)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(potentials, capacitances, 'r-', linewidth=3, label='Model')
    
    # Plot experimental data if provided
    if expt_cap is not None:
        plt.plot(expt_cap["pot"], expt_cap["cap"], '--k', label='Experimental')
    
    # Add labels and title
    plt.xlabel('Potential (V)', fontsize=12)
    plt.ylabel('Capacitance (μF/cm²)', fontsize=12)
    plt.title(f'Analytical Capacitance vs Potential\nConcentration: {system.concentration} M', fontsize=14)
    
    # Add details in text box
    ion_radii = system.get_ion_radii()
    ion_polarizabilities = system.get_ion_polarizabilities()
    textstr = (f'{system.cation.name}: radius = {ion_radii[0]} Å, α = {ion_polarizabilities[0]:.3f} Å³\n'
               f'{system.anion.name}: radius = {ion_radii[1]} Å, α = {ion_polarizabilities[1]:.3f} Å³')
    plt.annotate(textstr, xy=(0.02, 0.8), xycoords='axes fraction', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7))
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return potentials, capacitances


def save_capacitance_data(model, potentials, filename=None):
    """
    Calculate and save capacitance data to file.
    
    Parameters:
    -----------
    model : StericModel
        The model to use for calculations
    potentials : array-like
        Array of potentials to calculate capacitance for
    filename : str or None
        Output filename (if None, generates automatically)
    
    Returns:
    --------
    numpy.ndarray: Array of results
    """
    # Clear caches to ensure recalculation 
    model.reduced_dielectric_cache = {}
    model.surface_volume_fraction_cache = {}
    model.steric_thickness_cache = {}

    # Calculate capacitance for each potential
    results = []
    for pot in potentials:
        try:
            # Calculate capacitance and charge density
            capacitance = model.analytical_capacitance(pot)
            charge = model.charge_density(pot)
            volfrac_surface = model.get_steric_parameter_phi(pot, [])
            q, v, _ = model.get_ion_parameters(pot)
            c_s = volfrac_surface / v / 1000 / sc.N_A
            volfrac_bulk = model.volfrac_b
            
            # Get reduced dielectric constant
            reduced_dielectric = model.get_reduced_dielectric_from_potential(pot)
            
            results.append([pot, capacitance, charge, reduced_dielectric, 
                          volfrac_surface, volfrac_bulk, c_s])
            
        except Exception as e:
            print(f"Error processing potential {pot}: {str(e)}")
            # Add placeholder values on error
            results.append([pot, 0.0, 0.0, model.epsilon_r, 0.0, 0.0, 0.0])
    
    # Convert results to numpy array
    capacitance_array = np.array(results)
    
    # Generate filename if not provided
    if filename is None:
        filename = f'capacitance_radii{model.ion_radii[0]},{model.ion_radii[1]}_pol{model.ion_polarizabilities[0]},{model.ion_polarizabilities[1]}.dat'
    
    # Save results to file
    header = "potential(V)\tcapacitance(uF/cm^2)\tchargeDensity(C/m2)\treducedDielectric\tvolFraction_s\tvolfrac_b\tsurfaceConc"
    
    try:
        np.savetxt(filename, capacitance_array, fmt="%.10g", 
                  delimiter="\t", newline='\n', header=header)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return capacitance_array


def find_ion_permittivity_from_capacitance(model, potential, exp_capacitance, 
                                          epsilon_min=0.1, epsilon_max=10.0, 
                                          tolerance=0.001, max_iterations=100):
    """
    Find the ion permittivity that produces a capacitance matching
    the experimental value at a given potential.
    
    Parameters:
    -----------
    model : StericModel
        The electrochemical model instance
    potential : float
        Electric potential in V
    exp_capacitance : float
        Experimental capacitance value in μF/cm²
    epsilon_min : float
        Minimum value for ion permittivity search
    epsilon_max : float
        Maximum value for ion permittivity search
    tolerance : float
        Acceptable error in capacitance matching (μF/cm²)
    max_iterations : int
        Maximum number of iterations for the search
    
    Returns:
    --------
    tuple: (ion_permittivity, calculated_capacitance, error)
        The found ion permittivity, resulting capacitance, and error
    """
    # Save original values
    counterion_index = 0 if potential >= 0 else 1
    original_permittivity = model.ion_permitivities[counterion_index]
    
    # Clear caches to ensure recalculation with new values
    model.surface_volume_fraction_cache = {}
    model.reduced_dielectric_cache = {}
    model.steric_thickness_cache = {}
    
    # Define a function to calculate capacitance for a given ion permittivity
    def calculate_capacitance_with_epsilon(epsilon_i):
        try:
            # Set the ion permittivity for the appropriate ion
            model.ion_permitivities[counterion_index] = epsilon_i
            
            # Clear caches to ensure recalculation
            model.reduced_dielectric_cache = {}
            
            # Calculate capacitance
            capacitance = model.analytical_capacitance(potential)
            return capacitance
        except Exception as e:
            print(f"Error with ion permittivity {epsilon_i}: {str(e)}")
            return None
    
    # Use a grid search to find the ion permittivity
    epsilon_values = np.linspace(epsilon_min, epsilon_max, 20)
    
    best_epsilon = original_permittivity
    best_capacitance = None
    best_error = float('inf')
    
    # First pass - wide grid search
    print(f"Searching for ion permittivity at potential {potential}V with target capacitance {exp_capacitance} μF/cm²")
    for epsilon_i in epsilon_values:
        capacitance = calculate_capacitance_with_epsilon(epsilon_i)
        
        if capacitance is not None:
            error = abs(capacitance - exp_capacitance)
            
            # Update best result if this is better
            if error < best_error:
                best_error = error
                best_epsilon = epsilon_i
                best_capacitance = capacitance
                
                # If we're within tolerance, we can stop
                if error < tolerance:
                    break
    
    # Second pass - refined search around best value
    if best_error > tolerance:
        print(f"Refining search around ε_i = {best_epsilon:.2f} (current error: {best_error:.2f} μF/cm²)")
        
        # Define a narrower range around the best result
        margin = max(2.0, best_epsilon * 0.2)  # 20% margin or at least 2.0
        refined_min = max(1.0, best_epsilon - margin)
        refined_max = best_epsilon + margin
        
        refined_values = np.linspace(refined_min, refined_max, 10)
        
        for epsilon_i in refined_values:
            capacitance = calculate_capacitance_with_epsilon(epsilon_i)
            
            if capacitance is not None:
                error = abs(capacitance - exp_capacitance)
                
                # Update best result if this is better
                if error < best_error:
                    best_error = error
                    best_epsilon = epsilon_i
                    best_capacitance = capacitance
    
    # Restore original value
    model.ion_permitivities[counterion_index] = original_permittivity
    
    # Clear caches again
    model.surface_volume_fraction_cache = {}
    model.reduced_dielectric_cache = {}
    model.steric_thickness_cache = {}
    
    print(f"Found ion permittivity: {best_epsilon:.2f}")
    print(f"Resulting capacitance: {best_capacitance:.2f} μF/cm² (target: {exp_capacitance:.2f} μF/cm²)")
    print(f"Error: {best_error:.2f} μF/cm²")
    
    return best_epsilon, best_capacitance, best_error
