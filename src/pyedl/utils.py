"""
Utility functions for the electrochemical capacitance package.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from .fitting import fit_counterion_permittivity_curve
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


def _calculate_capacitance_sweep(model, potentials, use_jit_sweep=False):
    """Calculate capacitance over a potential sweep using the requested solver path."""
    potentials_array = np.asarray(potentials, dtype=float)

    if use_jit_sweep:
        return np.asarray(model.analytical_capacitance_sweep_jit(potentials_array), dtype=float)

    capacitances = []
    for pot in potentials_array:
        try:
            capacitance = model.analytical_capacitance(float(pot))
            capacitances.append(capacitance)
        except Exception as e:
            print(f"Error at potential {pot}V: {str(e)}")
            capacitances.append(np.nan)

    return np.array(capacitances, dtype=float)


def plot_capacitance_vs_potential(system, expt_cap=None,
                                 potential_range=(-1, 1), num_points=101,
                                 steric_model='cs', save_path=None, show_plot=True,
                                 use_jit_sweep=False):
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
    use_jit_sweep : bool
        Whether to evaluate the model sweep with the explicit JIT path
    
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
    capacitances = _calculate_capacitance_sweep(
        model,
        potentials,
        use_jit_sweep=use_jit_sweep,
    )
    
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


def save_capacitance_data(model, potentials, filename=None, use_jit_sweep=False):
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
    use_jit_sweep : bool
        Whether to evaluate the ordered sweep with the explicit JIT path
    
    Returns:
    --------
    numpy.ndarray: Array of results
    """
    # Clear caches to ensure recalculation
    model.invalidate_caches()

    potentials_array = np.asarray(potentials, dtype=float)
    capacitances_jit = None
    volfrac_surfaces_jit = None
    if use_jit_sweep:
        volfrac_surfaces_jit = np.asarray(
            model.get_steric_parameter_phi_sweep_jit(potentials_array),
            dtype=float,
        )
        capacitances_jit = np.asarray(
            model.analytical_capacitance_sweep_jit(
                potentials_array,
                phi_roots=volfrac_surfaces_jit,
            ),
            dtype=float,
        )

    # Calculate capacitance for each potential
    results = []
    for idx, pot in enumerate(potentials_array):
        potential = float(pot)
        try:
            if capacitances_jit is None:
                capacitance = model.analytical_capacitance(potential)
                volfrac_surface = model.get_steric_parameter_phi(potential, [])
            else:
                capacitance = float(capacitances_jit[idx])
                volfrac_surface = float(volfrac_surfaces_jit[idx])

            # Calculate charge density and dependent quantities
            charge = model.charge_density(potential)
            q, v, _ = model.get_ion_parameters(potential)
            c_s = volfrac_surface / v / 1000 / sc.N_A
            volfrac_bulk = model.volfrac_b

            # Get reduced dielectric constant
            reduced_dielectric = model.get_reduced_dielectric_from_potential(potential)

            results.append([potential, capacitance, charge, reduced_dielectric,
                          volfrac_surface, volfrac_bulk, c_s])

        except Exception as e:
            print(f"Error processing potential {potential}: {str(e)}")
            # Add placeholder values on error
            results.append([potential, 0.0, 0.0, model.epsilon_r, 0.0, 0.0, 0.0])
    
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
                                          tolerance=0.001, max_iterations=100,
                                          use_jit_sweep=False):
    """
    Find the ion permittivity that produces a capacitance matching
    the experimental value at a given potential.

    This compatibility wrapper now delegates to the optimizer-based
    curve-fitting API using a single-point capacitance target.
    
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
        Acceptable error in capacitance matching (μF/cm²). Used only to
        report whether the fitted result meets the requested target.
    max_iterations : int
        Maximum number of optimizer iterations
    use_jit_sweep : bool
        Whether to evaluate the single-point fit through the JIT sweep path
    
    Returns:
    --------
    tuple: (ion_permittivity, calculated_capacitance, error)
        The found ion permittivity, resulting capacitance, and error
    """
    warnings.warn(
        "find_ion_permittivity_from_capacitance is a compatibility wrapper; "
        "prefer fit_counterion_permittivity_curve for new code.",
        DeprecationWarning,
        stacklevel=2,
    )

    print(
        f"Fitting ion permittivity at potential {potential}V with target capacitance "
        f"{exp_capacitance} μF/cm²"
    )

    fit_result = fit_counterion_permittivity_curve(
        model,
        [potential],
        [exp_capacitance],
        epsilon_bounds=(epsilon_min, epsilon_max),
        use_jit_sweep=use_jit_sweep,
        maxiter=max_iterations,
    )

    fitted_capacitance = float(fit_result.fitted_capacitance[0])
    fit_error = float(abs(fitted_capacitance - exp_capacitance))

    print(f"Found ion permittivity: {fit_result.fitted_permittivity:.2f}")
    print(
        f"Resulting capacitance: {fitted_capacitance:.2f} μF/cm² "
        f"(target: {exp_capacitance:.2f} μF/cm²)"
    )
    print(f"Error: {fit_error:.2f} μF/cm²")
    if fit_error > tolerance:
        print(f"Warning: fitted error exceeds requested tolerance of {tolerance:.3g} μF/cm²")

    return fit_result.fitted_permittivity, fitted_capacitance, fit_error
