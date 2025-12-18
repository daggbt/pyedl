"""
Steric models (Carnahan-Starling, Liu) for electrochemical calculations.
"""

import numpy as np
import scipy.constants as sc
from scipy import optimize

from .base import BaseElectrochemicalModel, ElectrochemicalSystem


class StericModel(BaseElectrochemicalModel):
    """
    Model for electrochemical capacitance calculations with steric effects.
    """
    
    def __init__(self, 
                 system: ElectrochemicalSystem,
                 steric_model: str = 'cs'):
        """
        Initialize the electrochemical model with a system.
        
        Parameters:
        -----------
        system : ElectrochemicalSystem
            The electrochemical system containing ions and solvent
        steric_model : str
            Steric model type ('cs' for Carnahan-Starling or 'liu' for Liu model)
        """
        super().__init__(system)
        self.steric_model = steric_model
    
    def carnahan_starling(self, phi, relative_to_bulk=True):
        """Carnahan-Starling model for steric energy."""
        cs_energy = self.kT * phi * (8 - 9 * phi + 3 * phi**2) / (1 - phi)**3
        
        if relative_to_bulk:
            cs_energy -= self.kT * self.volfrac_b * (8 - 9 * self.volfrac_b + 3 * self.volfrac_b**2) / (1 - self.volfrac_b)**3
            
        return cs_energy
    
    def liu_model(self, phi, relative_to_bulk=True):
        """Liu model for steric energy."""
        liu_term = self.kT * (-5 * np.log(1 - phi) / 13 - 
                         (phi * (phi * (phi * (13 * (5 - 3 * phi) * phi - 146) + 418) - 396)) / 
                         52 / (1 - phi)**3)
        
        if relative_to_bulk:
            liu_term -= self.kT * (-5 * np.log(1 - self.volfrac_b) / 13 - 
                             (self.volfrac_b * (self.volfrac_b * (self.volfrac_b * 
                             (13 * (5 - 3 * self.volfrac_b) * self.volfrac_b - 146) + 418) - 396)) / 
                             52 / (1 - self.volfrac_b)**3)
            
        return liu_term
    
    def get_steric_energy(self, phi, relative_to_bulk=True):
        """Get steric energy based on the selected model."""
        if self.steric_model == 'cs':
            return self.carnahan_starling(phi, relative_to_bulk)
        else:
            return self.liu_model(phi, relative_to_bulk)
    
    def get_reduced_dielectric_from_potential(self, potential):
        """
        Get the reduced dielectric constant for a given potential.
        Uses caching to avoid recalculating values.
        """
        # Check if we've already calculated this value
        potential_key = round(potential, 9)  # Round to avoid float precision issues
        if potential_key in self.reduced_dielectric_cache:
            return self.reduced_dielectric_cache[potential_key]
        
        # Calculate volume fraction at interface
        try:
            q_i, v_i, epsilon_i = self.get_ion_parameters(potential)
            volfrac_0 = self.get_steric_parameter_phi(potential, [])

            # Calculate reduced dielectric constant
            reduced_dielectric = self.get_reduced_dielectric_from_volfrac(epsilon_i, volfrac_0)
            # Cache the result
            self.reduced_dielectric_cache[potential_key] = reduced_dielectric
            return reduced_dielectric
        
        except Exception as e:
            # Fallback to bulk dielectric constant on error
            print(f"Warning: Could not calculate reduced dielectric for potential {potential}V: {e}")
            return self.epsilon_r
    
    def debye_length(self, potential=0):
        """Calculate the Debye length in the electrolyte."""
        # Get reduced dielectric constant for this potential
        reduced_dielectric = self.get_reduced_dielectric_from_potential(potential)
        
        # Calculate Debye length with reduced dielectric
        epsilon_adjusted = sc.epsilon_0 * reduced_dielectric
        return np.sqrt(epsilon_adjusted * self.kT / (2 * self.c_bulk * sc.e**2 * 1000 * sc.N_A))
    
    def zero_func_phi(self, phi, potential, nes_all=None, relative_to_bulk=True):
        """Function to find the root for the volume fraction calculation."""
        # Get ion parameters
        q, v, _ = self.get_ion_parameters(potential)
        
        # Calculate excess potential
        excess_pot_nonsteric = self.get_electrostatic_interaction(q, potential)
        
        # Add non-electrostatic interactions if provided
        if nes_all is not None and len(nes_all) > 0:
            i = 1 if potential < 0 else -1
            nes_ion = nes_all[i]
            
            if isinstance(nes_ion, (list, tuple)):
                # Multiple nes for ion
                for nes_val in nes_ion[i]:
                    excess_pot_nonsteric += nes_val
            else:
                # Single nes for ion
                excess_pot_nonsteric += nes_ion
        
        # Calculate total excess potential
        pot_excess = excess_pot_nonsteric
        
        # Add steric energy if calculating relative to bulk
        if relative_to_bulk:
            pot_excess += self.get_steric_energy(phi, relative_to_bulk=True)
        
        # Calculate volume fraction
        volfrac_ion = 1000 * sc.N_A * v * self.c_bulk * np.exp(-pot_excess / self.kT)
        
        # Return difference between calculated and input phi
        return phi - volfrac_ion
    
    def get_steric_parameter_phi(self, potential, nes=None, relative_to_bulk=True):
        """Find the equilibrium volume fraction at the electrode surface."""
        # Check if we've already calculated this value
        potential_key = round(potential, 9)  # Round to avoid float precision issues
        if potential_key in self.surface_volume_fraction_cache:
            return self.surface_volume_fraction_cache[potential_key]
        
        if nes is None:
            nes = []
            
        # Define root finding function
        root_solvent = lambda phi: self.zero_func_phi(phi, potential, nes, relative_to_bulk)
        
        # First try using a more robust method - Brent's method with bracketing
        try:
            # Initial bracket bounds
            edge_limit_low = 1e-10
            edge_limit_high = 1 - 1e-10
            
            # Check if function values have opposite signs
            f_low = root_solvent(edge_limit_low)
            f_high = root_solvent(edge_limit_high)
                        
            if f_low * f_high > 0:
                # Function values don't have opposite signs, we need a different approach
                found_bracket = False
                
                # Try multiple sub-ranges
                test_points = np.linspace(edge_limit_low, edge_limit_high, 20)
                
                for i in range(len(test_points) - 1):
                    a, b = test_points[i], test_points[i+1]
                    fa, fb = root_solvent(a), root_solvent(b)
                    
                    if fa * fb <= 0:  # Found a bracket
                        edge_limit_low, edge_limit_high = a, b
                        found_bracket = True
                        break
                
                if not found_bracket:
                    # If we can't find a bracket, try different methods
                    
                    # First try to use a minimum finder approach
                    def objective(phi):
                        return abs(root_solvent(phi))
                    
                    result = optimize.minimize_scalar(objective, bounds=(edge_limit_low, edge_limit_high), 
                                                    method='bounded')
                    
                    if result.success and abs(root_solvent(result.x)) < 1e-20:
                        root = result.x
                        self.surface_volume_fraction_cache[potential_key] = root
                        return root
                    
                    # If that fails, fall back to an approximation
                    phi = self.volfrac_b
                    for _ in range(10):
                        # Calculate g(φ) directly to avoid the phi - g(phi) form
                        q, v, _ = self.get_ion_parameters(potential)
                        excess_pot_nonsteric = self.get_electrostatic_interaction(q, potential)
                        
                        if relative_to_bulk:
                            excess_pot_nonsteric += self.get_steric_energy(phi, relative_to_bulk=True)
                        
                        # Direct calculation of g(φ)
                        new_phi = 1000 * sc.N_A * v * self.c_bulk * np.exp(-excess_pot_nonsteric / self.kT)
                        
                        # Check for convergence
                        if abs(new_phi - phi) < 1e-6:
                            break
                            
                        # Update with damping to avoid oscillations
                        phi = 0.5 * phi + 0.5 * new_phi
                    
                    # Use this as our result
                    root = phi
                    
                    # Ensure the result is physically reasonable
                    if root <= 0 or root >= 1:
                        root = max(1e-6, min(0.999, root))
                    
                    # Cache the result
                    self.surface_volume_fraction_cache[potential_key] = root

                    return root
            
            # If we found a valid bracket or didn't need to search, use root_scalar
            root = optimize.root_scalar(root_solvent, bracket=[edge_limit_low, edge_limit_high]).root
            
        except Exception as e:
            # If root_scalar fails, try a more general approach - fsolve
            try:
                # Initial guess based on bulk volume fraction
                initial_guess = self.volfrac_b
                
                # Define function for fsolve (which doesn't require different signs)
                def equation_for_solver(phi):
                    return self.zero_func_phi(phi, potential, nes, relative_to_bulk)
                
                solution = optimize.fsolve(equation_for_solver, initial_guess, full_output=True)
                
                # Check for convergence
                if solution[2] != 1:
                    raise RuntimeError(f"fsolve did not converge: {solution[3]}")
                    
                root = solution[0][0]
                
            except Exception as nested_e:
                # If all else fails, use a direct approximation
                
                # For positive potentials, expect volume fraction to be lower than bulk
                # For negative potentials, expect volume fraction to be higher than bulk
                if potential > 0:
                    # Approximate reduction based on potential magnitude
                    factor = max(0.1, np.exp(-abs(potential) / self.kT / 10))
                    root = max(1e-6, self.volfrac_b * factor)
                else:
                    # Approximate increase based on potential magnitude
                    factor = min(10, np.exp(abs(potential) / self.kT / 10))
                    root = min(0.999, self.volfrac_b * factor)
        
        # Ensure the result is physically reasonable
        if root <= 0 or root >= 1:
            root = max(1e-6, min(0.999, root))
        
        # Cache the result
        self.surface_volume_fraction_cache[potential_key] = root
        
        return root
    
    def get_steric_layer_thickness(self, potential):
        """Get the steric layer thickness for a given potential."""
        # Check if we've already calculated this value
        potential_key = round(potential, 9)  # Round to avoid float precision issues
        if potential_key in self.steric_thickness_cache:
            return self.steric_thickness_cache[potential_key]
        
        # Get ion parameters
        q, v, _ = self.get_ion_parameters(potential)

        # Calculate volume fraction at electrode surface
        volfrac_surface = self.get_steric_parameter_phi(potential, [])
        
        # Convert to concentration
        concentration_surface = volfrac_surface / v / 1000 / sc.N_A

        # Get reduced dielectric constant
        reduced_dielectric = self.get_reduced_dielectric_from_potential(potential)
        epsilon_adjusted = sc.epsilon_0 * reduced_dielectric

        layer_thickness = np.sqrt(-6 * potential * epsilon_adjusted / q / 1000 / sc.N_A / (concentration_surface + 2 * self.c_bulk))

        # Cache the result
        self.steric_thickness_cache[potential_key] = layer_thickness
        
        return layer_thickness
    
    def concentration_profile_in_steric_layer(self, x, potential):
        """Calculate concentration profile as a function of distance from electrode."""
        # Get ion parameters
        q, v, _ = self.get_ion_parameters(potential)
        
        # Calculate volume fraction at electrode surface
        volfrac_surface = self.get_steric_parameter_phi(potential, [])
        
        # Convert to concentration
        concentration_surface = volfrac_surface / v / 1000 / sc.N_A
        
        # Calculate characteristic length with reduced dielectric
        H = self.get_steric_layer_thickness(potential)
        
        # Calculate concentration and potential at position x
        if x <= H:
            conc = x * (-concentration_surface + self.c_bulk) / H + concentration_surface
        else:
            conc = self.c_bulk
            phi = 0
            
        return conc, phi
    
    def electrostatic_potential_in_steric_layer(self, x, potential):
        """Calculate electrostatic potential as a function of distance from electrode."""
        # Get ion parameters
        q, v, _ = self.get_ion_parameters(potential)
        
        # Calculate volume fraction at electrode surface
        volfrac_surface = self.get_steric_parameter_phi(potential, [])
        
        # Convert to concentration
        c_s = volfrac_surface / v / 1000 / sc.N_A
        
        # Get reduced dielectric constant
        reduced_dielectric = self.get_reduced_dielectric_from_potential(potential)
        epsilon_adjusted = sc.epsilon_0 * reduced_dielectric
        
        # Calculate characteristic length with reduced dielectric
        H = self.get_steric_layer_thickness(potential)
        
        # Calculate concentration and potential at position x
        if x <= H:
            phi = q * (c_s - self.c_bulk) * x**3 / 6 / H 
            phi -= q * c_s * x**2 / 2 
            phi +=  q * (c_s + self.c_bulk) * H * x / 2 
            phi -= q * (c_s + 2 * self.c_bulk) * H**2 / 6 
            phi /= epsilon_adjusted
            phi *= 1000 * sc.N_A
        else:
            phi = 0
            
        return phi
    
    def electric_field_in_steric_layer(self, x, potential):
        """Calculate electric field at distance x from electrode."""
        # Get ion parameters
        q, v, _ = self.get_ion_parameters(potential)
        
        # Calculate volume fraction at electrode surface
        volfrac_surface = self.get_steric_parameter_phi(potential, [])
        
        # Convert to concentration
        c_s = volfrac_surface / v / 1000 / sc.N_A
        
        # Get reduced dielectric constant
        reduced_dielectric = self.get_reduced_dielectric_from_potential(potential)
        epsilon_adjusted = sc.epsilon_0 * reduced_dielectric
        
        # Calculate characteristic length with reduced dielectric
        H = self.get_steric_layer_thickness(potential)
        
        # Calculate concentration and potential at position x
        if x <= H:
            efield = - q  * (c_s - self.c_bulk) * x**2 / 2 / H 
            efield += q * c_s * x 
            efield -=  q * (c_s + self.c_bulk) * H / 2 
            efield /= epsilon_adjusted
            efield *= 1000 * sc.N_A
        else:
            efield = 0
            
        return efield
    
    def charge_density(self, potential):
        """
        Calculate charge density at the electrode using:
        σ = sgn(Φ₀)√[(-6εΦ₀/4)·(ze(c₀+cᵦ)²/(c₀+2cᵦ))]
        
        Parameters:
        -----------
        potential : float
            Electric potential in V
            
        Returns:
        --------
        float: Charge density in C/m²
        """
        # Special handling for zero potential
        if abs(potential) < 1e-10:
            return 0.0  # At equilibrium, there's no net charge
        
        # Get ion parameters
        q, v, _ = self.get_ion_parameters(potential)
        
        try:
            # Calculate volume fraction at electrode surface
            volfrac_surface = self.get_steric_parameter_phi(potential, [])
            
            # Convert to concentration
            c_s = volfrac_surface / v / 1000 / sc.N_A
            
            # Get reduced dielectric constant
            try:
                reduced_dielectric = self.get_reduced_dielectric_from_potential(potential)
            except:
                reduced_dielectric = self.epsilon_r
            
            epsilon_adjusted = sc.epsilon_0 * reduced_dielectric
                        
            # Calculate using the provided formula
            # sgn(Φ₀)√[(-6εΦ₀/4)·(ze(c₀+cᵦ)²/(c₀+2cᵦ))]
            sign = np.sign(potential)
            
            # Term under the square root
            term1 = -6 * epsilon_adjusted * potential / 4
            term2 = q * (c_s + self.c_bulk)**2 / (c_s + 2 * self.c_bulk)
            unit_conversion = 1000 * sc.N_A # to change the mol/L to SI unit
            
            # Calculate charge density
            charge_density = sign * np.sqrt(term1 * term2 * unit_conversion)
            
            return charge_density
            
        except Exception as e:
            # Handle errors gracefully
            print(f"Error calculating charge density at potential {potential}V: {str(e)}")
            
            # Fallback to linear approximation for charge density
            debye_len = self.debye_length(potential)
            return -self.epsilon_r * sc.epsilon_0 * potential / debye_len
    
    def analytical_capacitance(self, potential):
        """
        Calculate capacitance analytically for a given potential.
        
        Parameters:
        -----------
        potential : float
            Electric potential in V
            
        Returns:
        --------
        float: Capacitance in μF/cm²
        """
        beta = 1/self.kT
        
        # Special case for zero potential
        if abs(potential) < 1e-10:        
            # At zero potential, capacitance is determined by the Debye length
            debye_len = self.debye_length(potential)
            _, v_i, epsilon_i = self.get_ion_parameters(potential)
            alpha_i = max(self.ion_polarizabilities)
            # Calculate reduced dielectric constant
            reduced_dielectric = self.get_reduced_dielectric_from_volfrac(epsilon_i, self.volfrac_b)
            capacitance = 100 * reduced_dielectric * sc.epsilon_0 / debye_len  # Convert to μF/cm²
            return capacitance
        else:
            # Get ion parameters
            q, v, _ = self.get_ion_parameters(potential)
            
            # Calculate volume fraction at electrode surface
            volfrac_surface = self.get_steric_parameter_phi(potential, [])
            
            # Convert to concentration
            c_s = volfrac_surface / v / 1000 / sc.N_A
            
            reduced_dielectric = self.get_reduced_dielectric_from_potential(potential)
            epsilon_adjusted = sc.epsilon_0 * reduced_dielectric
            
            # Calculate characteristic length
            H = self.get_steric_layer_thickness(potential)
            
            # First term of capacitance
            cap = beta * q * c_s * potential 
            cap /= volfrac_surface * (2 * volfrac_surface - 8) / (1 - volfrac_surface)**4 -1
            cap *= (c_s + 3 * self.c_bulk) / (c_s + 2 * self.c_bulk)**2
            cap += (c_s + self.c_bulk) / (c_s + 2 * self.c_bulk)
            cap *= 3 * epsilon_adjusted / 2 / H
            
            # Convert to μF/cm²
            return 100 * cap
    
    def get_capacitance(self, potential):
        """
        Calculate differential capacitance at given potential.
        
        Parameters:
        -----------
        potential : float
            Electric potential in V
            
        Returns:
        --------
        tuple: (capacitance, charge_density)
            Differential capacitance in µF/cm² and charge density in C/m²
        """
        try:
            # Calculate charge density at given potential
            c_density = self.charge_density(potential)
            
            # Very small step for numerical derivative
            del_pot = min(1e-6, abs(potential) * 1e-3)  # Adaptive step size
            
            # Calculate charge density at slightly higher potential
            c_density_plus = self.charge_density(potential + del_pot)
            
            # Calculate capacitance as derivative of charge density
            del_sigma = c_density_plus - c_density
            capacitance = 100 * del_sigma / del_pot  # Convert to μF/cm²
            
            return capacitance, c_density
            
        except Exception as e:
            # Handle errors gracefully
            print(f"Error calculating capacitance at potential {potential}V: {str(e)}")
            
            # Fallback to Gouy-Chapman-Stern approximation
            debye_len = self.debye_length(potential)
            capacitance = 100 * self.epsilon_r * sc.epsilon_0 / debye_len  # Convert to μF/cm²
            c_density = -self.epsilon_r * sc.epsilon_0 * potential / debye_len
            
            return capacitance, c_density

    def get_entropic_energy(self, potential):
        """
        Calculate the entropic component of the grand potential.
        Eq. 17 in manuscript.
        """
        if abs(potential) < 1e-10:
            return 0.0
            
        # Get parameters
        q, v, _ = self.get_ion_parameters(potential)
        volfrac_surface = self.get_steric_parameter_phi(potential, [])
        
        # Concentrations in number density (m^-3)
        c_0 = volfrac_surface / v 
        c_b = self.c_bulk * 1000 * sc.N_A
        
        H = self.get_steric_layer_thickness(potential)
        
        # Eq 17
        # Ω_en / kBT = (H/2) * [ c0^2/(c0-cb) * ln(c0/cb) + (cb - 3c0)/2 ]
        
        # Handle singularity when c0 -> cb (low potential)
        if abs(c_0 - c_b) < 1e-10:
             return 0.0

        term1 = (c_0**2 / (c_0 - c_b)) * np.log(c_0 / c_b)
        term2 = (c_b - 3 * c_0) / 2
        
        omega_en_kbt = (H / 2) * (term1 + term2)
        
        return omega_en_kbt * self.kT

    def get_electrostatic_energy(self, potential):
        """
        Calculate the electrostatic component of the grand potential.
        Eq. 18 in manuscript.
        """
        if abs(potential) < 1e-10:
            return 0.0

        # Get parameters
        q, v, _ = self.get_ion_parameters(potential)
        volfrac_surface = self.get_steric_parameter_phi(potential, [])
        
        # Concentrations in number density (m^-3)
        c_0 = volfrac_surface / v 
        c_b = self.c_bulk * 1000 * sc.N_A
        
        H = self.get_steric_layer_thickness(potential)
        
        reduced_dielectric = self.get_reduced_dielectric_from_potential(potential)
        epsilon_adjusted = sc.epsilon_0 * reduced_dielectric
        
        # Eq 18
        # Ω_el = (z^2 e^2 H^3 / 120ε) * [3c0^2 + 9c0cb + 8cb^2]
        # z*e is q
        
        prefactor = (q**2 * H**3) / (120 * epsilon_adjusted)
        bracket = 3 * c_0**2 + 9 * c_0 * c_b + 8 * c_b**2
        
        return prefactor * bracket

    def get_steric_free_energy(self, potential):
        """
        Calculate the steric component of the grand potential.
        Eq. 19 in manuscript.
        """
        if abs(potential) < 1e-10:
            return 0.0

        # Get parameters
        q, v, _ = self.get_ion_parameters(potential)
        volfrac_surface = self.get_steric_parameter_phi(potential, [])
        
        # Concentrations in number density (m^-3)
        c_0 = volfrac_surface / v 
        c_b = self.c_bulk * 1000 * sc.N_A
        
        phi_0 = volfrac_surface
        phi_b = self.volfrac_b
        
        H = self.get_steric_layer_thickness(potential)
        
        # Eq 19
        # Ω_st / kBT = (H/v) * [ (1 + 3(1-φ0)(1-φb))/((1-φ0)(1-φb)) + 2/(φb-φ0) * ln((1-φ0)/(1-φb)) ] 
        #              - (H/2)(cb-c0) * [ (4-3φb)/(1-φb)^2 ]
        
        # Handle singularity when phi_0 -> phi_b
        if abs(phi_0 - phi_b) < 1e-10:
            return 0.0

        term1_num = 1 + 3 * (1 - phi_0) * (1 - phi_b)
        term1_den = (1 - phi_0) * (1 - phi_b)
        term1 = term1_num / term1_den
        
        term2 = (2 / (phi_b - phi_0)) * np.log((1 - phi_0) / (1 - phi_b))
        
        bracket1 = term1 + term2
        part1 = (H / v) * bracket1
        
        bracket2 = (4 - 3 * phi_b) / (1 - phi_b)**2
        part2 = (H / 2) * (c_b - c_0) * bracket2
        
        omega_st_kbt = part1 - part2
        
        return omega_st_kbt * self.kT

    def get_total_energy(self, potential):
        """Calculate total grand potential energy."""
        return self.get_entropic_energy(potential) + \
               self.get_electrostatic_energy(potential) + \
               self.get_steric_free_energy(potential)
