"""
Base classes for electrochemical models.
"""

import numpy as np
import scipy.constants as sc
from typing import Tuple

from ..materials.ions import Ion
from ..materials.solvents import Solvent, solvent_database


class ElectrochemicalSystem:
    """Represents an electrochemical system with specified ions and solvent."""
    
    def __init__(self, cation: Ion, anion: Ion, solvent: Solvent, 
                 concentration: float, temperature: float = 298.15,
                 n_hydration_cation: float = 0, n_hydration_anion: float = 0):
        """
        Initialize the electrochemical system.
        
        Parameters:
        -----------
        cation : Ion
            Positive ion in the system
        anion : Ion
            Negative ion in the system
        solvent : Solvent
            Solvent properties
        concentration : float
            Bulk concentration in mol/L
        temperature : float
            Temperature in Kelvin
        n_hydration_cation : float
            Number of water molecules hydrating the cation
        n_hydration_anion : float
            Number of water molecules hydrating the anion
        """
        self.cation = cation
        self.anion = anion
        self.solvent = solvent
        self.concentration = concentration
        self.temperature = temperature
        self.n_hydration_cation = n_hydration_cation
        self.n_hydration_anion = n_hydration_anion
        
        # Calculate effective ion polarizabilities (hydrated)
        water_pol = solvent_database['water'].solventPolarizability
        self.cation_polarizability = cation.get_hydrated_polarizability(n_hydration_cation, water_pol)
        self.anion_polarizability = anion.get_hydrated_polarizability(n_hydration_anion, water_pol)
        
    def get_ion_radii(self) -> Tuple[float, float]:
        """Get ion radii in Angstroms as a tuple (cation, anion)."""
        return (self.cation.radiusAng, self.anion.radiusAng)
    
    def get_ion_polarizabilities(self) -> Tuple[float, float]:
        """Get ion polarizabilities as a tuple (cation, anion)."""
        return (self.cation_polarizability, self.anion_polarizability)
    
    def get_dielectric_constant(self) -> float:
        """Get dielectric constant of the solvent."""
        return self.solvent.dielectricConstant


class BaseElectrochemicalModel:
    """
    Base model for electrochemical calculations.
    """
    
    def __init__(self, system: ElectrochemicalSystem):
        """
        Initialize the electrochemical model with a system.
        
        Parameters:
        -----------
        system : ElectrochemicalSystem
            The electrochemical system containing ions and solvent
        """
        # Physical constants
        self.temperature = system.temperature
        self.epsilon_r = system.get_dielectric_constant()
        self.epsilon = system.solvent.get_permittivity()
        self.kT = sc.k * self.temperature
                
        # Concentration parameters
        self.c_bulk = system.concentration
                
        # Ion parameters - convert from Ion objects to original format
        self.ion_radii = system.get_ion_radii()
        self.ion_volumes = [system.cation.get_volume_m3(), system.anion.get_volume_m3()]
        
        # Volume fraction in bulk
        self.volfrac_b = self.c_bulk * sum(self.ion_volumes) * 1000 * sc.N_A
        
        # Ion polarizabilities (hydrated)
        self.ion_polarizabilities = system.get_ion_polarizabilities()
        
        # Initialize caches
        self.surface_volume_fraction_cache = {}
        self.reduced_dielectric_cache = {}
        self.steric_thickness_cache = {}

        self.ion_permitivities = [self.get_ion_permittivity(self.ion_polarizabilities[i], self.ion_volumes[i]) 
                                  for i in range(len(self.ion_volumes))]

    def get_ion_parameters(self, potential):
        """
        Returns the parameters for the ion at the specified potential.
        
        Parameters:
        -----------
        potential : float
            Electric potential in V
            
        Returns:
        --------
        tuple: (charge, volume, epsilon_i)
            Charge, volume, and permittivity of the counterion
        """
        # Counterion is determined by the sign of the potential
        counterion_index = 0 if potential < 0 else 1
        charge = -1 * sc.e if potential >= 0 else 1 * sc.e
        volume = self.ion_volumes[counterion_index]
        epsilon_i = self.ion_permitivities[counterion_index]
        
        return charge, volume, epsilon_i

    def get_ion_permittivity(self, alpha_i, volume_i):
        """Calculate the permittivity of an ion from its polarizability."""
        v_i_cubic_angstrom = volume_i * 1e30
        k = 4 * sc.pi * alpha_i / 3 / v_i_cubic_angstrom

        # Compute the numerator and denominator of the Clausius-Mossotti equation
        numerator = -1 - 2 * k
        denominator = k - 1

        # Calculate the permittivity of the ion
        epsilon_i = numerator / denominator
        
        return epsilon_i
    
    def get_reduced_dielectric_from_volfrac(self, epsilon_i, volfrac_0):
        """Calculate reduced dielectric constant from volume fraction."""
        ### https://pubs.acs.org/doi/epdf/10.1021/la2025445?ref=article_openPDF
        numerator = (1 + 2 * volfrac_0) * epsilon_i + 2 * (1 - volfrac_0) * self.epsilon_r
        denominator = (1 - volfrac_0) * epsilon_i + (2 + volfrac_0) * self.epsilon_r

        # Check to avoid division by zero
        if denominator == 0:
            raise ValueError("Denominator becomes zero. Check the values of dielectric_constant and surface volume fraction.")
        
        epsilon_eff = self.epsilon_r * numerator / denominator

        return epsilon_eff

    def get_electrostatic_interaction(self, charge, potential):
        """Calculate the electrostatic interaction energy."""
        return charge * potential
