"""Composite diffuse layer analytical approximation.

The Composite Diffuse Layer (CDL) model approximates the high-potential
Bikerman crowded layer by fully capping the counterion concentration in a
steric layer and matching it to an outer Gouy-Chapman diffuse tail.
"""

import numpy as np
import scipy.constants as sc
from scipy.integrate import quad

from .base import BaseElectrochemicalModel, ElectrochemicalSystem


class CDLModel(BaseElectrochemicalModel):
    """Analytical composite diffuse layer model with a capped counterion layer."""

    def __init__(
        self,
        system: ElectrochemicalSystem,
        maximum_packing_fraction: float = np.pi * np.sqrt(3) / 8,
    ):
        """Initialize the CDL model.

        Parameters
        ----------
        system : ElectrochemicalSystem
            Electrolyte and solvent definition.
        maximum_packing_fraction : float
            Counterion packing cap used to calculate the maximum concentration.
            The default matches the BCC-cell cap used in the CDL reference code.
        """
        super().__init__(system)
        self.system = system
        self.maximum_packing_fraction = maximum_packing_fraction

    def _counterion_index(self, potential: float) -> int:
        """Return the cation index for negative potentials, otherwise anion."""
        return 0 if potential < 0.0 else 1

    def _counterion(self, potential: float):
        return self.system.cation if potential < 0.0 else self.system.anion

    def _coion(self, potential: float):
        return self.system.anion if potential < 0.0 else self.system.cation

    def _thermal_voltage(self) -> float:
        return self.kT / sc.e

    def debye_length(self, potential=0) -> float:
        """Calculate the bulk Debye length using the solvent permittivity."""
        sum_cz2 = self.c_bulk * (self.system.cation.charge**2 + self.system.anion.charge**2)
        denominator = sc.e**2 * sum_cz2 * 1000 * sc.N_A
        return np.sqrt(self.epsilon * self.kT / denominator)

    def get_ion_steric_concentration(self, ion_index_or_radius) -> float:
        """Return the steric concentration cap for an ion in mol/L."""
        if isinstance(ion_index_or_radius, (int, np.integer)):
            radius_ang = self.ion_radii[ion_index_or_radius]
        else:
            radius_ang = float(ion_index_or_radius)

        ion_radius_m = radius_ang * 1e-10
        ion_volume = 4 * np.pi * ion_radius_m**3 / 3
        return self.maximum_packing_fraction / ion_volume / 1000 / sc.N_A

    def get_counterion_steric_concentration(self, potential: float) -> float:
        """Return the steric concentration cap for the counterion branch."""
        return self.get_ion_steric_concentration(self._counterion_index(potential))

    def get_threshold_potential(self, potential_or_ion) -> float:
        """Return the signed steric-onset threshold potential in V."""
        if hasattr(potential_or_ion, 'charge') and hasattr(potential_or_ion, 'radiusAng'):
            ion = potential_or_ion
        else:
            ion = self._counterion(float(potential_or_ion))

        c_cap = self.get_ion_steric_concentration(ion.radiusAng)
        return -self._thermal_voltage() * np.log(c_cap / self.c_bulk) / ion.charge

    def surface_volume_fraction(self, potential: float) -> float:
        """Return the surface counterion volume fraction in the CDL approximation."""
        _, volume, _ = self.get_ion_parameters(potential)
        H = self.get_steric_layer_thickness(potential)
        if H > 0.0:
            concentration = self.get_counterion_steric_concentration(potential)
        else:
            ion = self._counterion(potential)
            concentration = self.c_bulk * np.exp(-ion.charge * potential / self._thermal_voltage())

        return concentration * 1000 * sc.N_A * volume

    def get_steric_parameter_phi(self, potential, nes=None, relative_to_bulk=True):
        """Compatibility alias for the surface counterion volume fraction."""
        if nes is not None and len(nes) > 0:
            raise ValueError("CDLModel does not support non-electrostatic interaction terms.")
        if not relative_to_bulk:
            raise ValueError("CDLModel supports only relative_to_bulk=True.")
        return self.surface_volume_fraction(float(potential))

    def get_reduced_dielectric_from_potential(self, potential):
        """Return the bulk solvent dielectric used by the CDL approximation."""
        return self.epsilon_r

    def get_steric_layer_thickness(self, potential: float) -> float:
        """Calculate the CDL steric-layer thickness `H` in meters."""
        potential = float(potential)
        if abs(potential) < 1e-12:
            return 0.0

        counterion = self._counterion(potential)
        c_cap = self.get_counterion_steric_concentration(potential)
        nu = 2 * self.c_bulk / c_cap
        threshold = self.get_threshold_potential(counterion)

        if abs(potential) < abs(threshold):
            return 0.0

        term_potential = -counterion.charge * potential / self._thermal_voltage()
        term_sqrt_inner = (1 - 0.5 * nu) ** 2 + term_potential - np.log(2 / nu)
        term_sqrt_inner = max(term_sqrt_inner, 0.0)

        thickness = self.debye_length() * np.sqrt(2 * nu) * (
            -1 + 0.5 * nu + np.sqrt(term_sqrt_inner)
        )
        return max(float(thickness), 0.0)

    def charge_density(self, potential: float) -> float:
        """Calculate electrode surface charge density in C/m²."""
        potential = float(potential)
        if abs(potential) < 1e-12:
            return 0.0

        counterion = self._counterion(potential)
        threshold = self.get_threshold_potential(counterion)
        lambda_d = self.debye_length()
        vt = self._thermal_voltage()
        z_abs = abs(counterion.charge)

        if abs(potential) < abs(threshold):
            prefactor = 2 * self.epsilon * self.kT / (z_abs * sc.e * lambda_d)
            return prefactor * np.sinh(z_abs * potential / (2 * vt))

        c_cap = self.get_counterion_steric_concentration(potential)
        nu = 2 * self.c_bulk / c_cap
        rho_bulk = counterion.charge * sc.e * self.c_bulk * 1000 * sc.N_A
        term_potential = -counterion.charge * potential / vt
        term_sqrt_inner = (1 - 0.5 * nu) ** 2 + term_potential - np.log(2 / nu)
        term_sqrt_inner = max(term_sqrt_inner, 0.0)

        return -2 * rho_bulk * lambda_d * np.sqrt(2 / nu) * np.sqrt(term_sqrt_inner)

    def analytical_capacitance(self, potential: float) -> float:
        """Calculate analytical differential capacitance in μF/cm²."""
        potential = float(potential)
        counterion = self._counterion(potential)
        threshold = self.get_threshold_potential(counterion)
        lambda_d = self.debye_length()
        vt = self._thermal_voltage()
        z_abs = abs(counterion.charge)

        if abs(potential) < abs(threshold):
            capacitance_si = (self.epsilon / lambda_d) * np.cosh(z_abs * potential / (2 * vt))
        else:
            c_cap = self.get_counterion_steric_concentration(potential)
            nu = 2 * self.c_bulk / c_cap
            term_potential = -counterion.charge * potential / vt
            term_sqrt_inner = (1 - 0.5 * nu) ** 2 + term_potential - np.log(2 / nu)
            term_sqrt_inner = max(term_sqrt_inner, 1e-12)
            capacitance_si = (
                (self.epsilon / lambda_d)
                / np.sqrt(2 * nu)
                / np.sqrt(term_sqrt_inner)
            )

        return 100 * capacitance_si

    def get_capacitance(self, potential: float):
        """Return `(differential_capacitance, charge_density)` for compatibility."""
        return self.analytical_capacitance(potential), self.charge_density(potential)

    def _tail_potential(self, x_diff: float, matching_potential: float) -> float:
        vt = self._thermal_voltage()
        z_abs = abs(self._counterion(matching_potential).charge)
        gamma = np.tanh(z_abs * abs(matching_potential) / (4 * vt))
        value = gamma * np.exp(-x_diff / self.debye_length())
        value = np.clip(value, -1 + 1e-15, 1 - 1e-15)
        phi_magnitude = (4 * vt / z_abs) * np.arctanh(value)
        return np.sign(matching_potential) * phi_magnitude

    def electrostatic_potential_in_steric_layer(self, x: float, potential: float) -> float:
        """Calculate the CDL potential profile `Φ(x)` in V."""
        x = float(x)
        potential = float(potential)
        H = self.get_steric_layer_thickness(potential)
        sigma = self.charge_density(potential)
        counterion = self._counterion(potential)
        c_cap = self.get_counterion_steric_concentration(potential)
        rho_cap = counterion.charge * sc.e * c_cap * 1000 * sc.N_A

        if H > 0.0 and x <= H:
            return potential - (rho_cap / (2 * self.epsilon)) * x**2 - (sigma / self.epsilon) * x

        matching_potential = self.get_threshold_potential(counterion) if H > 0.0 else potential
        return self._tail_potential(max(x - H, 0.0), matching_potential)

    def electric_field_in_steric_layer(self, x: float, potential: float) -> float:
        """Calculate the CDL electric field profile `E(x)` in V/m."""
        x = float(x)
        potential = float(potential)
        H = self.get_steric_layer_thickness(potential)
        sigma = self.charge_density(potential)
        counterion = self._counterion(potential)
        c_cap = self.get_counterion_steric_concentration(potential)
        rho_cap = counterion.charge * sc.e * c_cap * 1000 * sc.N_A

        if H > 0.0 and x <= H:
            return sigma / self.epsilon + (rho_cap / self.epsilon) * x

        phi = self.electrostatic_potential_in_steric_layer(x, potential)
        vt = self._thermal_voltage()
        z_abs = abs(counterion.charge)
        return (2 * vt / (z_abs * self.debye_length())) * np.sinh(z_abs * phi / (2 * vt))

    def counterion_concentration_profile(self, x: float, potential: float) -> float:
        """Return the counterion concentration profile in mol/L."""
        H = self.get_steric_layer_thickness(potential)
        if H > 0.0 and x <= H:
            return self.get_counterion_steric_concentration(potential)

        phi = self.electrostatic_potential_in_steric_layer(x, potential)
        counterion = self._counterion(potential)
        return self.c_bulk * np.exp(-counterion.charge * phi / self._thermal_voltage())

    def coion_concentration_profile(self, x: float, potential: float) -> float:
        """Return the coion concentration profile in mol/L."""
        H = self.get_steric_layer_thickness(potential)
        if H > 0.0 and x <= H:
            return 0.0

        phi = self.electrostatic_potential_in_steric_layer(x, potential)
        coion = self._coion(potential)
        return self.c_bulk * np.exp(-coion.charge * phi / self._thermal_voltage())

    def ion_concentrations(self, x: float, potential: float):
        """Return cation and anion concentration profiles as a dictionary in mol/L."""
        counterion_concentration = self.counterion_concentration_profile(x, potential)
        coion_concentration = self.coion_concentration_profile(x, potential)
        if potential < 0:
            return {
                'cation': counterion_concentration,
                'anion': coion_concentration,
            }
        return {
            'cation': coion_concentration,
            'anion': counterion_concentration,
        }

    def concentration_profile_in_steric_layer(self, x, potential):
        """Compatibility profile returning counterion concentration and local potential."""
        return (
            self.counterion_concentration_profile(float(x), float(potential)),
            self.electrostatic_potential_in_steric_layer(float(x), float(potential)),
        )

    def volume_charge_density(self, x: float, potential: float) -> float:
        """Return local volume charge density in C/m³."""
        concentrations = self.ion_concentrations(x, potential)
        return (
            concentrations['cation'] * self.system.cation.charge
            + concentrations['anion'] * self.system.anion.charge
        ) * 1000 * sc.N_A * sc.e

    def get_entropic_energy(self, potential: float) -> float:
        """Calculate the entropic free-energy component in J/m²."""
        potential = float(potential)
        if abs(potential) < 1e-12:
            return 0.0

        H = self.get_steric_layer_thickness(potential)
        if H > 0.0:
            c_cap = self.get_counterion_steric_concentration(potential)
            n_cap = c_cap * 1000 * sc.N_A
            return self.kT * n_cap * np.log(c_cap / self.c_bulk) * H

        def integrand(x):
            phi = self.electrostatic_potential_in_steric_layer(x, potential)
            value = 0.0
            n_bulk = self.c_bulk * 1000 * sc.N_A
            for ion in (self.system.cation, self.system.anion):
                n = n_bulk * np.exp(-ion.charge * phi / self._thermal_voltage())
                value += n * (-ion.charge * phi / self._thermal_voltage())
            return value * self.kT

        result, _ = quad(integrand, 0, 20 * self.debye_length())
        return result

    def get_electrostatic_energy(self, potential: float) -> float:
        """Calculate the electrostatic free-energy component in J/m²."""
        potential = float(potential)
        if abs(potential) < 1e-12:
            return 0.0

        H = self.get_steric_layer_thickness(potential)
        if H > 0.0:
            sigma = self.charge_density(potential)
            counterion = self._counterion(potential)
            c_cap = self.get_counterion_steric_concentration(potential)
            rho_cap = counterion.charge * sc.e * c_cap * 1000 * sc.N_A
            term1 = (rho_cap**2 / 3) * H**3
            term2 = sigma * rho_cap * H**2
            term3 = sigma**2 * H
            return (term1 + term2 + term3) / (2 * self.epsilon)

        def integrand(x):
            electric_field = self.electric_field_in_steric_layer(x, potential)
            return 0.5 * self.epsilon * electric_field**2

        result, _ = quad(integrand, 0, 20 * self.debye_length())
        return result

    def get_steric_free_energy(self, potential: float) -> float:
        """Calculate the CDL steric free-energy component in J/m²."""
        potential = float(potential)
        if abs(potential) < 1e-12:
            return 0.0

        H = self.get_steric_layer_thickness(potential)
        if H <= 0.0:
            return 0.0

        sigma = self.charge_density(potential)
        counterion = self._counterion(potential)
        c_cap = self.get_counterion_steric_concentration(potential)
        rho_cap = counterion.charge * sc.e * c_cap * 1000 * sc.N_A
        n_cap = c_cap * 1000 * sc.N_A
        mu_cap = -self.kT * np.log(c_cap / self.c_bulk)
        term_inner = potential - (sigma * H) / (2 * self.epsilon) - (rho_cap * H**2) / (6 * self.epsilon)

        return (mu_cap * n_cap - rho_cap * term_inner) * H

    def get_total_energy(self, potential: float) -> float:
        """Calculate total CDL free energy in J/m²."""
        return (
            self.get_entropic_energy(potential)
            + self.get_electrostatic_energy(potential)
            + self.get_steric_free_energy(potential)
        )