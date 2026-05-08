"""Double-electrode cell wrappers built from single-interface EDL models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import scipy.constants as sc
from scipy import optimize


@dataclass(frozen=True)
class PotentialSplit:
    """Voltage split and charge-neutrality information for a full cell."""

    cell_voltage: float
    left_potential: float
    right_potential: float
    left_charge_density: float
    right_charge_density: float
    charge_balance_residual: float
    method: str


class DoubleElectrodeCell:
    """Full-cell wrapper that computes electrode potentials from charge neutrality.

    The wrapped model remains a single-electrode/interface model.  Users provide
    only the full-cell voltage, and this wrapper solves the left/right electrode
    potential split from:

        sigma(left_potential) + sigma(right_potential) = 0

    with right_potential = left_potential - cell_voltage.
    """

    def __init__(
        self,
        model,
        split_method: str = 'auto',
        separation_distance: float = 20.0,
        distance_unit: str = 'debye_lengths',
        split_tolerance: float = 1e-10,
    ):
        """Initialize the double-electrode cell.

        Parameters
        ----------
        model
            Single-electrode model with `charge_density`, `analytical_capacitance`,
            profile, and energy methods.
        split_method : {'auto', 'root', 'cdl_analytical'}
            Potential-split method. `auto` uses the CDL analytical expression when
            available and otherwise uses the generic charge-neutral root solve.
        separation_distance : float
            Default electrode separation for profile sampling.
        distance_unit : {'debye_lengths', 'm'}
            Unit for `separation_distance`.
        split_tolerance : float
            Charge-balance residual tolerance in C/m².
        """
        self.model = model
        self.split_method = split_method
        self.separation_distance = separation_distance
        self.distance_unit = distance_unit
        self.split_tolerance = split_tolerance

    def _supports_cdl_analytical_split(self) -> bool:
        """Return True when the wrapped model exposes the CDL cap quantities."""
        return (
            self.model.__class__.__name__ == 'CDLModel'
            and hasattr(self.model, 'get_ion_steric_concentration')
            and hasattr(self.model, 'get_threshold_potential')
            and hasattr(self.model, 'system')
        )

    def _split_residual(self, left_potential: float, cell_voltage: float) -> float:
        """Charge-neutrality residual for a candidate left potential."""
        right_potential = left_potential - cell_voltage
        return self.model.charge_density(left_potential) + self.model.charge_density(right_potential)

    def _build_split(self, cell_voltage: float, left_potential: float, method: str) -> PotentialSplit:
        """Construct a `PotentialSplit` object from a left electrode potential."""
        right_potential = left_potential - cell_voltage
        left_charge = self.model.charge_density(left_potential)
        right_charge = self.model.charge_density(right_potential)
        return PotentialSplit(
            cell_voltage=float(cell_voltage),
            left_potential=float(left_potential),
            right_potential=float(right_potential),
            left_charge_density=float(left_charge),
            right_charge_density=float(right_charge),
            charge_balance_residual=float(left_charge + right_charge),
            method=method,
        )

    def _solve_split_root(self, cell_voltage: float) -> PotentialSplit:
        """Solve the potential split with a generic charge-neutrality root solve."""
        cell_voltage = float(cell_voltage)
        if abs(cell_voltage) < 1e-14:
            return self._build_split(0.0, 0.0, 'root')

        lower, upper = (0.0, cell_voltage) if cell_voltage > 0.0 else (cell_voltage, 0.0)
        f_lower = self._split_residual(lower, cell_voltage)
        f_upper = self._split_residual(upper, cell_voltage)

        if abs(f_lower) <= self.split_tolerance:
            return self._build_split(cell_voltage, lower, 'root')
        if abs(f_upper) <= self.split_tolerance:
            return self._build_split(cell_voltage, upper, 'root')

        if f_lower * f_upper < 0.0:
            result = optimize.root_scalar(
                lambda phi: self._split_residual(phi, cell_voltage),
                bracket=(lower, upper),
                xtol=1e-12,
                rtol=1e-12,
            )
            if result.converged:
                split = self._build_split(cell_voltage, result.root, 'root')
                if abs(split.charge_balance_residual) <= max(self.split_tolerance, 1e-8):
                    return split

        result = optimize.minimize_scalar(
            lambda phi: abs(self._split_residual(phi, cell_voltage)),
            bounds=(lower, upper),
            method='bounded',
            options={'xatol': 1e-12},
        )
        split = self._build_split(cell_voltage, result.x, 'root-minimize')
        if not result.success or abs(split.charge_balance_residual) > max(self.split_tolerance, 1e-8):
            raise RuntimeError(
                "Could not find a charge-neutral voltage split for "
                f"cell_voltage={cell_voltage}. Residual={split.charge_balance_residual:.3e} C/m²."
            )
        return split

    def _cdl_counter_coions_for_left(self, cell_voltage: float):
        """Return left-electrode counterion and coion for a CDL full cell."""
        if cell_voltage < 0.0:
            return self.model.system.cation, self.model.system.anion
        return self.model.system.anion, self.model.system.cation

    def _ion_nu(self, ion) -> float:
        """Return CDL crowding parameter nu for an ion."""
        c_cap = self.model.get_ion_steric_concentration(ion.radiusAng)
        return 2 * self.model.c_bulk / c_cap

    def _solve_split_cdl_analytical(self, cell_voltage: float) -> PotentialSplit:
        """Use the analytical CDL voltage split from electrode charge neutrality."""
        if not self._supports_cdl_analytical_split():
            raise TypeError("The analytical CDL voltage split requires a CDLModel instance.")

        cell_voltage = float(cell_voltage)
        if abs(cell_voltage) < 1e-14:
            return self._build_split(0.0, 0.0, 'cdl_analytical')

        counterion, coion = self._cdl_counter_coions_for_left(cell_voltage)
        nu_counter = self._ion_nu(counterion)
        nu_coion = self._ion_nu(coion)

        left_potential = cell_voltage / 2
        right_potential = left_potential - cell_voltage
        left_threshold = self.model.get_threshold_potential(counterion)
        right_threshold = self.model.get_threshold_potential(coion)

        if abs(left_potential) > abs(left_threshold) or abs(right_potential) > abs(right_threshold):
            numerator = (
                cell_voltage * coion.charge * nu_counter / self.model._thermal_voltage()
                + nu_counter * (1 - nu_coion / 2) ** 2
                - nu_coion * (1 - nu_counter / 2) ** 2
                + nu_coion * np.log(2 / nu_counter)
                - nu_counter * np.log(2 / nu_coion)
            )
            denominator = coion.charge * nu_counter - counterion.charge * nu_coion
            left_potential = self.model._thermal_voltage() * numerator / denominator

        split = self._build_split(cell_voltage, left_potential, 'cdl_analytical')

        # Near mixed threshold cases the closed form can be less robust than the
        # generic root solve. Fall back rather than returning a non-neutral split.
        if abs(split.charge_balance_residual) > max(self.split_tolerance, 1e-7):
            return self._solve_split_root(cell_voltage)
        return split

    def get_potential_split(self, cell_voltage: float, method: str | None = None) -> PotentialSplit:
        """Compute left/right electrode potentials from the full-cell voltage."""
        selected_method = method or self.split_method
        if selected_method == 'auto':
            selected_method = 'cdl_analytical' if self._supports_cdl_analytical_split() else 'root'

        if selected_method in {'root', 'charge_neutrality'}:
            return self._solve_split_root(cell_voltage)
        if selected_method == 'cdl_analytical':
            return self._solve_split_cdl_analytical(cell_voltage)

        raise ValueError("split_method must be one of 'auto', 'root', 'charge_neutrality', or 'cdl_analytical'.")

    def charge_density(self, cell_voltage: float) -> float:
        """Return the left electrode surface charge density in C/m²."""
        return self.get_potential_split(cell_voltage).left_charge_density

    def charge_density_magnitude(self, cell_voltage: float) -> float:
        """Return the magnitude of the electrode surface charge density in C/m²."""
        return abs(self.charge_density(cell_voltage))

    def electrode_charge_densities(self, cell_voltage: float) -> Dict[str, float]:
        """Return left and right electrode surface charge densities in C/m²."""
        split = self.get_potential_split(cell_voltage)
        return {
            'left': split.left_charge_density,
            'right': split.right_charge_density,
        }

    def analytical_capacitance(self, cell_voltage: float) -> float:
        """Return full-cell differential capacitance in μF/cm²."""
        split = self.get_potential_split(cell_voltage)
        left_capacitance = self.model.analytical_capacitance(split.left_potential)
        right_capacitance = self.model.analytical_capacitance(split.right_potential)
        if left_capacitance == 0.0 or right_capacitance == 0.0:
            return 0.0
        return left_capacitance * right_capacitance / (left_capacitance + right_capacitance)

    def get_capacitance(self, cell_voltage: float):
        """Return `(full_cell_capacitance, left_charge_density)` for compatibility."""
        return self.analytical_capacitance(cell_voltage), self.charge_density(cell_voltage)

    def get_entropic_energy(self, cell_voltage: float) -> float:
        """Return the full-cell entropic energy component in J/m²."""
        split = self.get_potential_split(cell_voltage)
        return self.model.get_entropic_energy(split.left_potential) + self.model.get_entropic_energy(split.right_potential)

    def get_electrostatic_energy(self, cell_voltage: float) -> float:
        """Return the full-cell electrostatic energy component in J/m²."""
        split = self.get_potential_split(cell_voltage)
        return self.model.get_electrostatic_energy(split.left_potential) + self.model.get_electrostatic_energy(split.right_potential)

    def get_steric_free_energy(self, cell_voltage: float) -> float:
        """Return the full-cell steric free-energy component in J/m²."""
        split = self.get_potential_split(cell_voltage)
        return self.model.get_steric_free_energy(split.left_potential) + self.model.get_steric_free_energy(split.right_potential)

    def get_total_energy(self, cell_voltage: float) -> float:
        """Return total full-cell energy in J/m²."""
        return (
            self.get_entropic_energy(cell_voltage)
            + self.get_electrostatic_energy(cell_voltage)
            + self.get_steric_free_energy(cell_voltage)
        )

    def get_energy_components(self, cell_voltage: float) -> Dict[str, float]:
        """Return full-cell energy components in J/m²."""
        entropic = self.get_entropic_energy(cell_voltage)
        electrostatic = self.get_electrostatic_energy(cell_voltage)
        steric = self.get_steric_free_energy(cell_voltage)
        return {
            'entropic': entropic,
            'electrostatic': electrostatic,
            'steric': steric,
            'total': entropic + electrostatic + steric,
        }

    def get_separation_distance(self, separation_distance: float | None = None, distance_unit: str | None = None) -> float:
        """Return electrode separation in meters."""
        distance = self.separation_distance if separation_distance is None else separation_distance
        unit = self.distance_unit if distance_unit is None else distance_unit
        if unit in {'m', 'meter', 'meters'}:
            return float(distance)
        if unit in {'debye_lengths', 'debye_length', 'lambda_D'}:
            return float(distance) * self.model.debye_length(0.0)
        raise ValueError("distance_unit must be 'm' or 'debye_lengths'.")

    def _validate_separation(self, split: PotentialSplit, separation_m: float):
        """Raise if steric layers are too large for the chosen separation."""
        left_H = self.model.get_steric_layer_thickness(split.left_potential)
        right_H = self.model.get_steric_layer_thickness(split.right_potential)
        required = 2 * max(left_H, right_H)
        if separation_m < required:
            raise ValueError(
                "Electrode separation is too small for the steric layers: "
                f"separation={separation_m:.3e} m, required>{required:.3e} m."
            )

    def _single_interface_ion_concentrations(self, distance: float, potential: float) -> Dict[str, float]:
        """Return cation/anion concentrations for one interface in mol/L."""
        if hasattr(self.model, 'ion_concentrations'):
            return self.model.ion_concentrations(distance, potential)

        counterion_concentration, _ = self.model.concentration_profile_in_steric_layer(distance, potential)
        H = self.model.get_steric_layer_thickness(potential)
        coion_concentration = 0.0 if H > 0.0 and distance <= H else self.model.c_bulk

        if potential < 0.0:
            return {
                'cation': counterion_concentration,
                'anion': coion_concentration,
            }
        return {
            'cation': coion_concentration,
            'anion': counterion_concentration,
        }

    def potential_profile(self, x: float, cell_voltage: float, separation_distance: float | None = None, distance_unit: str | None = None) -> float:
        """Return the full-cell potential profile at position `x` in V."""
        split = self.get_potential_split(cell_voltage)
        separation_m = self.get_separation_distance(separation_distance, distance_unit)
        return (
            self.model.electrostatic_potential_in_steric_layer(x, split.left_potential)
            + self.model.electrostatic_potential_in_steric_layer(separation_m - x, split.right_potential)
        )

    def electric_field_profile(self, x: float, cell_voltage: float, separation_distance: float | None = None, distance_unit: str | None = None) -> float:
        """Return the full-cell electric field profile at position `x` in V/m."""
        split = self.get_potential_split(cell_voltage)
        separation_m = self.get_separation_distance(separation_distance, distance_unit)
        return (
            self.model.electric_field_in_steric_layer(x, split.left_potential)
            - self.model.electric_field_in_steric_layer(separation_m - x, split.right_potential)
        )

    def ion_concentrations(self, x: float, cell_voltage: float, separation_distance: float | None = None, distance_unit: str | None = None) -> Dict[str, float]:
        """Return full-cell cation and anion concentrations at `x` in mol/L."""
        split = self.get_potential_split(cell_voltage)
        separation_m = self.get_separation_distance(separation_distance, distance_unit)
        left = self._single_interface_ion_concentrations(x, split.left_potential)
        right = self._single_interface_ion_concentrations(separation_m - x, split.right_potential)
        c_bulk = self.model.c_bulk
        return {
            'cation': c_bulk + (left['cation'] - c_bulk) + (right['cation'] - c_bulk),
            'anion': c_bulk + (left['anion'] - c_bulk) + (right['anion'] - c_bulk),
        }

    def volume_charge_density(self, x: float, cell_voltage: float, separation_distance: float | None = None, distance_unit: str | None = None) -> float:
        """Return full-cell volume charge density at `x` in C/m³."""
        concentrations = self.ion_concentrations(x, cell_voltage, separation_distance, distance_unit)
        return (
            concentrations['cation'] * self.model.system.cation.charge
            + concentrations['anion'] * self.model.system.anion.charge
        ) * 1000 * sc.N_A * sc.e

    def sample_profiles(
        self,
        cell_voltage: float,
        separation_distance: float | None = None,
        distance_unit: str | None = None,
        num_points: int = 200,
        validate_separation: bool = True,
    ):
        """Sample full-cell potential, field, concentration, and charge profiles."""
        split = self.get_potential_split(cell_voltage)
        separation_m = self.get_separation_distance(separation_distance, distance_unit)
        if validate_separation:
            self._validate_separation(split, separation_m)

        x = np.linspace(0.0, separation_m, num_points)
        potential = []
        electric_field = []
        cation = []
        anion = []
        volume_charge_density = []

        for x_value in x:
            potential.append(self.potential_profile(float(x_value), cell_voltage, separation_m, 'm'))
            electric_field.append(self.electric_field_profile(float(x_value), cell_voltage, separation_m, 'm'))
            concentrations = self.ion_concentrations(float(x_value), cell_voltage, separation_m, 'm')
            cation.append(concentrations['cation'])
            anion.append(concentrations['anion'])
            volume_charge_density.append(self.volume_charge_density(float(x_value), cell_voltage, separation_m, 'm'))

        return {
            'x': x,
            'potential': np.array(potential, dtype=float),
            'electric_field': np.array(electric_field, dtype=float),
            'cation_concentration': np.array(cation, dtype=float),
            'anion_concentration': np.array(anion, dtype=float),
            'volume_charge_density': np.array(volume_charge_density, dtype=float),
            'cell_voltage': float(cell_voltage),
            'separation_distance': float(separation_m),
            'left_potential': split.left_potential,
            'right_potential': split.right_potential,
            'left_steric_layer_thickness': float(self.model.get_steric_layer_thickness(split.left_potential)),
            'right_steric_layer_thickness': float(self.model.get_steric_layer_thickness(split.right_potential)),
        }