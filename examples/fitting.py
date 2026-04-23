#!/usr/bin/env python3
"""Example script demonstrating counterion permittivity fitting."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

src_path = Path(__file__).resolve().parents[1] / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pyedl import ElectrochemicalSystem, StericModel, fit_counterion_permittivity_curve
from pyedl import ion_database, solvent_database


def main():
    print("Example: fitting a counterion permittivity to a capacitance curve")

    system = ElectrochemicalSystem(
        cation=ion_database['Na+_hydrated'],
        anion=ion_database['F-_hydrated'],
        solvent=solvent_database['water'],
        concentration=1.0,
        temperature=298.15,
    )

    potentials = np.linspace(0.02, 1.0, 251)
    target_epsilon = 4.75
    counterion_index = 1  # Positive-potential sweep fits the anion response.

    target_model = StericModel(system, steric_model='cs')
    original_target_permittivity = target_model.ion_permitivities[counterion_index]
    target_model.ion_permitivities[counterion_index] = target_epsilon
    target_model.invalidate_caches()
    target_capacitance = np.asarray(
        target_model.analytical_capacitance_sweep_jit(potentials),
        dtype=float,
    )
    target_model.ion_permitivities[counterion_index] = original_target_permittivity
    target_model.invalidate_caches()

    fit_model = StericModel(system, steric_model='cs')
    fit_result = fit_counterion_permittivity_curve(
        fit_model,
        potentials,
        target_capacitance,
        use_jit_sweep=True,
        epsilon_bounds=(1.0, 10.0),
    )

    print(f"Target permittivity: {target_epsilon:.3f}")
    print(f"Recovered permittivity: {fit_result.fitted_permittivity:.3f}")
    print(f"Curve-fit RMSE: {fit_result.rmse:.3e} μF/cm²")
    print(f"Optimizer success: {fit_result.success}")

    plt.figure(figsize=(8, 6))
    plt.plot(potentials, target_capacitance, 'k-', linewidth=2, label='Synthetic target')
    plt.plot(potentials, fit_result.fitted_capacitance, 'r--', linewidth=2.5, label='Fitted curve')
    plt.xlabel('Potential (V)')
    plt.ylabel('Capacitance (μF/cm²)')
    plt.title('Counterion Permittivity Fit')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('permittivity_fit.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('Saved fitted-curve plot to permittivity_fit.png')


if __name__ == '__main__':
    main()