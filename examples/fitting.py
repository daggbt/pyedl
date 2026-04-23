#!/usr/bin/env python3
"""Example script demonstrating a simple capacitance-curve inversion."""

import time

import matplotlib.pyplot as plt
import numpy as np

from pyedl import ElectrochemicalSystem, StericModel
from pyedl import ion_database, solvent_database


def fit_counterion_permittivity(model, potentials, target_capacitance, epsilon_values, use_jit_sweep):
    """Fit one counterion permittivity against a capacitance curve."""
    potentials_array = np.asarray(potentials, dtype=float)
    target_array = np.asarray(target_capacitance, dtype=float)

    if np.any(potentials_array == 0.0):
        raise ValueError("Use a single-sign sweep that does not include zero potential.")

    counterion_index = 0 if np.all(potentials_array < 0.0) else 1
    original_permittivity = model.ion_permitivities[counterion_index]

    best_epsilon = None
    best_rmse = float("inf")
    best_capacitance = None

    try:
        for epsilon_i in epsilon_values:
            model.ion_permitivities[counterion_index] = float(epsilon_i)
            model.invalidate_caches()

            if use_jit_sweep:
                capacitance = np.asarray(model.analytical_capacitance_sweep_jit(potentials_array), dtype=float)
            else:
                capacitance = np.array(
                    [model.analytical_capacitance(float(pot)) for pot in potentials_array],
                    dtype=float,
                )

            rmse = float(np.sqrt(np.mean((capacitance - target_array) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_epsilon = float(epsilon_i)
                best_capacitance = capacitance.copy()
    finally:
        model.ion_permitivities[counterion_index] = original_permittivity
        model.invalidate_caches()

    return best_epsilon, best_rmse, best_capacitance


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
    epsilon_grid = np.linspace(1.0, 10.0, 40)
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

    warm_model = StericModel(system, steric_model='cs')
    warm_model.analytical_capacitance_sweep_jit(potentials)

    scalar_model = StericModel(system, steric_model='cs')
    start = time.perf_counter()
    scalar_epsilon, scalar_rmse, scalar_capacitance = fit_counterion_permittivity(
        scalar_model,
        potentials,
        target_capacitance,
        epsilon_grid,
        use_jit_sweep=False,
    )
    scalar_elapsed = time.perf_counter() - start

    jit_model = StericModel(system, steric_model='cs')
    start = time.perf_counter()
    jit_epsilon, jit_rmse, jit_capacitance = fit_counterion_permittivity(
        jit_model,
        potentials,
        target_capacitance,
        epsilon_grid,
        use_jit_sweep=True,
    )
    jit_elapsed = time.perf_counter() - start

    print(f"Target permittivity: {target_epsilon:.3f}")
    print(f"Scalar fit: epsilon={scalar_epsilon:.3f}, RMSE={scalar_rmse:.3e}, time={scalar_elapsed:.3f}s")
    print(f"JIT fit:    epsilon={jit_epsilon:.3f}, RMSE={jit_rmse:.3e}, time={jit_elapsed:.3f}s")
    print(f"Speedup: {scalar_elapsed / jit_elapsed:.2f}x")

    plt.figure(figsize=(8, 6))
    plt.plot(potentials, target_capacitance, 'k-', linewidth=2, label='Synthetic target')
    plt.plot(potentials, scalar_capacitance, 'b--', linewidth=2, label='Best scalar fit')
    plt.plot(potentials, jit_capacitance, 'r:', linewidth=2.5, label='Best JIT fit')
    plt.xlabel('Potential (V)')
    plt.ylabel('Capacitance (μF/cm²)')
    plt.title('Counterion Permittivity Fit')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('permittivity_fit.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('Saved comparison plot to permittivity_fit.png')


if __name__ == '__main__':
    main()