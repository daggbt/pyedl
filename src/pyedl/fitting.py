"""Curve-fitting helpers for capacitance inversion workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .models import StericModel


@dataclass
class CounterionPermittivityFitResult:
    """Result returned by counterion permittivity curve fitting."""

    fitted_permittivity: float
    fitted_capacitance: np.ndarray
    residuals: np.ndarray
    rmse: float
    counterion_index: int
    use_jit_sweep: bool
    success: bool
    message: str
    optimizer_result: optimize.OptimizeResult


def _validate_curve_fit_inputs(potentials, target_capacitance):
    """Validate the sweep used for one-parameter counterion fitting."""
    potentials_array = np.asarray(potentials, dtype=float).reshape(-1)
    target_array = np.asarray(target_capacitance, dtype=float).reshape(-1)

    if potentials_array.size == 0:
        raise ValueError("potentials must contain at least one value.")
    if target_array.shape != potentials_array.shape:
        raise ValueError("target_capacitance must have the same shape as potentials.")
    if not np.all(np.isfinite(potentials_array)):
        raise ValueError("potentials must contain only finite values.")
    if not np.all(np.isfinite(target_array)):
        raise ValueError("target_capacitance must contain only finite values.")
    if np.any(np.abs(potentials_array) < 1e-12):
        raise ValueError("potentials must not contain zero when fitting one counterion permittivity.")
    if np.any(potentials_array > 0.0) and np.any(potentials_array < 0.0):
        raise ValueError("potentials must all have the same sign for one-counterion fitting.")

    counterion_index = 0 if np.all(potentials_array < 0.0) else 1
    return potentials_array, target_array, counterion_index


def _evaluate_capacitance_sweep(model: StericModel, potentials: np.ndarray, use_jit_sweep: bool) -> np.ndarray:
    """Evaluate a capacitance curve through the requested solver path."""
    if use_jit_sweep:
        return np.asarray(model.analytical_capacitance_sweep_jit(potentials), dtype=float)

    return np.array(
        [model.analytical_capacitance(float(potential)) for potential in potentials],
        dtype=float,
    )


def fit_counterion_permittivity_curve(
    model: StericModel,
    potentials,
    target_capacitance,
    epsilon_bounds=(0.1, 10.0),
    use_jit_sweep: bool = False,
    xatol: float = 1e-3,
    maxiter: int = 100,
) -> CounterionPermittivityFitResult:
    """Fit one counterion permittivity to a capacitance curve with a bounded scalar optimizer."""
    lower_bound, upper_bound = epsilon_bounds
    if lower_bound >= upper_bound:
        raise ValueError("epsilon_bounds must be an increasing (min, max) pair.")

    potentials_array, target_array, counterion_index = _validate_curve_fit_inputs(
        potentials,
        target_capacitance,
    )
    original_permittivity = model.ion_permitivities[counterion_index]

    def objective(epsilon_i: float) -> float:
        model.ion_permitivities[counterion_index] = float(epsilon_i)
        model.invalidate_caches()
        fitted_capacitance = _evaluate_capacitance_sweep(model, potentials_array, use_jit_sweep)
        residuals = fitted_capacitance - target_array
        return float(np.sqrt(np.mean(residuals ** 2)))

    try:
        optimizer_result = optimize.minimize_scalar(
            objective,
            bounds=(float(lower_bound), float(upper_bound)),
            method='bounded',
            options={'xatol': float(xatol), 'maxiter': int(maxiter)},
        )

        fitted_permittivity = float(optimizer_result.x)
        model.ion_permitivities[counterion_index] = fitted_permittivity
        model.invalidate_caches()
        fitted_capacitance = _evaluate_capacitance_sweep(model, potentials_array, use_jit_sweep)
        residuals = fitted_capacitance - target_array
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
    finally:
        model.ion_permitivities[counterion_index] = original_permittivity
        model.invalidate_caches()

    return CounterionPermittivityFitResult(
        fitted_permittivity=fitted_permittivity,
        fitted_capacitance=fitted_capacitance,
        residuals=residuals,
        rmse=rmse,
        counterion_index=counterion_index,
        use_jit_sweep=use_jit_sweep,
        success=bool(optimizer_result.success),
        message=str(optimizer_result.message),
        optimizer_result=optimizer_result,
    )