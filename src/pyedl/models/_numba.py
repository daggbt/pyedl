"""Compiled sweep kernels for ordered potential evaluations."""

from __future__ import annotations

import numpy as np
import scipy.constants as sc
from numba import njit

MODEL_CS = 0
MODEL_LIU = 1

ELEMENTARY_CHARGE = float(sc.e)
AVOGADRO = float(sc.N_A)
VACUUM_PERMITTIVITY = float(sc.epsilon_0)


@njit(cache=True)
def clamp_phi(phi: float) -> float:
    if phi < 1e-6:
        return 1e-6
    if phi > 0.999:
        return 0.999
    return phi


@njit(cache=True)
def counterion_index(potential: float) -> int:
    return 0 if potential < 0.0 else 1


@njit(cache=True)
def counterion_charge(potential: float) -> float:
    return ELEMENTARY_CHARGE if potential < 0.0 else -ELEMENTARY_CHARGE


@njit(cache=True)
def steric_energy(phi: float, phi_bulk: float, kT: float, model_code: int) -> float:
    if model_code == MODEL_CS:
        surface = kT * phi * (8.0 - 9.0 * phi + 3.0 * phi * phi) / (1.0 - phi) ** 3
        bulk = kT * phi_bulk * (8.0 - 9.0 * phi_bulk + 3.0 * phi_bulk * phi_bulk) / (1.0 - phi_bulk) ** 3
        return surface - bulk

    surface = kT * (
        -5.0 * np.log(1.0 - phi) / 13.0
        - (
            phi
            * (
                phi
                * (phi * (13.0 * (5.0 - 3.0 * phi) * phi - 146.0) + 418.0)
                - 396.0
            )
        )
        / 52.0
        / (1.0 - phi) ** 3
    )
    bulk = kT * (
        -5.0 * np.log(1.0 - phi_bulk) / 13.0
        - (
            phi_bulk
            * (
                phi_bulk
                * (
                    phi_bulk * (13.0 * (5.0 - 3.0 * phi_bulk) * phi_bulk - 146.0)
                    + 418.0
                )
                - 396.0
            )
        )
        / 52.0
        / (1.0 - phi_bulk) ** 3
    )
    return surface - bulk


@njit(cache=True)
def zero_func_phi_numba(
    phi: float,
    potential: float,
    c_bulk: float,
    kT: float,
    phi_bulk: float,
    ion_volumes: np.ndarray,
    model_code: int,
) -> float:
    idx = counterion_index(potential)
    q = counterion_charge(potential)
    v = ion_volumes[idx]
    pot_excess = q * potential + steric_energy(phi, phi_bulk, kT, model_code)
    volfrac_ion = 1000.0 * AVOGADRO * v * c_bulk * np.exp(-pot_excess / kT)
    return phi - volfrac_ion


@njit(cache=True)
def fixed_point_phi_numba(
    phi_initial: float,
    potential: float,
    c_bulk: float,
    kT: float,
    phi_bulk: float,
    ion_volumes: np.ndarray,
    model_code: int,
    max_iter: int,
    tol: float,
) -> float:
    idx = counterion_index(potential)
    q = counterion_charge(potential)
    v = ion_volumes[idx]
    phi = clamp_phi(phi_initial)

    for _ in range(max_iter):
        pot_excess = q * potential + steric_energy(phi, phi_bulk, kT, model_code)
        updated = 1000.0 * AVOGADRO * v * c_bulk * np.exp(-pot_excess / kT)
        updated = clamp_phi(updated)
        if abs(updated - phi) < tol:
            return updated
        phi = 0.5 * phi + 0.5 * updated

    return phi


@njit(cache=True)
def bracketed_phi_bisection_numba(
    potential: float,
    c_bulk: float,
    kT: float,
    phi_bulk: float,
    ion_volumes: np.ndarray,
    model_code: int,
    max_iter: int,
    tol: float,
) -> float:
    low = 1e-10
    high = 1.0 - 1e-10
    f_low = zero_func_phi_numba(low, potential, c_bulk, kT, phi_bulk, ion_volumes, model_code)
    f_high = zero_func_phi_numba(high, potential, c_bulk, kT, phi_bulk, ion_volumes, model_code)

    if f_low * f_high > 0.0:
        previous_x = low
        previous_f = f_low
        found = False
        for step in range(1, 20):
            x = low + (high - low) * step / 19.0
            f_x = zero_func_phi_numba(x, potential, c_bulk, kT, phi_bulk, ion_volumes, model_code)
            if previous_f * f_x <= 0.0:
                low = previous_x
                high = x
                f_low = previous_f
                f_high = f_x
                found = True
                break
            previous_x = x
            previous_f = f_x

        if not found:
            return np.nan

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = zero_func_phi_numba(mid, potential, c_bulk, kT, phi_bulk, ion_volumes, model_code)
        if abs(f_mid) < tol or abs(high - low) < tol:
            return mid
        if f_low * f_mid <= 0.0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)


@njit(cache=True)
def steric_parameter_phi_sweep_numba(
    potentials: np.ndarray,
    c_bulk: float,
    kT: float,
    phi_bulk: float,
    ion_volumes: np.ndarray,
    model_code: int,
    fixed_point_max_iter: int = 30,
    fixed_point_tol: float = 1e-10,
    residual_tol: float = 1e-8,
    bisection_max_iter: int = 80,
    bisection_tol: float = 1e-12,
) -> np.ndarray:
    roots = np.empty(potentials.shape[0], dtype=np.float64)
    previous_root = phi_bulk

    for idx in range(potentials.shape[0]):
        potential = potentials[idx]
        candidate = fixed_point_phi_numba(
            previous_root,
            potential,
            c_bulk,
            kT,
            phi_bulk,
            ion_volumes,
            model_code,
            fixed_point_max_iter,
            fixed_point_tol,
        )
        residual = abs(
            zero_func_phi_numba(candidate, potential, c_bulk, kT, phi_bulk, ion_volumes, model_code)
        )

        if residual > residual_tol:
            candidate = bracketed_phi_bisection_numba(
                potential,
                c_bulk,
                kT,
                phi_bulk,
                ion_volumes,
                model_code,
                bisection_max_iter,
                bisection_tol,
            )
            if np.isnan(candidate):
                candidate = clamp_phi(previous_root)

        candidate = clamp_phi(candidate)
        roots[idx] = candidate
        previous_root = candidate

    return roots


@njit(cache=True)
def reduced_dielectric_from_volfrac_numba(epsilon_r: float, epsilon_i: float, volfrac_0: float) -> float:
    numerator = (1.0 + 2.0 * volfrac_0) * epsilon_i + 2.0 * (1.0 - volfrac_0) * epsilon_r
    denominator = (1.0 - volfrac_0) * epsilon_i + (2.0 + volfrac_0) * epsilon_r
    if denominator == 0.0:
        return epsilon_r
    return epsilon_r * numerator / denominator


@njit(cache=True)
def analytical_capacitance_sweep_numba(
    potentials: np.ndarray,
    phi_roots: np.ndarray,
    c_bulk: float,
    kT: float,
    epsilon_r: float,
    phi_bulk: float,
    ion_volumes: np.ndarray,
    ion_permittivities: np.ndarray,
) -> np.ndarray:
    capacitances = np.empty(potentials.shape[0], dtype=np.float64)
    beta = 1.0 / kT

    for idx in range(potentials.shape[0]):
        potential = potentials[idx]
        counterion_idx = counterion_index(potential)
        q = counterion_charge(potential)
        v = ion_volumes[counterion_idx]
        epsilon_i = ion_permittivities[counterion_idx]

        if abs(potential) < 1e-10:
            reduced_dielectric = reduced_dielectric_from_volfrac_numba(epsilon_r, epsilon_i, phi_bulk)
            reduced_dielectric_debye = reduced_dielectric_from_volfrac_numba(
                epsilon_r,
                epsilon_i,
                phi_roots[idx],
            )
            debye_length = np.sqrt(
                VACUUM_PERMITTIVITY * reduced_dielectric_debye * kT / (2.0 * c_bulk * ELEMENTARY_CHARGE ** 2 * 1000.0 * AVOGADRO)
            )
            capacitances[idx] = 100.0 * reduced_dielectric * VACUUM_PERMITTIVITY / debye_length
            continue

        phi_surface = phi_roots[idx]
        c_surface = phi_surface / v / 1000.0 / AVOGADRO
        reduced_dielectric = reduced_dielectric_from_volfrac_numba(epsilon_r, epsilon_i, phi_surface)
        epsilon_adjusted = VACUUM_PERMITTIVITY * reduced_dielectric
        steric_thickness = np.sqrt(
            -6.0 * potential * epsilon_adjusted / q / 1000.0 / AVOGADRO / (c_surface + 2.0 * c_bulk)
        )

        cap = beta * q * c_surface * potential
        cap /= phi_surface * (2.0 * phi_surface - 8.0) / (1.0 - phi_surface) ** 4 - 1.0
        cap *= (c_surface + 3.0 * c_bulk) / (c_surface + 2.0 * c_bulk) ** 2
        cap += (c_surface + c_bulk) / (c_surface + 2.0 * c_bulk)
        cap *= 3.0 * epsilon_adjusted / 2.0 / steric_thickness
        capacitances[idx] = 100.0 * cap

    return capacitances