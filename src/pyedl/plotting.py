"""Sampling and plotting helpers for analytical observables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .models import ElectrochemicalSystem, StericModel


def _resolve_model(model=None, system=None, steric_model='cs'):
    """Return a model instance from either a model or a system."""
    if model is not None and system is not None:
        raise ValueError("Provide either model or system, not both.")
    if model is None and system is None:
        raise ValueError("Provide either model or system.")
    if model is not None:
        return model
    return StericModel(system, steric_model=steric_model)


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
        except Exception as exc:
            print(f"Error at potential {pot}V: {exc}")
            capacitances.append(np.nan)

    return np.array(capacitances, dtype=float)


def _finalize_plot(fig, save_path=None, show_plot=False):
    """Apply optional save/show behavior without coupling rendering to computation."""
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()


def sample_capacitance_curve(
    model=None,
    system=None,
    potential_range=(-1, 1),
    num_points=101,
    steric_model='cs',
    use_jit_sweep=False,
):
    """Sample a capacitance curve without creating a figure."""
    resolved_model = _resolve_model(model=model, system=system, steric_model=steric_model)
    potentials = np.linspace(potential_range[0], potential_range[1], num_points)
    capacitance = _calculate_capacitance_sweep(
        resolved_model,
        potentials,
        use_jit_sweep=use_jit_sweep,
    )
    return {
        'potentials': potentials,
        'capacitance': capacitance,
    }


def sample_energy_components(
    model=None,
    system=None,
    potential_range=(0.1, 1.0),
    num_points=50,
    steric_model='cs',
):
    """Sample analytical grand-potential components without creating a figure."""
    resolved_model = _resolve_model(model=model, system=system, steric_model=steric_model)
    potentials = np.linspace(potential_range[0], potential_range[1], num_points)

    entropic = []
    electrostatic = []
    steric = []
    total = []
    for potential in potentials:
        try:
            entropic.append(resolved_model.get_entropic_energy(float(potential)))
            electrostatic.append(resolved_model.get_electrostatic_energy(float(potential)))
            steric.append(resolved_model.get_steric_free_energy(float(potential)))
            total.append(resolved_model.get_total_energy(float(potential)))
        except Exception as exc:
            print(f"Error at potential {potential}V: {exc}")
            entropic.append(np.nan)
            electrostatic.append(np.nan)
            steric.append(np.nan)
            total.append(np.nan)

    return {
        'potentials': potentials,
        'entropic': np.array(entropic, dtype=float),
        'electrostatic': np.array(electrostatic, dtype=float),
        'steric': np.array(steric, dtype=float),
        'total': np.array(total, dtype=float),
    }


def sample_profiles(
    model=None,
    system=None,
    potential=1.0,
    x_max=None,
    num_points=200,
    steric_model='cs',
):
    """Sample concentration, potential, and electric-field profiles at one electrode potential."""
    resolved_model = _resolve_model(model=model, system=system, steric_model=steric_model)
    steric_layer_thickness = resolved_model.get_steric_layer_thickness(potential)
    if x_max is None:
        x_max = 1.5 * steric_layer_thickness

    x = np.linspace(0.0, x_max, num_points)
    concentration = []
    electrostatic_potential = []
    electric_field = []
    for x_value in x:
        conc, _ = resolved_model.concentration_profile_in_steric_layer(float(x_value), potential)
        concentration.append(conc)
        electrostatic_potential.append(resolved_model.electrostatic_potential_in_steric_layer(float(x_value), potential))
        electric_field.append(resolved_model.electric_field_in_steric_layer(float(x_value), potential))

    return {
        'x': x,
        'concentration': np.array(concentration, dtype=float),
        'electrostatic_potential': np.array(electrostatic_potential, dtype=float),
        'electric_field': np.array(electric_field, dtype=float),
        'steric_layer_thickness': float(steric_layer_thickness),
        'potential': float(potential),
    }


def plot_capacitance_vs_potential(
    model=None,
    system=None,
    expt_cap=None,
    potential_range=(-1, 1),
    num_points=101,
    steric_model='cs',
    ax=None,
    save_path=None,
    show_plot=False,
    use_jit_sweep=False,
):
    """Plot an analytical capacitance curve and return the figure, axis, and sampled data."""
    resolved_model = _resolve_model(model=model, system=system, steric_model=steric_model)
    data = sample_capacitance_curve(
        model=resolved_model,
        potential_range=potential_range,
        num_points=num_points,
        use_jit_sweep=use_jit_sweep,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(data['potentials'], data['capacitance'], 'r-', linewidth=3, label='Model')
    if expt_cap is not None:
        ax.plot(expt_cap['pot'], expt_cap['cap'], '--k', label='Experimental')

    concentration = resolved_model.c_bulk
    ax.set_xlabel('Potential (V)')
    ax.set_ylabel('Capacitance (μF/cm²)')
    ax.set_title(f'Analytical Capacitance vs Potential\nConcentration: {concentration} M')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    _finalize_plot(fig, save_path=save_path, show_plot=show_plot)
    return fig, ax, data


def plot_energy_components_vs_potential(
    model=None,
    system=None,
    potential_range=(0.1, 1.0),
    num_points=50,
    steric_model='cs',
    ax=None,
    save_path=None,
    show_plot=False,
):
    """Plot analytical energy components and return the figure, axis, and sampled data."""
    resolved_model = _resolve_model(model=model, system=system, steric_model=steric_model)
    data = sample_energy_components(
        model=resolved_model,
        potential_range=potential_range,
        num_points=num_points,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(data['potentials'], data['entropic'], label='Entropic', marker='o')
    ax.plot(data['potentials'], data['electrostatic'], label='Electrostatic', marker='s')
    ax.plot(data['potentials'], data['steric'], label='Steric', marker='^')
    ax.plot(data['potentials'], data['total'], label='Total', marker='*', linewidth=2, color='black')

    concentration = resolved_model.c_bulk
    ax.set_xlabel('Potential (V)')
    ax.set_ylabel('Grand Potential Energy (J/m²)')
    ax.set_title(f'Grand Potential Components vs Electrode Potential\nConcentration: {concentration} M')
    ax.grid(True, alpha=0.3)
    ax.legend()

    _finalize_plot(fig, save_path=save_path, show_plot=show_plot)
    return fig, ax, data


def plot_profiles_at_potential(
    model=None,
    system=None,
    potential=1.0,
    x_max=None,
    num_points=200,
    steric_model='cs',
    axes=None,
    save_path=None,
    show_plot=False,
):
    """Plot sampled steric-layer profiles and return the figure, axes, and sampled data."""
    resolved_model = _resolve_model(model=model, system=system, steric_model=steric_model)
    data = sample_profiles(
        model=resolved_model,
        potential=potential,
        x_max=x_max,
        num_points=num_points,
    )

    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    else:
        fig = axes[0].figure

    x_nm = data['x'] * 1e9
    h_nm = data['steric_layer_thickness'] * 1e9

    axes[0].plot(x_nm, data['concentration'], color='tab:blue')
    axes[0].axvline(h_nm, color='0.4', linestyle='--', linewidth=1)
    axes[0].set_ylabel('Concentration (mol/L)')
    axes[0].set_title(f'Profiles at {potential:.2f} V')

    axes[1].plot(x_nm, data['electrostatic_potential'], color='tab:orange')
    axes[1].axvline(h_nm, color='0.4', linestyle='--', linewidth=1)
    axes[1].set_ylabel('Potential (V)')

    axes[2].plot(x_nm, data['electric_field'], color='tab:green')
    axes[2].axvline(h_nm, color='0.4', linestyle='--', linewidth=1)
    axes[2].set_ylabel('Electric field (V/m)')
    axes[2].set_xlabel('Distance from electrode (nm)')

    for axis in axes:
        axis.grid(True, alpha=0.3)

    _finalize_plot(fig, save_path=save_path, show_plot=show_plot)
    return fig, axes, data