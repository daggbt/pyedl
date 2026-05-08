"""Sampling and plotting helpers for analytical observables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .cells import DoubleElectrodeCell
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


def _resolve_cell(cell=None, model=None):
    """Return a double-electrode cell from either a cell or a single-interface model."""
    if cell is not None and model is not None:
        raise ValueError("Provide either cell or model, not both.")
    if cell is not None:
        return cell
    if model is None:
        raise ValueError("Provide either cell or model.")
    return DoubleElectrodeCell(model)


def sample_cell_capacitance_curve(
    cell=None,
    model=None,
    voltage_range=(-1, 1),
    num_points=101,
):
    """Sample a double-electrode capacitance curve versus full-cell voltage."""
    resolved_cell = _resolve_cell(cell=cell, model=model)
    voltages = np.linspace(voltage_range[0], voltage_range[1], num_points)
    capacitance = []
    left_potential = []
    right_potential = []
    charge_density = []

    for voltage in voltages:
        split = resolved_cell.get_potential_split(float(voltage))
        capacitance.append(resolved_cell.analytical_capacitance(float(voltage)))
        left_potential.append(split.left_potential)
        right_potential.append(split.right_potential)
        charge_density.append(split.left_charge_density)

    return {
        'voltages': voltages,
        'capacitance': np.array(capacitance, dtype=float),
        'left_potential': np.array(left_potential, dtype=float),
        'right_potential': np.array(right_potential, dtype=float),
        'charge_density': np.array(charge_density, dtype=float),
    }


def sample_cell_energy_components(
    cell=None,
    model=None,
    voltage_range=(0.1, 1.0),
    num_points=50,
):
    """Sample double-electrode energy components versus full-cell voltage."""
    resolved_cell = _resolve_cell(cell=cell, model=model)
    voltages = np.linspace(voltage_range[0], voltage_range[1], num_points)
    entropic = []
    electrostatic = []
    steric = []
    total = []

    for voltage in voltages:
        components = resolved_cell.get_energy_components(float(voltage))
        entropic.append(components['entropic'])
        electrostatic.append(components['electrostatic'])
        steric.append(components['steric'])
        total.append(components['total'])

    return {
        'voltages': voltages,
        'entropic': np.array(entropic, dtype=float),
        'electrostatic': np.array(electrostatic, dtype=float),
        'steric': np.array(steric, dtype=float),
        'total': np.array(total, dtype=float),
    }


def sample_cell_profiles(
    cell=None,
    model=None,
    cell_voltage=1.0,
    separation_distance=None,
    distance_unit=None,
    num_points=200,
):
    """Sample full-cell double-electrode profiles at one cell voltage."""
    resolved_cell = _resolve_cell(cell=cell, model=model)
    return resolved_cell.sample_profiles(
        cell_voltage=cell_voltage,
        separation_distance=separation_distance,
        distance_unit=distance_unit,
        num_points=num_points,
    )


def plot_cell_capacitance_vs_voltage(
    cell=None,
    model=None,
    voltage_range=(-1, 1),
    num_points=101,
    ax=None,
    save_path=None,
    show_plot=False,
):
    """Plot double-electrode capacitance and return the figure, axis, and data."""
    data = sample_cell_capacitance_curve(
        cell=cell,
        model=model,
        voltage_range=voltage_range,
        num_points=num_points,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(data['voltages'], data['capacitance'], linewidth=2.5, label='Full cell')
    ax.set_xlabel('Cell voltage (V)')
    ax.set_ylabel('Differential capacitance (μF/cm²)')
    ax.set_title('Double-electrode capacitance vs cell voltage')
    ax.grid(True, alpha=0.3)
    ax.legend()

    _finalize_plot(fig, save_path=save_path, show_plot=show_plot)
    return fig, ax, data


def plot_cell_energy_components_vs_voltage(
    cell=None,
    model=None,
    voltage_range=(0.1, 1.0),
    num_points=50,
    ax=None,
    save_path=None,
    show_plot=False,
):
    """Plot full-cell energy components and return the figure, axis, and data."""
    data = sample_cell_energy_components(
        cell=cell,
        model=model,
        voltage_range=voltage_range,
        num_points=num_points,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(data['voltages'], data['entropic'], label='Entropic', marker='o')
    ax.plot(data['voltages'], data['electrostatic'], label='Electrostatic', marker='s')
    ax.plot(data['voltages'], data['steric'], label='Steric', marker='^')
    ax.plot(data['voltages'], data['total'], label='Total', marker='*', linewidth=2, color='black')
    ax.set_xlabel('Cell voltage (V)')
    ax.set_ylabel('Full-cell energy (J/m²)')
    ax.set_title('Double-electrode energy components vs cell voltage')
    ax.grid(True, alpha=0.3)
    ax.legend()

    _finalize_plot(fig, save_path=save_path, show_plot=show_plot)
    return fig, ax, data


def plot_cell_profiles(
    cell=None,
    model=None,
    cell_voltage=1.0,
    separation_distance=None,
    distance_unit=None,
    num_points=200,
    axes=None,
    save_path=None,
    show_plot=False,
):
    """Plot double-electrode profiles and return the figure, axes, and data."""
    data = sample_cell_profiles(
        cell=cell,
        model=model,
        cell_voltage=cell_voltage,
        separation_distance=separation_distance,
        distance_unit=distance_unit,
        num_points=num_points,
    )

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    else:
        fig = axes.flat[0].figure

    axes = np.asarray(axes)
    x_nm = data['x'] * 1e9
    left_h_nm = data['left_steric_layer_thickness'] * 1e9
    right_h_nm = (data['separation_distance'] - data['right_steric_layer_thickness']) * 1e9

    axes[0, 0].plot(x_nm, data['potential'], color='tab:blue')
    axes[0, 0].set_ylabel('Potential (V)')
    axes[0, 0].set_title('Electric potential')

    axes[0, 1].plot(x_nm, data['electric_field'], color='tab:orange')
    axes[0, 1].set_ylabel('Electric field (V/m)')
    axes[0, 1].set_title('Electric field')

    axes[1, 0].plot(x_nm, data['cation_concentration'], label='Cation', color='tab:red')
    axes[1, 0].plot(x_nm, data['anion_concentration'], label='Anion', color='tab:purple', linestyle='--')
    axes[1, 0].set_ylabel('Concentration (mol/L)')
    axes[1, 0].set_xlabel('Distance from left electrode (nm)')
    axes[1, 0].set_title('Ion concentrations')
    axes[1, 0].legend()

    axes[1, 1].plot(x_nm, data['volume_charge_density'], color='tab:green')
    axes[1, 1].set_ylabel('Charge density (C/m³)')
    axes[1, 1].set_xlabel('Distance from left electrode (nm)')
    axes[1, 1].set_title('Volume charge density')

    for axis in axes.flat:
        axis.axvline(left_h_nm, color='0.4', linestyle=':', linewidth=1)
        axis.axvline(right_h_nm, color='0.4', linestyle=':', linewidth=1)
        axis.grid(True, alpha=0.3)

    fig.suptitle(
        f"Double-electrode profiles at {cell_voltage:.2f} V "
        f"(left={data['left_potential']:.3f} V, right={data['right_potential']:.3f} V)"
    )
    fig.tight_layout()

    _finalize_plot(fig, save_path=save_path, show_plot=show_plot)
    return fig, axes, data