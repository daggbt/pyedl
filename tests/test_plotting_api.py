import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyedl.plotting import (
    plot_capacitance_vs_potential,
    plot_energy_components_vs_potential,
    plot_profiles_at_potential,
    sample_capacitance_curve,
    sample_energy_components,
    sample_profiles,
)


def test_sample_capacitance_curve_matches_direct_model_calls(naf_model):
    data = sample_capacitance_curve(model=naf_model, potential_range=(-0.5, 0.5), num_points=11)

    direct = np.array([
        naf_model.analytical_capacitance(float(potential))
        for potential in data['potentials']
    ])

    assert list(data.keys()) == ['potentials', 'capacitance']
    assert data['potentials'].shape == (11,)
    assert data['capacitance'].shape == (11,)
    assert data['capacitance'] == pytest.approx(direct, rel=1e-12, abs=1e-12)


def test_sample_energy_components_has_expected_keys_and_balance(lipf6_model):
    data = sample_energy_components(model=lipf6_model, potential_range=(0.1, 1.0), num_points=10)

    assert set(data) == {'potentials', 'entropic', 'electrostatic', 'steric', 'total'}
    assert data['potentials'].shape == (10,)
    assert data['total'] == pytest.approx(
        data['entropic'] + data['electrostatic'] + data['steric'],
        rel=1e-12,
        abs=1e-12,
    )


def test_sample_profiles_returns_expected_arrays(naf_model):
    data = sample_profiles(model=naf_model, potential=0.8, num_points=25)

    assert set(data) == {
        'x',
        'concentration',
        'electrostatic_potential',
        'electric_field',
        'steric_layer_thickness',
        'potential',
    }
    assert data['x'].shape == (25,)
    assert data['concentration'].shape == (25,)
    assert data['electrostatic_potential'].shape == (25,)
    assert data['electric_field'].shape == (25,)
    assert data['steric_layer_thickness'] > 0.0


def test_plot_capacitance_vs_potential_returns_figure_axis_and_data(tmp_path, naf_system):
    output_path = tmp_path / 'capacitance.png'

    fig, ax, data = plot_capacitance_vs_potential(
        system=naf_system,
        potential_range=(-0.5, 0.5),
        num_points=11,
        save_path=output_path,
    )

    assert output_path.exists()
    assert len(ax.lines) == 1
    assert data['capacitance'].shape == (11,)
    plt.close(fig)


def test_plot_energy_components_vs_potential_returns_figure_axis_and_data(tmp_path, lipf6_system):
    output_path = tmp_path / 'energy.png'

    fig, ax, data = plot_energy_components_vs_potential(
        system=lipf6_system,
        potential_range=(0.1, 1.0),
        num_points=10,
        save_path=output_path,
    )

    assert output_path.exists()
    assert len(ax.lines) == 4
    assert data['total'].shape == (10,)
    plt.close(fig)


def test_plot_profiles_at_potential_returns_axes_and_data(tmp_path, naf_system):
    output_path = tmp_path / 'profiles.png'

    fig, axes, data = plot_profiles_at_potential(
        system=naf_system,
        potential=0.8,
        num_points=25,
        save_path=output_path,
    )

    assert output_path.exists()
    assert len(axes) == 3
    assert all(axis.lines for axis in axes)
    assert data['x'].shape == (25,)
    plt.close(fig)