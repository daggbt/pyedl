import numpy as np
import pytest

from pyedl import CDLModel, DoubleElectrodeCell, ElectrochemicalSystem, StericModel
from pyedl.materials import Ion, Solvent
from pyedl.plotting import (
    plot_cell_capacitance_vs_voltage,
    plot_cell_energy_components_vs_voltage,
    plot_cell_profiles,
    sample_cell_capacitance_curve,
    sample_cell_energy_components,
    sample_cell_profiles,
)


@pytest.fixture
def symmetric_system():
    cation = Ion(name='Cat+', charge=1, radiusAng=2.0, dispersionB=0.0, ionPolarizability=1.0)
    anion = Ion(name='An-', charge=-1, radiusAng=2.0, dispersionB=0.0, ionPolarizability=1.0)
    solvent = Solvent(name='Solvent', dielectricConstant=50.0, solventPolarizability=1.0)
    return ElectrochemicalSystem(cation=cation, anion=anion, solvent=solvent, concentration=1.0)


def test_symmetric_cell_splits_voltage_equally(symmetric_system):
    cell = DoubleElectrodeCell(CDLModel(symmetric_system), split_method='root')
    split = cell.get_potential_split(1.0)

    assert split.left_potential == pytest.approx(0.5, abs=1e-10)
    assert split.right_potential == pytest.approx(-0.5, abs=1e-10)
    assert split.charge_balance_residual == pytest.approx(0.0, abs=1e-10)


def test_cdl_analytical_and_root_splits_match_for_asymmetric_cell(lipf6_system):
    model = CDLModel(lipf6_system)
    analytical_cell = DoubleElectrodeCell(model, split_method='cdl_analytical')
    root_cell = DoubleElectrodeCell(model, split_method='root')

    analytical_split = analytical_cell.get_potential_split(1.0)
    root_split = root_cell.get_potential_split(1.0)

    assert analytical_split.left_potential == pytest.approx(root_split.left_potential, rel=1e-10, abs=1e-12)
    assert analytical_split.right_potential == pytest.approx(root_split.right_potential, rel=1e-10, abs=1e-12)
    assert analytical_split.left_potential != pytest.approx(0.5, abs=1e-3)
    assert analytical_split.charge_balance_residual == pytest.approx(0.0, abs=1e-10)
    assert analytical_split.method == 'cdl_analytical'


def test_cs_cell_split_uses_charge_neutrality_for_asymmetric_system(lipf6_system):
    cell = DoubleElectrodeCell(StericModel(lipf6_system), split_method='root')
    split = cell.get_potential_split(1.0)

    assert split.left_potential + abs(split.right_potential) == pytest.approx(1.0)
    assert split.left_potential != pytest.approx(0.5, abs=1e-3)
    assert split.charge_balance_residual == pytest.approx(0.0, abs=1e-8)


def test_full_cell_capacitance_is_series_combination(lipf6_system):
    model = CDLModel(lipf6_system)
    cell = DoubleElectrodeCell(model)
    split = cell.get_potential_split(1.0)

    left_capacitance = model.analytical_capacitance(split.left_potential)
    right_capacitance = model.analytical_capacitance(split.right_potential)
    expected = left_capacitance * right_capacitance / (left_capacitance + right_capacitance)

    capacitance, charge_density = cell.get_capacitance(1.0)
    assert capacitance == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert charge_density == pytest.approx(split.left_charge_density)


def test_full_cell_energy_components_balance(lipf6_system):
    cell = DoubleElectrodeCell(CDLModel(lipf6_system))
    components = cell.get_energy_components(1.0)

    assert set(components) == {'entropic', 'electrostatic', 'steric', 'total'}
    assert components['total'] == pytest.approx(
        components['entropic'] + components['electrostatic'] + components['steric'],
        rel=1e-12,
        abs=1e-12,
    )


def test_double_electrode_profile_sampling(lipf6_system):
    cell = DoubleElectrodeCell(CDLModel(lipf6_system), separation_distance=30.0)
    data = cell.sample_profiles(1.0, num_points=31)

    assert data['x'].shape == (31,)
    assert data['potential'].shape == (31,)
    assert data['electric_field'].shape == (31,)
    assert data['cation_concentration'].shape == (31,)
    assert data['anion_concentration'].shape == (31,)
    assert data['volume_charge_density'].shape == (31,)
    assert data['left_potential'] - data['right_potential'] == pytest.approx(1.0)
    assert data['potential'][0] == pytest.approx(data['left_potential'], rel=1e-6, abs=1e-6)
    assert data['potential'][-1] == pytest.approx(data['right_potential'], rel=1e-6, abs=1e-6)


def test_double_electrode_plotting_helpers(tmp_path, lipf6_system):
    cell = DoubleElectrodeCell(CDLModel(lipf6_system))

    capacitance_data = sample_cell_capacitance_curve(cell=cell, voltage_range=(0.1, 1.0), num_points=6)
    energy_data = sample_cell_energy_components(cell=cell, voltage_range=(0.1, 1.0), num_points=6)
    profile_data = sample_cell_profiles(cell=cell, cell_voltage=1.0, num_points=12)

    assert capacitance_data['voltages'].shape == (6,)
    assert energy_data['total'].shape == (6,)
    assert profile_data['x'].shape == (12,)

    fig, _, _ = plot_cell_capacitance_vs_voltage(
        cell=cell,
        voltage_range=(0.1, 1.0),
        num_points=6,
        save_path=tmp_path / 'cell-capacitance.png',
    )
    assert (tmp_path / 'cell-capacitance.png').exists()
    fig.clf()

    fig, _, _ = plot_cell_energy_components_vs_voltage(
        cell=cell,
        voltage_range=(0.1, 1.0),
        num_points=6,
        save_path=tmp_path / 'cell-energy.png',
    )
    assert (tmp_path / 'cell-energy.png').exists()
    fig.clf()

    fig, _, _ = plot_cell_profiles(
        cell=cell,
        cell_voltage=1.0,
        num_points=12,
        save_path=tmp_path / 'cell-profiles.png',
    )
    assert (tmp_path / 'cell-profiles.png').exists()
    fig.clf()