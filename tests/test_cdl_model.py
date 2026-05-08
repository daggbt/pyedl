import numpy as np
import pytest

from pyedl import CDLModel
from pyedl.plotting import sample_profiles


@pytest.fixture
def cdl_model(lipf6_system):
    return CDLModel(lipf6_system)


def test_cdl_threshold_and_steric_layer_onset(cdl_model):
    positive_threshold = cdl_model.get_threshold_potential(1.0)
    negative_threshold = cdl_model.get_threshold_potential(-1.0)

    assert positive_threshold > 0.0
    assert negative_threshold < 0.0
    assert cdl_model.get_steric_layer_thickness(0.5 * positive_threshold) == 0.0
    assert cdl_model.get_steric_layer_thickness(0.5 * negative_threshold) == 0.0
    assert cdl_model.get_steric_layer_thickness(2.0 * positive_threshold) > 0.0
    assert cdl_model.get_steric_layer_thickness(2.0 * negative_threshold) > 0.0
    assert cdl_model.get_steric_layer_thickness(3.0 * positive_threshold) > cdl_model.get_steric_layer_thickness(
        2.0 * positive_threshold
    )


def test_cdl_concentration_profile_caps_counterion_and_excludes_coion(cdl_model):
    potential = 2.0 * cdl_model.get_threshold_potential(1.0)
    H = cdl_model.get_steric_layer_thickness(potential)
    c_cap = cdl_model.get_counterion_steric_concentration(potential)

    assert H > 0.0
    assert cdl_model.counterion_concentration_profile(0.0, potential) == pytest.approx(c_cap)
    assert cdl_model.counterion_concentration_profile(0.5 * H, potential) == pytest.approx(c_cap)
    assert cdl_model.coion_concentration_profile(0.5 * H, potential) == 0.0

    concentrations = cdl_model.ion_concentrations(0.5 * H, potential)
    assert concentrations['anion'] == pytest.approx(c_cap)
    assert concentrations['cation'] == 0.0

    assert cdl_model.counterion_concentration_profile(1.001 * H, potential) == pytest.approx(c_cap, rel=0.02)


def test_cdl_potential_profile_matches_surface_and_tail(cdl_model):
    potential = 2.0 * cdl_model.get_threshold_potential(1.0)
    H = cdl_model.get_steric_layer_thickness(potential)
    threshold = cdl_model.get_threshold_potential(potential)

    assert cdl_model.electrostatic_potential_in_steric_layer(0.0, potential) == pytest.approx(potential)
    assert cdl_model.electrostatic_potential_in_steric_layer(H, potential) == pytest.approx(threshold)
    assert abs(cdl_model.electrostatic_potential_in_steric_layer(H + 10 * cdl_model.debye_length(), potential)) < abs(threshold)
    assert np.isfinite(cdl_model.electric_field_in_steric_layer(0.5 * H, potential))


def test_cdl_charge_density_and_capacitance_branches(cdl_model):
    assert cdl_model.charge_density(0.0) == 0.0
    assert cdl_model.analytical_capacitance(0.0) > 0.0

    positive_potential = 1.0
    negative_potential = -1.0
    capacitance, charge_density = cdl_model.get_capacitance(positive_potential)

    assert charge_density > 0.0
    assert cdl_model.charge_density(negative_potential) < 0.0
    assert capacitance == pytest.approx(cdl_model.analytical_capacitance(positive_potential))
    assert capacitance > 0.0


def test_cdl_high_potential_lipf6_regression(cdl_model):
    assert cdl_model.get_threshold_potential(1.0) == pytest.approx(0.0719543250503108, rel=1e-9)
    assert cdl_model.get_steric_layer_thickness(1.0) == pytest.approx(7.081732285503857e-10, rel=1e-9)
    assert cdl_model.charge_density(1.0) == pytest.approx(1.329585041847866, rel=1e-9)
    assert cdl_model.analytical_capacitance(1.0) == pytest.approx(69.9258811713863, rel=1e-9)


def test_cdl_energy_components_balance(cdl_model):
    potential = 1.0
    entropic = cdl_model.get_entropic_energy(potential)
    electrostatic = cdl_model.get_electrostatic_energy(potential)
    steric = cdl_model.get_steric_free_energy(potential)
    total = cdl_model.get_total_energy(potential)

    assert np.isfinite(entropic)
    assert np.isfinite(electrostatic)
    assert np.isfinite(steric)
    assert total == pytest.approx(entropic + electrostatic + steric, rel=1e-12, abs=1e-12)


def test_cdl_model_supports_profile_sampler(cdl_model):
    data = sample_profiles(model=cdl_model, potential=1.0, num_points=25)

    assert data['x'].shape == (25,)
    assert data['concentration'].shape == (25,)
    assert data['electrostatic_potential'].shape == (25,)
    assert data['electric_field'].shape == (25,)
    assert data['steric_layer_thickness'] > 0.0