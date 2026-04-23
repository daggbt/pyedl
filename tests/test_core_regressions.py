import pytest

from pyedl import StericModel
from pyedl.utils import find_ion_permittivity_from_capacitance


def test_zero_potential_capacitance_matches_analytical_path(naf_model):
    capacitance, charge_density = naf_model.get_capacitance(0.0)

    assert charge_density == 0.0
    assert capacitance > 0.0
    assert capacitance == pytest.approx(naf_model.analytical_capacitance(0.0), rel=1e-12, abs=1e-12)


def test_invalidate_caches_clears_state_and_changes_repeated_evaluation(naf_model):
    potential = 0.8
    base_capacitance = naf_model.analytical_capacitance(potential)

    assert naf_model.surface_volume_fraction_cache
    assert naf_model.reduced_dielectric_cache
    assert naf_model.steric_thickness_cache

    naf_model.ion_permitivities[1] *= 1.25
    naf_model.invalidate_caches()

    assert not naf_model.surface_volume_fraction_cache
    assert not naf_model.reduced_dielectric_cache
    assert not naf_model.steric_thickness_cache

    updated_capacitance = naf_model.analytical_capacitance(potential)
    assert updated_capacitance != pytest.approx(base_capacitance, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize(
    ('potential', 'counterion_index', 'target_permittivity'),
    [
        (0.8, 1, 4.75),
        (-0.8, 0, 3.25),
    ],
)
def test_fitting_helper_tracks_the_correct_counterion_branch(
    naf_system,
    potential,
    counterion_index,
    target_permittivity,
):
    target_model = StericModel(naf_system, steric_model='cs')
    target_model.ion_permitivities[counterion_index] = target_permittivity
    target_model.invalidate_caches()
    target_capacitance = target_model.analytical_capacitance(potential)

    fit_model = StericModel(naf_system, steric_model='cs')
    original_permittivities = tuple(fit_model.ion_permitivities)

    with pytest.warns(DeprecationWarning):
        fitted_permittivity, fitted_capacitance, fit_error = find_ion_permittivity_from_capacitance(
            fit_model,
            potential,
            target_capacitance,
            epsilon_min=1.0,
            epsilon_max=10.0,
            tolerance=1e-3,
            max_iterations=100,
        )

    assert fitted_permittivity == pytest.approx(target_permittivity, rel=5e-4, abs=5e-4)
    assert fitted_capacitance == pytest.approx(target_capacitance, rel=1e-5, abs=1e-4)
    assert fit_error < 1e-3
    assert tuple(fit_model.ion_permitivities) == pytest.approx(original_permittivities, rel=1e-12, abs=1e-12)