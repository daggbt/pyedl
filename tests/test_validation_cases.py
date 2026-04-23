import pytest


def test_validation_case_naf_water_cs_capacitance(naf_model):
    assert naf_model.analytical_capacitance(0.5) == pytest.approx(101.02862598931277, rel=1e-6)
    assert naf_model.charge_density(0.5) == pytest.approx(0.8315315518942794, rel=1e-6)


def test_validation_case_naf_water_liu_capacitance(naf_liu_model):
    assert naf_liu_model.analytical_capacitance(0.5) == pytest.approx(101.00165329447988, rel=1e-6)


def test_validation_case_lipf6_pc_energy_components(lipf6_model):
    assert lipf6_model.get_entropic_energy(1.0) == pytest.approx(0.021766585463793175, rel=1e-6)
    assert lipf6_model.get_electrostatic_energy(1.0) == pytest.approx(0.19150296568454805, rel=1e-6)
    assert lipf6_model.get_steric_free_energy(1.0) == pytest.approx(0.523100288917692, rel=1e-6)
    assert lipf6_model.get_total_energy(1.0) == pytest.approx(0.7363698400660332, rel=1e-6)


def test_validation_case_emim_tfsi_illustrative_capacitance(il_model):
    assert il_model.analytical_capacitance(1.0) == pytest.approx(8.530413648596378, rel=1e-6)