from pathlib import Path
import sys

import pytest

src_path = Path(__file__).resolve().parents[1] / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pyedl import ElectrochemicalSystem, StericModel, ion_database, solvent_database
from pyedl.materials import Ion, Solvent


@pytest.fixture
def naf_system():
    return ElectrochemicalSystem(
        cation=ion_database['Na+_hydrated'],
        anion=ion_database['F-_hydrated'],
        solvent=solvent_database['water'],
        concentration=3.89,
        temperature=298.15,
        n_hydration_cation=3.5,
        n_hydration_anion=2.7,
    )


@pytest.fixture
def naf_model(naf_system):
    return StericModel(naf_system, steric_model='cs')


@pytest.fixture
def naf_liu_model(naf_system):
    return StericModel(naf_system, steric_model='liu')


@pytest.fixture
def lipf6_system():
    pc_solvent = Solvent(name='Propylene Carbonate', dielectricConstant=66.14, solventPolarizability=6.0)
    li_ion = Ion(name='Li+', charge=1, radiusAng=2.82, dispersionB=0.0, ionPolarizability=0.03)
    pf6_ion = Ion(name='PF6-', charge=-1, radiusAng=2.54, dispersionB=0.0, ionPolarizability=4.0)
    return ElectrochemicalSystem(
        cation=li_ion,
        anion=pf6_ion,
        solvent=pc_solvent,
        concentration=1.0,
        temperature=298.15,
    )


@pytest.fixture
def lipf6_model(lipf6_system):
    return StericModel(lipf6_system, steric_model='cs')


@pytest.fixture
def il_system():
    return ElectrochemicalSystem(
        cation=ion_database['EMIM+'],
        anion=ion_database['TFSI-'],
        solvent=solvent_database['ionic_liquid'],
        concentration=3.89,
        temperature=298.15,
    )


@pytest.fixture
def il_model(il_system):
    return StericModel(il_system, steric_model='cs')