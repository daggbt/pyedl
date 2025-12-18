"""
Solvent class and database for electrochemical calculations.
"""

import scipy.constants as sc
from dataclasses import dataclass
from typing import Dict


@dataclass
class Solvent:
    """Solvent class to store solvent properties."""
    name: str
    dielectricConstant: float  # Relative permittivity
    solventPolarizability: float  # Solvent polarizability in cubic Angstroms
    
    def get_permittivity(self) -> float:
        """Get absolute permittivity in F/m."""
        return self.dielectricConstant * sc.epsilon_0


# Solvent database  
solvent_database: Dict[str, Solvent] = {
    'water': Solvent(name='water', dielectricConstant=78.5, solventPolarizability=1.4255),
    'ethanol': Solvent(name='ethanol', dielectricConstant=24.6, solventPolarizability=5.13),
    'methanol': Solvent(name='methanol', dielectricConstant=32.7, solventPolarizability=3.29),
    'acetonitrile': Solvent(name='acetonitrile', dielectricConstant=36.6, solventPolarizability=4.48),
    'dimethylsulfoxide': Solvent(name='dimethylsulfoxide', dielectricConstant=46.7, solventPolarizability=8.13),
    'ionic_liquid': Solvent(name='ionic_liquid', dielectricConstant=1.0, solventPolarizability=0.0),
}
