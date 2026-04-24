#!/usr/bin/env python3
"""
Example script demonstrating the free energy calculations.
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

src_path = Path(__file__).resolve().parents[1] / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pyedl import ion_database, solvent_database
from pyedl import ElectrochemicalSystem, StericModel
from pyedl.plotting import plot_energy_components_vs_potential, sample_energy_components

def main():
    print("Free Energy Calculation Example")
    print("===============================")

    # Define system: LiPF6 in Propylene Carbonate (PC)
    from pyedl.materials import Solvent, Ion
    
    # Custom Solvent: Propylene Carbonate
    pc_solvent = Solvent(name='Propylene Carbonate', dielectricConstant=66.14, solventPolarizability=6.0)
    
    li_hydrated = Ion(name='Li+_hydrated', charge=1, radiusAng=2.82, dispersionB=0.0, ionPolarizability=0.03)
    pf6 = Ion(name='PF6-', charge=-1, radiusAng=2.54, dispersionB=0.0, ionPolarizability=4.0)

    concentration = 1.0 # M
    
    system = ElectrochemicalSystem(
        cation=li_hydrated, 
        anion=pf6, 
        solvent=pc_solvent, 
        concentration=concentration
    )
    
    model = StericModel(system, steric_model='cs')
    
    print(f"System: {concentration} M LiPF6 in Propylene Carbonate")
    
    energy_data = sample_energy_components(
        model=model,
        potential_range=(0.1, 1.0),
        num_points=10,
    )
    results = pd.DataFrame(
        {
            'Potential': energy_data['potentials'],
            'Entropic': energy_data['entropic'],
            'Electrostatic': energy_data['electrostatic'],
            'Steric': energy_data['steric'],
            'Total': energy_data['total'],
        }
    )
    
    print(f"\n{'Potential (V)':<15} {'Entropic (J/m2)':<20} {'Electrostatic (J/m2)':<20} {'Steric (J/m2)':<20} {'Total (J/m2)':<20}")
    print("-" * 95)
    for row in results.itertuples(index=False):
        print(
            f"{row.Potential:<15.2f} {row.Entropic:<20.4e} {row.Electrostatic:<20.4e} "
            f"{row.Steric:<20.4e} {row.Total:<20.4e}"
        )
            
    # Save results to file
    output_file = 'energy_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    fig, _, _ = plot_energy_components_vs_potential(
        model=model,
        potential_range=(0.1, 1.0),
        num_points=10,
        save_path='energy_components.png',
    )
    plt.close(fig)
    print("Plot saved to energy_components.png")

if __name__ == "__main__":
    main()
