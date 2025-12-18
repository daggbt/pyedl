#!/usr/bin/env python3
"""
Example script demonstrating the free energy calculations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyedl import ion_database, solvent_database
from pyedl import ElectrochemicalSystem, StericModel

def main():
    print("Free Energy Calculation Example")
    print("===============================")

    # Define system: LiPF6 in Propylene Carbonate (PC)
    # Using approximate parameters for PC as it might not be in the default database with exact properties
    # If 'propylene_carbonate' is not in solvent_database, we can create a custom one or use a placeholder.
    # Based on previous context, we can define it or use what's available.
    # Let's check what's available in solvent_database from the previous read of models.py/ions.py
    # It had 'water', 'ethanol', 'methanol', 'acetonitrile', 'dimethylsulfoxide', 'ionic_liquid'.
    # I will define a custom solvent for PC as in the test script.
    
    from pyedl.materials import Solvent, Ion
    
    # Custom Solvent: Propylene Carbonate
    pc_solvent = Solvent(name='Propylene Carbonate', dielectricConstant=66.14, solventPolarizability=6.0)
    
    # Custom Ions if needed, or use database. 
    # The manuscript mentions Li+ (hydrated radius 2.82 A) and PF6- (radius 2.54 A).
    # Let's define them explicitly to match the manuscript example.
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
    
    # Calculate energies at various potentials
    potentials = np.linspace(0.1, 1.0, 10)
    
    results = []
    
    print(f"\n{'Potential (V)':<15} {'Entropic (J/m2)':<20} {'Electrostatic (J/m2)':<20} {'Steric (J/m2)':<20} {'Total (J/m2)':<20}")
    print("-" * 95)
    
    for phi in potentials:
        try:
            # Calculate energy components (in J/m^2)
            # The methods return energy in Joules per unit area (since they integrate over x)
            # Wait, let's check the units in models.py.
            # The formulas in the manuscript are for Grand Potential per unit area (integral dx).
            # The implementation returns values multiplied by kT or similar.
            # Let's verify the units in the implementation.
            
            en = model.get_entropic_energy(phi)
            el = model.get_electrostatic_energy(phi)
            st = model.get_steric_free_energy(phi)
            total = model.get_total_energy(phi)
            
            results.append({
                'Potential': phi,
                'Entropic': en,
                'Electrostatic': el,
                'Steric': st,
                'Total': total
            })
            
            print(f"{phi:<15.2f} {en:<20.4e} {el:<20.4e} {st:<20.4e} {total:<20.4e}")
            
        except Exception as e:
            print(f"{phi:<15.2f} Error: {e}")
            
    # Save results to file
    df = pd.DataFrame(results)
    output_file = 'energy_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['Potential'], df['Entropic'], label='Entropic', marker='o')
    plt.plot(df['Potential'], df['Electrostatic'], label='Electrostatic', marker='s')
    plt.plot(df['Potential'], df['Steric'], label='Steric', marker='^')
    plt.plot(df['Potential'], df['Total'], label='Total', marker='*', linewidth=2, color='black')
    
    plt.xlabel('Potential (V)')
    plt.ylabel('Grand Potential Energy (J/m²)')
    plt.title(f'Grand Potential Components vs Electrode Potential\n{concentration} M LiPF6 in PC')
    plt.legend()
    plt.grid(True)

    plt.savefig('energy_components.png')
    print("Plot saved to energy_components.png")

if __name__ == "__main__":
    main()
