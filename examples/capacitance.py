#!/usr/bin/env python3
"""
Example script demonstrating the pyedl package.
"""

import numpy as np
import pandas as pd
from pyedl import ion_database, solvent_database
from pyedl import ElectrochemicalSystem, StericModel
from pyedl import plot_capacitance_vs_potential, save_capacitance_data


def main():
    # Example 1: NaF in water with hydration
    print("Example 1: NaF in water with hydration effects")
    system_naf = ElectrochemicalSystem(
        cation=ion_database['Na+_hydrated'],
        anion=ion_database['F-_hydrated'],
        solvent=solvent_database['water'],
        concentration=3.89,  # mol/L
        temperature=298.15,  # K
        n_hydration_cation=3.5,
        n_hydration_anion=2.7
    )
    
    model_naf = StericModel(system_naf, steric_model='cs')
    
    # Calculate capacitance at specific potentials
    potentials = [0.0, 0.5, 1.0]
    for pot in potentials:
        cap = model_naf.analytical_capacitance(pot)
        print(f"Capacitance at {pot}V: {cap:.2f} μF/cm²")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Ionic liquid system
    print("Example 2: Ionic liquid system (EMIM+ TFSI-)")
    system_il = ElectrochemicalSystem(
        cation=ion_database['EMIM+'],
        anion=ion_database['TFSI-'],
        solvent=solvent_database['ionic_liquid'],
        concentration=3.89,  # mol/L
        temperature=298.15,  # K
        n_hydration_cation=0,  # No hydration in IL
        n_hydration_anion=0
    )
    
    model_il = StericModel(system_il, steric_model='cs')
    
    # Plot capacitance curves
    print("Plotting capacitance curves...")
    
    # Load experimental data if available
    try:
        expt_cap = pd.read_excel("expt_cap.xlsx")
    except:
        print("No experimental data file found")
        expt_cap = None
    
    # Plot NaF system
    potentials, capacitances = plot_capacitance_vs_potential(
        system_naf,
        expt_cap=expt_cap,
        potential_range=(-1, 1),
        save_path='naf_capacitance.png'
    )
    
    # Plot IL system
    plot_capacitance_vs_potential(
        system_il,
        potential_range=(-1, 1),
        save_path='il_capacitance.png'
    )
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Save detailed capacitance data
    print("Example 3: Saving detailed capacitance data")
    potentials = np.linspace(-1, 1, 101)
    
    # Save NaF data
    save_capacitance_data(model_naf, potentials, 'naf_capacitance_data.dat')
    
    # Save IL data  
    save_capacitance_data(model_il, potentials, 'il_capacitance_data.dat')
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Study effect of different steric models
    print("Example 4: Comparing steric models")
    
    # Liu model
    model_liu = StericModel(system_naf, steric_model='liu')
    
    # Compare at different potentials
    test_potentials = np.linspace(-1, 1, 101)
    cs_caps = []
    liu_caps = []
    
    for pot in test_potentials:
        cs_caps.append(model_naf.analytical_capacitance(pot))
        liu_caps.append(model_liu.analytical_capacitance(pot))
    
    # Plot comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(test_potentials, cs_caps, 'b-', label='Carnahan-Starling')
    plt.plot(test_potentials, liu_caps, 'r--', label='Liu')
    plt.xlabel('Potential (V)')
    plt.ylabel('Capacitance (μF/cm²)')
    plt.title('Comparison of Steric Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('steric_model_comparison.png', dpi=300)
    plt.close()
    
    print("Examples completed successfully!")


if __name__ == "__main__":
    main()