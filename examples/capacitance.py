#!/usr/bin/env python3
"""
Example script demonstrating the pyedl package.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

src_path = Path(__file__).resolve().parents[1] / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pyedl import ion_database, solvent_database
from pyedl import ElectrochemicalSystem, StericModel
from pyedl import save_capacitance_data
from pyedl.plotting import (
    plot_capacitance_vs_potential,
    plot_profiles_at_potential,
    sample_capacitance_curve,
)


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
    fig, _, naf_curve = plot_capacitance_vs_potential(
        system=system_naf,
        expt_cap=expt_cap,
        potential_range=(-1, 1),
        save_path='naf_capacitance.png',
        show_plot=False,
    )
    plt.close(fig)
    print(f"Saved naf_capacitance.png with {naf_curve['potentials'].size} sampled points")
    
    # Plot IL system
    fig, _, il_curve = plot_capacitance_vs_potential(
        system=system_il,
        potential_range=(-1, 1),
        save_path='il_capacitance.png',
        show_plot=False,
    )
    plt.close(fig)
    print(f"Saved il_capacitance.png with {il_curve['potentials'].size} sampled points")
    
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
    cs_curve = sample_capacitance_curve(model=model_naf, potential_range=(-1, 1), num_points=101)
    liu_curve = sample_capacitance_curve(model=model_liu, potential_range=(-1, 1), num_points=101)
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.plot(cs_curve['potentials'], cs_curve['capacitance'], 'b-', label='Carnahan-Starling')
    plt.plot(liu_curve['potentials'], liu_curve['capacitance'], 'r--', label='Liu')
    plt.xlabel('Potential (V)')
    plt.ylabel('Capacitance (μF/cm²)')
    plt.title('Comparison of Steric Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('steric_model_comparison.png', dpi=300)
    plt.close()

    print("\n" + "="*50 + "\n")

    # Example 5: Plot steric-layer profiles with the public API
    print("Example 5: Plotting steric-layer profiles")
    fig, _, profile_data = plot_profiles_at_potential(
        model=model_naf,
        potential=0.8,
        save_path='naf_profiles.png',
    )
    plt.close(fig)
    print(
        "Steric layer thickness at 0.8 V: "
        f"{profile_data['steric_layer_thickness'] * 1e9:.2f} nm"
    )
    
    print("Examples completed successfully!")


if __name__ == "__main__":
    main()