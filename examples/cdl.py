#!/usr/bin/env python3
"""Example comparing the Composite Diffuse Layer approximation with CS."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

src_path = Path(__file__).resolve().parents[1] / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pyedl import CDLModel, ElectrochemicalSystem, StericModel
from pyedl.materials import Ion, Solvent
from pyedl.plotting import plot_profiles_at_potential


def main():
    print("Composite Diffuse Layer example")
    print("================================")

    pc_solvent = Solvent(name='Propylene Carbonate', dielectricConstant=66.14, solventPolarizability=6.0)
    li_ion = Ion(name='Li+', charge=1, radiusAng=2.82, dispersionB=0.0, ionPolarizability=0.03)
    pf6_ion = Ion(name='PF6-', charge=-1, radiusAng=2.54, dispersionB=0.0, ionPolarizability=4.0)

    system = ElectrochemicalSystem(
        cation=li_ion,
        anion=pf6_ion,
        solvent=pc_solvent,
        concentration=1.0,
        temperature=298.15,
    )

    cdl_model = CDLModel(system)
    cs_model = StericModel(system, steric_model='cs')

    potentials = np.linspace(0.02, 1.2, 160)
    cdl_capacitance = np.array([cdl_model.analytical_capacitance(float(phi)) for phi in potentials])
    cs_capacitance = np.array([cs_model.analytical_capacitance(float(phi)) for phi in potentials])

    threshold = cdl_model.get_threshold_potential(1.0)
    print(f"Positive-potential CDL steric threshold: {threshold:.4f} V")
    print(f"CDL steric-layer thickness at 1.0 V: {cdl_model.get_steric_layer_thickness(1.0) * 1e9:.3f} nm")
    print(f"CDL capacitance at 1.0 V: {cdl_model.analytical_capacitance(1.0):.2f} μF/cm²")
    print(f"CS capacitance at 1.0 V: {cs_model.analytical_capacitance(1.0):.2f} μF/cm²")
    print(f"CDL total free energy at 1.0 V: {cdl_model.get_total_energy(1.0):.4e} J/m²")

    plt.figure(figsize=(8, 6))
    plt.plot(potentials, cdl_capacitance, label='CDL approximation', linewidth=2.5)
    plt.plot(potentials, cs_capacitance, label='Carnahan-Starling', linewidth=2.5, linestyle='--')
    plt.axvline(threshold, color='0.4', linestyle=':', label='CDL steric onset')
    plt.xlabel('Potential (V)')
    plt.ylabel('Differential capacitance (μF/cm²)')
    plt.title('CDL vs Carnahan-Starling capacitance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('cdl_vs_cs_capacitance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved cdl_vs_cs_capacitance.png')

    fig, _, _ = plot_profiles_at_potential(
        model=cdl_model,
        potential=1.0,
        save_path='cdl_profiles.png',
    )
    plt.close(fig)
    print('Saved cdl_profiles.png')


if __name__ == '__main__':
    main()