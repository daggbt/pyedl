#!/usr/bin/env python3
"""Example demonstrating double-electrode full-cell calculations."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

src_path = Path(__file__).resolve().parents[1] / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pyedl import CDLModel, DoubleElectrodeCell, ElectrochemicalSystem, StericModel
from pyedl.materials import Ion, Solvent
from pyedl.plotting import (
    plot_cell_capacitance_vs_voltage,
    plot_cell_energy_components_vs_voltage,
    plot_cell_profiles,
)


def main():
    print("Double-electrode full-cell example")
    print("===================================")

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

    cell_voltage = 1.0
    cdl_cell = DoubleElectrodeCell(CDLModel(system), split_method='auto')
    cs_cell = DoubleElectrodeCell(StericModel(system, steric_model='cs'), split_method='root')

    for name, cell in [('CDL', cdl_cell), ('Carnahan-Starling', cs_cell)]:
        split = cell.get_potential_split(cell_voltage)
        print(f"\n{name} full cell at {cell_voltage:.2f} V")
        print(f"  left electrode potential:  {split.left_potential:.4f} V")
        print(f"  right electrode potential: {split.right_potential:.4f} V")
        print(f"  charge residual: {split.charge_balance_residual:.3e} C/m²")
        print(f"  full-cell capacitance: {cell.analytical_capacitance(cell_voltage):.2f} μF/cm²")
        print(f"  total full-cell energy: {cell.get_total_energy(cell_voltage):.4e} J/m²")

    fig, _, _ = plot_cell_capacitance_vs_voltage(
        cell=cdl_cell,
        voltage_range=(0.05, 1.2),
        num_points=80,
        save_path='double_electrode_capacitance.png',
    )
    plt.close(fig)
    print("\nSaved double_electrode_capacitance.png")

    fig, _, _ = plot_cell_energy_components_vs_voltage(
        cell=cdl_cell,
        voltage_range=(0.05, 1.2),
        num_points=30,
        save_path='double_electrode_energy.png',
    )
    plt.close(fig)
    print("Saved double_electrode_energy.png")

    fig, _, _ = plot_cell_profiles(
        cell=cdl_cell,
        cell_voltage=cell_voltage,
        num_points=200,
        save_path='double_electrode_profiles.png',
    )
    plt.close(fig)
    print("Saved double_electrode_profiles.png")


if __name__ == '__main__':
    main()