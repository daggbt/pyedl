# pyedl: Semianalytical Models for Electric Double Layers

**pyedl** (formerly `pycapacitance`) is a Python package for modeling the physics of Electric Double Layers (EDL) in electrochemical systems. It implements semianalytical approximations for steric models (Carnahan-Starling and Liu) to efficiently calculate properties like electrode charge density, differential capacitance, and grand potential energies, particularly in regimes of high electrode potential and high electrolyte concentration where traditional dilute solution models fail.

This codebase accompanies the manuscript:

> **Semianalytical approximation of Ion Adsorption Layers and Capacitance in Carnahan–Starling-like steric models**  
> Dagmawi B. Tadesse and Drew F. Parsons  
> *Electrochimica Acta*, 531, 146266 (2025).  
> DOI: [10.1016/j.electacta.2025.146266](https://doi.org/10.1016/j.electacta.2025.146266)  
> URL: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0013468625006279)

## Features

*   **Steric Models**: Implements the Carnahan-Starling (CS) and Liu equations of state for hard-sphere fluids within mean-field theory.
*   **Semianalytical Approximation**: Uses a linear concentration profile approximation to solve the Poisson-Boltzmann equations analytically in the steric layer, providing rapid convergence to full numerical solutions at high potentials (>0.2 V) and concentrations (>1 M).
*   **EDL Properties**:
    *   **Charge Density**: Calculate electrode surface charge density ($\sigma$).
    *   **Differential Capacitance**: Calculate differential capacitance ($C_d$) with potential-dependent dielectric effects.
    *   **Profiles**: Generate concentration, electric field, and potential profiles within the double layer.
*   **Thermodynamics**:
    *   **Grand Potential Energy**: Calculate the total energy stored in the EDL, decomposed into:
        *   **Entropic Energy**: Ideal ion configuration entropy.
        *   **Electrostatic Energy**: Energy stored in the electric field.
        *   **Steric Energy**: Excess free energy due to finite ion size (excluded volume).
*   **Material Database**: Includes extensible databases for common ions (alkali metals, halides, ionic liquids) and solvents (water, organic solvents).

## Installation

The package requires Python 3.12+.

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/daggbt/pyedl.git
cd pyedl

# Run examples directly
uv run examples/capacitance.py
uv run examples/energy.py
```

### Using pip

```bash
pip install .
```

## Usage

### 1. Calculating Capacitance

```python
from pyedl import ion_database, solvent_database
from pyedl import ElectrochemicalSystem, StericModel

# Define the system: NaF in Water
system = ElectrochemicalSystem(
    cation=ion_database['Na+_hydrated'],
    anion=ion_database['F-_hydrated'],
    solvent=solvent_database['water'],
    concentration=1.0,  # mol/L
    temperature=298.15, # K
    n_hydration_cation=3.5,
    n_hydration_anion=2.7
)

# Initialize the model (Carnahan-Starling)
model = StericModel(system, steric_model='cs')

# Calculate capacitance at 1.0 V
potential = 1.0 # V
capacitance = model.analytical_capacitance(potential)
print(f"Capacitance at {potential}V: {capacitance:.2f} μF/cm²")
```

### 2. Free Energy Analysis

Calculate the components of the Grand Potential Energy stored in the EDL.

```python
from pyedl.materials import Ion, Solvent
from pyedl import ElectrochemicalSystem, StericModel

# Define custom materials (e.g., LiPF6 in Propylene Carbonate)
pc_solvent = Solvent(name='Propylene Carbonate', dielectricConstant=66.14, solventPolarizability=6.0)
li_ion = Ion(name='Li+', charge=1, radiusAng=2.82, dispersionB=0.0, ionPolarizability=0.03)
pf6_ion = Ion(name='PF6-', charge=-1, radiusAng=2.54, dispersionB=0.0, ionPolarizability=4.0)

system = ElectrochemicalSystem(
    cation=li_ion, 
    anion=pf6_ion, 
    solvent=pc_solvent, 
    concentration=1.0
)

model = StericModel(system, steric_model='cs')

# Calculate energy components at 1.0 V
phi = 1.0
entropic = model.get_entropic_energy(phi)
electrostatic = model.get_electrostatic_energy(phi)
steric = model.get_steric_free_energy(phi)
total = model.get_total_energy(phi)

print(f"Total Energy: {total:.4e} J/m²")
print(f"  - Entropic: {entropic:.4e} J/m²")
print(f"  - Electrostatic: {electrostatic:.4e} J/m²")
print(f"  - Steric: {steric:.4e} J/m²")
```

### 3. Plotting

The package includes utility functions for quick visualization.

```python
from pyedl import plot_capacitance_vs_potential

# Plot capacitance vs potential curve
plot_capacitance_vs_potential(
    system,
    potential_range=(-1.5, 1.5),
    save_path='capacitance_curve.png'
)
```

## Package Structure

*   `pyedl.models`: Core physics implementation.
    *   `StericModel`: Implementation of Carnahan-Starling and Liu models.
    *   `ElectrochemicalSystem`: Container for system properties.
*   `pyedl.materials`: Chemical property definitions.
    *   `Ion`: Properties like radius, charge, polarizability.
    *   `Solvent`: Dielectric constant, polarizability.
*   `pyedl.utils`: Helper functions for plotting and data export.

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{TADESSE2025146266,
title = {Semianalytical approximation of Ion Adsorption Layers and Capacitance in Carnahan–Starling-like steric models},
journal = {Electrochimica Acta},
volume = {531},
pages = {146266},
year = {2025},
issn = {0013-4686},
doi = {10.1016/j.electacta.2025.146266},
url = {https://www.sciencedirect.com/science/article/pii/S0013468625006279},
author = {Dagmawi B. Tadesse and Drew F. Parsons},
keywords = {Carnahan–Starling equation, Steric forces, Semianalytical Carnahan–Starling approximations, Electric double layers, Electric double layer capacitors},
abstract = {The Carnahan–Starling (CS) steric model is the best description of hard-sphere fluids within the mean-field theory. Here we introduce an approximation of the near-linear adsorption concentration profile of a counterion near an electrode for a CS model and derive the subsequent electric field and electrostatic potential profile in a double layer. This enables the derivation of a semianalytical approximation of the electrode charge density, differential capacitance, and total energies (grand potentials) of an electric double-layer capacitor. These semianalytical equations are valid for electrode potentials between 0.2–4 V and converge to the full numerical solutions of the CS model at high potentials of 1V and bulk concentration of 1M with relative errors less than 2% for the electrode charge densities, and less than 5% for the capacitance and total energies. We find the steric contribution comprises approximately one-quarter of the total energy at high electrode potentials, while the contribution from ideal ion entropies becomes insignificant. The model shows very good agreement with experimental measurements of an aqueous electrolyte, and good agreement at high potentials with computer simulations of an ionic liquid. These semianalytical approximations are effective for applications with concentrated solutions or ionic liquids at high applied voltages where the full numerical solution is computationally expensive or in some cases impossible.}
}
```

## License

MIT License
