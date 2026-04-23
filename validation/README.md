# Validation Cases

This directory documents the named analytical validation cases used by the Phase 0 regression tests.

The goal is not to claim these are exhaustive physical benchmarks. The goal is to keep a small set of repeatable reference calculations so future API and model changes can be checked against stable outputs.

## Case 1: NaF in water with hydrated ions

Configuration:

- cation: `Na+_hydrated`
- anion: `F-_hydrated`
- solvent: `water`
- concentration: `3.89 mol/L`
- temperature: `298.15 K`
- hydration numbers: `3.5` for the cation, `2.7` for the anion
- models covered: Carnahan-Starling and Liu

Used for:

- capacitance regression checks on the main aqueous example
- one Liu-model numerical anchor

## Case 2: LiPF6 in propylene carbonate

Configuration:

- cation: custom `Li+`
- anion: custom `PF6-`
- solvent: custom propylene carbonate surrogate
- concentration: `1.0 mol/L`
- temperature: `298.15 K`
- model: Carnahan-Starling

Used for:

- energy-component stability checks in the free-energy example path

## Case 3: EMIM-TFSI ionic liquid

Configuration:

- cation: `EMIM+`
- anion: `TFSI-`
- solvent: `ionic_liquid`
- concentration: `3.89 mol/L`
- temperature: `298.15 K`
- model: Carnahan-Starling

Note:

This case is illustrative, not a paper-fitted benchmark. It is kept as a stable regression anchor for the ionic-liquid example path.