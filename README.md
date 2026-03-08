# Physical

Small math and physics experiments, simulations, and visualizations.

This repository is not a single polished package. It is a playground for testing ideas: stochastic processes, numerical integration, optics, sampling, quantum-inspired numerics, and a few supporting math utilities.

## Topic Index

### Stochastic Processes

- `src/brownian/`
  Brownian motion experiments in 1D and 3D, plus scripts for generating animations.
- `src/mh/`
  Metropolis-Hastings sampling with a Gaussian target and a lightweight sampler implementation in PyTorch.

### Numerical Integration

- `src/integrals/line_integral.py`
  Numerical line integral along a parametric curve.
- `src/integrals/riemann.py`
  Riemann-sum and trapezoidal integration experiments, with derivative and cumulative-integral plots.
- `src/integrals/surface_integral.py`
  More structured numerical surface-integral code for scalar surface integrals and flux through parametric surfaces.

### Optics

- `src/optics/refraction.py`
  Refraction at a planar interface via Snell's law, including static plots and GIF animation.
- `src/optics/`
  Also contains rendered outputs such as `refraction.png` and `tir.gif`.

### Quantum / Electronic Structure

- `src/dft/dft.py`
  A finite-difference density-functional toy model for hydrogen-like radial states.

### Mathematical Physics

- `src/physical/spherical_armonics/main.py`
  Legendre and associated Legendre polynomial experiments connected to spherical harmonics.
- `src/physical/magnetic_field/main.py`
  Early magnetic-field utilities based on point or wire geometry.
- `src/experimental_magnetic_field/main.py`
  Simple plotting of experimental magnetic-field versus current data.

### Linear Algebra

- `src/descompositions/main.py`
  Gram-Schmidt orthonormalization, QR-style reconstruction, and reconstruction error checks.

### Neural / Approximation Experiments

- `src/kans/main.py`
  A scratch implementation of a Kolmogorov-Arnold Network style layer trained on a toy regression task.

### Visualization Utilities

- `src/matplotlib/main.py`
  Plotting helpers for heat-equation style comparisons, errors, scalar fields, and 3D views.

### Concurrency Notes

- `src/async/async.py`
  Small async vs threading comparison for I/O-bound tasks.

### Notebooks

- `src/a.ipynb`
  Notebook workspace for ad hoc exploration.

## Suggested Entry Points

If you want a quick sense of the repo, start here:

- `src/optics/refraction.py` for a clean, self-contained physics visualization.
- `src/integrals/surface_integral.py` for the most reusable numerical-analysis code.
- `src/brownian/main.py` for the simplest stochastic simulation.
- `src/dft/dft.py` for a compact quantum-inspired numerical experiment.

## Running Things

Most scripts are standalone and can be run directly from the repository root, for example:

```bash
python src/brownian/main.py
python src/integrals/riemann.py
python src/optics/refraction.py --show
python src/integrals/surface_integral.py --demo
```

Some experiments use extra scientific or ML dependencies such as `numpy`, `matplotlib`, `scipy`, and `torch`.

## Notes

- The codebase mixes quick sketches with more structured modules.
- Some folders contain generated media alongside source files.
- Naming is intentionally left close to the original experiments rather than normalized into a formal package.