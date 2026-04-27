# Physical simulations

Small math and physics experiments, simulations, and visualizations—mostly for curiosity and learning. This repository is a playground for stochastic processes, numerical integration, differential equations, optics, sampling, quantum-inspired numerics, and a few supporting utilities.

## Setup

Python **3.13+** (see `.python-version`). Dependencies are declared in `pyproject.toml` (NumPy, SciPy, SymPy, Matplotlib) and locked in `uv.lock`.

```bash
cd physical_simulations
uv sync
```

Some scripts additionally expect **PyTorch** (for example `src/metropolis_hasting/`, `src/kans/`, and `src/matplotlib/main.py`). Install it in the same environment when you need those modules.

## Topic index

### Stochastic processes

- `src/brownian/`  
  Brownian motion in 1D and 3D, plus scripts for generating animations.
- `src/metropolis_hasting/`  
  Metropolis–Hastings sampling with a Gaussian target and a small PyTorch-based sampler (`MH.py`).

### Numerical integration

- `src/integrals/line_integral.py`  
  Line integral along a parametric curve.
- `src/integrals/riemann.py`  
  Riemann and trapezoidal rules, with derivative and cumulative-integral plots.
- `src/integrals/surface_integral.py`  
  Scalar surface integrals and flux through parametric surfaces.

### Differential equations

- `src/differential_equations/`  
  Numerical ODE experiments and Wronskian-related code.
- `src/de_integration_me/`  
  2D vector-field examples integrated with naive stepping, finite differences, and RK4.
- `src/curricular/mathematics_methods_lab_01/`  
  Course lab material: symbolic DEs, Euler method, numerical comparison, Wronskian script, LaTeX report (`main.tex`), and supporting assets.

### Optics

- `src/optics/refraction.py`  
  Refraction at a planar interface (Snell’s law), static plots and GIF animation.  
  Rendered outputs live under `src/optics/` (for example `refraction.png`, `tir.gif`).

### Quantum / electronic structure

- `src/dft/dft.py`  
  Finite-difference density-functional toy model for hydrogen-like radial states.

### Classical / mathematical physics

- `src/spherical_armonics/main.py`  
  Legendre and associated Legendre experiments tied to spherical harmonics.
- `src/magnetic_field/main.py`  
  Magnetic-field sketches from simple wire or point geometry.
- `src/experimental_magnetic_field/main.py`  
  Plotting experimental magnetic-field versus current data.
- `src/vector_fields/vector_fields.py`  
  Small library of illustrative 2D fields and plots.

### Linear algebra and fitting

- `src/descompositions/main.py`  
  Gram–Schmidt orthonormalization, QR-style reconstruction, and error checks.
- `src/least_squares/least_squares.py`  
  Linear least squares via the normal equation.

### Neural / approximation experiments

- `src/kans/main.py`  
  Scratch Kolmogorov–Arnold style layer on a toy regression task (PyTorch).

### Visualization utilities

- `src/matplotlib/main.py`  
  Heat-equation-style comparisons, errors, scalar fields, and 3D views (uses PyTorch in places).

### Concurrency

- `src/async/async.py`  
  Small async versus threading comparison for I/O-bound work.

---

Most entry points are plain scripts: run them with `uv run python path/to/script.py` after `uv sync`. Have fun experimenting.
