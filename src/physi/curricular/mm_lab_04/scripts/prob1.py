"""Problema 1: Teorema de Stokes en paraboloide z = 4 - x^2 - y^2.

(a) Circulacion ∮ V·dr sobre la frontera (circulo x^2+y^2=4, z=0)
(b) Flujo de ∇×V a traves del paraboloide
(c) Verificacion numerica con error relativo
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad

# ─── (a) Circulacion analitica ───────────────────────────────────
# Frontera: r(t) = (2 cos t, 2 sin t, 0),  t ∈ [0, 2π]
# V = (y²+z², x²+z², x²+y²) → en frontera z=0: V = (y², x², x²+y²)
# V·dr = -8 sin³t + 8 cos³t  → ∮ = 0

circ_anal = 0.0
print(f"Circulacion analitica = {circ_anal}")


# ─── (b) Flujo del rotacional (analitico, via Stokes = 0) ────────
flux_anal = 0.0
print(f"Flujo analitico del rotacional = {flux_anal}")


# ─── (c) Verificacion numerica ────────────────────────────────────


def circulation_numerical(n_pts: int = 2000) -> float:
    """Aproxima ∮ V·dr discretizando el circulo frontera con N segmentos."""
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    dt = 2 * np.pi / n_pts

    x = 2 * np.cos(t)
    y = 2 * np.sin(t)
    z = np.zeros_like(t)

    # V en cada punto de la frontera  (z=0)
    Vx = y**2 + z**2
    Vy = x**2 + z**2
    Vz = x**2 + y**2

    # dr = (-2 sin t, 2 cos t, 0) dt
    drx = -2 * np.sin(t) * dt
    dry = 2 * np.cos(t) * dt
    drz = np.zeros_like(t) * dt

    return float(np.sum(Vx * drx + Vy * dry + Vz * drz))


def curl_flux_numerical(n_r: int = 100, n_th: int = 200) -> float:
    """Flujo de ∇×V a traves del paraboloide por cuadratura 2D."""
    # Parametrizacion: r(u, v) = (u cos v, u sin v, 4 - u²)
    # u ∈ [0, 2], v ∈ [0, 2π]
    # dS = (2u² cos v, 2u² sin v, u) du dv   (normal hacia arriba)

    def flux_integrand(v, u):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        x = u * np.cos(v)
        y = u * np.sin(v)
        z = 4 - u**2

        # ∇×V = (2y - 2z, 2z - 2x, 2x - 2y)
        curl_x = 2 * y - 2 * z
        curl_y = 2 * z - 2 * x
        curl_z = 2 * x - 2 * y

        # dS components
        dSx = 2 * u**2 * np.cos(v)
        dSy = 2 * u**2 * np.sin(v)
        dSz = u

        return curl_x * dSx + curl_y * dSy + curl_z * dSz

    flux, err = dblquad(
        flux_integrand,
        0,
        2,
        lambda u: 0,
        lambda u: 2 * np.pi,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    return flux


N_vals = [50, 100, 200, 500, 1000, 2000, 5000]
circ_errors = []
flux_num = curl_flux_numerical(150, 300)

print(f"\nFlujo numerico del rotacional = {flux_num:.12f}")
print(
    f"Error relativo flujo = {abs(flux_num - flux_anal) / max(1e-15, abs(flux_anal)):.2e}"
)
print()

print("Convergencia de la circulacion:")
for N in N_vals:
    cn = circulation_numerical(N)
    err = abs(cn - circ_anal)
    circ_errors.append(err)
    print(f"  N = {N:5d}  →  circulacion = {cn: .12f}  error = {err:.2e}")

# ─── Grafico de convergencia ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(N_vals, circ_errors, "o-", color="tab:blue", label=r"$\propto 1/N^2$")
# referencia 1/N^2
ref = np.array(circ_errors[0]) * (N_vals[0] / np.array(N_vals, dtype=float)) ** 2
ax.loglog(N_vals, ref, "--", color="gray", alpha=0.6, label=r"$O(1/N^2)$")
ax.set_xlabel("$N$ (puntos en la frontera)")
ax.set_ylabel("Error absoluto")
ax.set_title(r"Convergencia num\'erica de $\oint \vec{V} \cdot d\vec{r}$")
ax.grid(True, alpha=0.3)
ax.legend()

out = Path(__file__).resolve().parent.parent / "images"
out.mkdir(parents=True, exist_ok=True)
path = out / "prob1_convergence.pdf"
fig.savefig(path, bbox_inches="tight", dpi=150)
print(f"\nGuardado: {path}")
plt.show()
