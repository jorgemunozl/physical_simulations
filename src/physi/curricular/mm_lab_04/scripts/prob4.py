"""Problema 4: Potencial vector magnetico en coordenadas esfericas."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad

A0 = 1.0
R_val = 2.0


def B_field_spherical(r, theta):
    """B en coordenadas esfericas: solo componente radial."""
    Br = 2 * A0 * np.cos(theta) / r**2
    return Br


def B_cartesian(x, z):
    """B en coordenadas cartesianas (plano xz, y=0)."""
    y = 0.0
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-12)  # evitar division por cero
    Br = 2 * A0 * z / r**3
    Bx = Br * x / r
    Bz = Br * z / r
    return Bx, Bz


flux_anal = np.pi * A0
print(f"Flujo magnetico analitico Φ_B = π·A₀ = {flux_anal:.12f}")


stokes_anal = np.pi * A0
print(f"∮ A·dr (Stokes) = π·A₀ = {stokes_anal:.12f}")
print(f"Diferencia = {abs(flux_anal - stokes_anal):.2e}")


# ─── (c) Verificacion numerica ────────────────────────────────────


def flux_numerical(n_th=200, n_ph=400):
    """Flujo magnetico numerico por cuadratura 2D."""

    def integrand(phi, theta):
        return A0 * np.sin(2 * theta)

    val, err = dblquad(
        integrand,
        0,
        np.pi / 4,  # theta
        lambda th: 0,
        lambda th: 2 * np.pi,  # phi
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return val, err


flux_num, flux_err = flux_numerical()
print(f"\nFlujo magnetico numerico = {flux_num:.12f}")
err_rel_flux = abs(flux_num - flux_anal) / max(1e-15, abs(flux_anal)) * 100
print(f"Error relativo = {err_rel_flux:.4f} %")


def stokes_numerical(n_pts=10000):
    """∮ A·dr numericamente sobre contorno θ=π/4."""
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    dt = 2 * np.pi / n_pts

    r_xy = R_val / np.sqrt(2)
    z0 = R_val / np.sqrt(2)

    x = r_xy * np.cos(t)
    y = r_xy * np.sin(t)
    z = np.full_like(t, z0)

    rho = np.sqrt(x**2 + y**2)
    A_phi = A0 * np.sin(np.pi / 4) / R_val  # constante en el borde

    Ax = A_phi * (-y / rho)
    Ay = A_phi * (x / rho)
    Az = np.zeros_like(t)

    # dr
    drx = -r_xy * np.sin(t) * dt
    dry = r_xy * np.cos(t) * dt
    drz = np.zeros_like(t) * dt

    return float(np.sum(Ax * drx + Ay * dry + Az * drz))


stokes_num = stokes_numerical()
print(f"∮ A·dr numerico = {stokes_num:.12f}")
print(
    f"Error relativo Stokes = {abs(stokes_num - stokes_anal) / max(1e-15, abs(stokes_anal)) * 100:.4f} %"
)


nx, nz = 30, 30
x_vals = np.linspace(-3, 3, nx)
z_vals = np.linspace(-3, 3, nz)
X, Z = np.meshgrid(x_vals, z_vals)
Bx, Bz = B_cartesian(X, Z)

# Magnitud para colorear
Bmag = np.sqrt(Bx**2 + Bz**2)

fig, ax = plt.subplots(figsize=(8, 6))
# Flechas
skip = 2
ax.quiver(
    X[::skip, ::skip],
    Z[::skip, ::skip],
    Bx[::skip, ::skip],
    Bz[::skip, ::skip],
    Bmag[::skip, ::skip],
    cmap="plasma",
    scale=8,
    width=0.003,
    pivot="mid",
)

# Dibujar el casquete esferico
theta_cap = np.linspace(0, np.pi / 4, 50)
x_cap = R_val * np.sin(theta_cap)
z_cap = R_val * np.cos(theta_cap)
ax.plot(x_cap, z_cap, "w--", linewidth=2, label=r"Casquete ($0\leq\theta\leq\pi/4$)")
ax.plot(-x_cap, z_cap, "w--", linewidth=2)

# Esfera completa
theta_full = np.linspace(0, np.pi, 100)
x_sph = R_val * np.sin(theta_full)
z_sph = R_val * np.cos(theta_full)
ax.plot(x_sph, z_sph, "gray", alpha=0.3, linewidth=1)
ax.plot(-x_sph, z_sph, "gray", alpha=0.3, linewidth=1)

ax.set_xlabel("$x$")
ax.set_ylabel("$z$")
ax.set_title(r"Campo magn\'etico $\vec{B}$ en el plano $xz$")
ax.set_aspect("equal")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.2)
plt.colorbar(ax.collections[0], ax=ax, label=r"$|\vec{B}|$", shrink=0.7)

out = Path(__file__).resolve().parent.parent / "images"
out.mkdir(parents=True, exist_ok=True)
path = out / "prob4_B_field.pdf"
fig.savefig(path, bbox_inches="tight", dpi=150)
print(f"\nGuardado: {path}")
plt.show()
