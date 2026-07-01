"""Problema 3: Identidad integral ∫_S (da × ∇) × P = ∮_∂S dr × P."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R = 2.0  # radio de la esfera

# ─── Campo P ──────────────────────────────────────────────────────


def P_field(x, y, z):
    """P = (x²y, yz, z²x)."""
    return np.array([x**2 * y, y * z, z**2 * x])


def line_integral_arc_zx(n_pts=2000):
    """Arco en plano zx (y=0): de (0,0,R) a (R,0,0).
    r(t) = (R sin t, 0, R cos t), t ∈ [0, π/2]."""
    t = np.linspace(0, np.pi / 2, n_pts)
    x = R * np.sin(t)
    y = np.zeros_like(t)
    z = R * np.cos(t)
    Px, Py, Pz = P_field(x, y, z)

    # dr = (R cos t dt, 0, -R sin t dt)
    drx = R * np.cos(t)
    dry = np.zeros_like(t)
    drz = -R * np.sin(t)
    dt = (np.pi / 2) / (n_pts - 1)

    # dr × P
    cross_x = dry * Pz - drz * Py
    cross_y = drz * Px - drx * Pz
    cross_z = drx * Py - dry * Px

    Ix = np.sum(cross_x) * dt
    Iy = np.sum(cross_y) * dt
    Iz = np.sum(cross_z) * dt
    return np.array([Ix, Iy, Iz])


def line_integral_arc_xy(n_pts=2000):
    """Arco en plano xy (z=0): r(t) = (R cos t, R sin t, 0), t ∈ [0, π/2]."""
    t = np.linspace(0, np.pi / 2, n_pts)
    x = R * np.cos(t)
    y = R * np.sin(t)
    z = np.zeros_like(t)
    Px, Py, Pz = P_field(x, y, z)

    drx = -R * np.sin(t)
    dry = R * np.cos(t)
    drz = np.zeros_like(t)
    dt = (np.pi / 2) / (n_pts - 1)

    cross_x = dry * Pz - drz * Py
    cross_y = drz * Px - drx * Pz
    cross_z = drx * Py - dry * Px

    Ix = np.sum(cross_x) * dt
    Iy = np.sum(cross_y) * dt
    Iz = np.sum(cross_z) * dt
    return np.array([Ix, Iy, Iz])


def line_integral_arc_yz(n_pts=2000):
    """Arco en plano yz (x=0): r(t) = (0, R cos t, R sin t), t ∈ [0, π/2]."""
    t = np.linspace(0, np.pi / 2, n_pts)
    x = np.zeros_like(t)
    y = R * np.cos(t)
    z = R * np.sin(t)
    Px, Py, Pz = P_field(x, y, z)

    drx = np.zeros_like(t)
    dry = -R * np.sin(t)
    drz = R * np.cos(t)
    dt = (np.pi / 2) / (n_pts - 1)

    cross_x = dry * Pz - drz * Py
    cross_y = drz * Px - drx * Pz
    cross_z = drx * Py - dry * Px

    Ix = np.sum(cross_x) * dt
    Iy = np.sum(cross_y) * dt
    Iz = np.sum(cross_z) * dt
    return np.array([Ix, Iy, Iz])


def surface_integral_numerical(n_th=100, n_ph=100):
    """Integral de superficie ∫_S (da × ∇) × P.

    Identidad tensorial: ((da × ∇) × P)_i = da_i (∇·P) - da_k ∂_i P_k
    donde ∂_i P_k es la derivada de la componente k de P respecto a la coordenada i.
    """
    th = np.linspace(0, np.pi / 2, n_th)
    ph = np.linspace(0, np.pi / 2, n_ph)
    dth = th[1] - th[0]
    dph = ph[1] - ph[0]

    I = np.zeros(3)
    for i in range(n_th):
        for j in range(n_ph):
            theta = th[i]
            phi = ph[j]
            st, ct = np.sin(theta), np.cos(theta)
            sp, cp = np.sin(phi), np.cos(phi)

            x = R * st * cp
            y = R * st * sp
            z = R * ct

            da = -np.array([st * cp, st * sp, ct]) * (R**2 * st * dth * dph)

            dP = np.array(
                [
                    [2 * x * y, 0, z**2],  # ∂_x P0, ∂_x P1, ∂_x P2
                    [x**2, z, 0],  # ∂_y P0, ∂_y P1, ∂_y P2
                    [0, y, 2 * z * x],  # ∂_z P0, ∂_z P1, ∂_z P2
                ]
            )

            # ∇·P = 2xy + z + 2zx
            divP = 2 * x * y + z + 2 * z * x

            # ((da × ∇) × P)_i = da_i (∇·P) - da_k ∂_i P_k
            for idx in range(3):
                # term2 = sum_k da_k * ∂_i P_k
                term2 = da[0] * dP[idx, 0] + da[1] * dP[idx, 1] + da[2] * dP[idx, 2]
                I[idx] += da[idx] * divP - term2

    return I


# ─── (c) Convergencia ─────────────────────────────────────────────


def line_integral_total(N: int):
    """∮ dr × P con N puntos por arco."""
    return line_integral_arc_xy(N) + line_integral_arc_yz(N) + line_integral_arc_zx(N)


# ─── Ejecutar ─────────────────────────────────────────────────────
print("=" * 60)
print("Problema 3: Identidad integral con producto vectorial")
print("=" * 60)

# Linea con muchos puntos como referencia "analitica"
I_line_ref = line_integral_total(20000)
print(f"\nIntegral de linea (ref, N=20000):")
print(f"  {I_line_ref}")

# Superficie
I_surf = surface_integral_numerical(120, 120)
print(f"\nIntegral de superficie (n_th=n_ph=120):")
print(f"  {I_surf}")

diff = np.linalg.norm(I_surf - I_line_ref)
print(f"\n  ||I_surf - I_line|| = {diff:.6e}")

# Convergencia
print(f"\nConvergencia de la integral de linea:")
N_vals = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
errors = []
for N in N_vals:
    Il = line_integral_total(N)
    err = np.linalg.norm(Il - I_line_ref)
    errors.append(err)
    print(f"  N = {N:5d}  →  error = {err:.6e}")

# Grafico de convergencia
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(N_vals, errors, "o-", color="tab:green", label="Error numérico")
ref = np.array(errors[1]) * (N_vals[1] / np.array(N_vals, dtype=float)) ** 2
ax.loglog(N_vals, ref, "--", color="gray", alpha=0.6, label=r"$O(1/N^2)$")
ax.set_xlabel("$N$ (puntos por arco)")
ax.set_ylabel(r"$\|\mathrm{error}\|_2$")
ax.set_title(r"Convergencia num\'erica de $\oint_{\partial S} d\vec{r} \times \vec{P}$")
ax.grid(True, alpha=0.3)
ax.legend()

out = Path(__file__).resolve().parent.parent / "images"
out.mkdir(parents=True, exist_ok=True)
path = out / "prob3_convergence.pdf"
fig.savefig(path, bbox_inches="tight", dpi=150)
print(f"\nGuardado: {path}")
plt.show()
