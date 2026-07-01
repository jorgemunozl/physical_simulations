"""Problema 2: Campo electrostatico en coordenadas cilindricas.

E = (ρ₀/(ρ²+a²)) ρ̂ + z e^(-ρ) k̂
Volumen: cilindro radio R, altura H (z ∈ [0,H]).

(a) Divergencia e integral de volumen
(b) Flujo por las 3 superficies (Teorema de la Divergencia)
(c) Verificacion numerica para R=2, H=3, a=1, ρ₀=1
"""

from pathlib import Path

import numpy as np
from scipy.integrate import dblquad, tplquad

R_val = 2.0
H_val = 3.0
a_val = 1.0
rho0_val = 1.0


def div_E(rho, z, rho0=rho0_val, a=a_val):
    """Divergencia de E en coordenadas cilindricas."""
    term1 = rho0 * (a**2 - rho**2) / (rho * (rho**2 + a**2) ** 2)
    term2 = np.exp(-rho)
    return term1 + term2


def vol_integral_numerical(R=R_val, H=H_val, a=a_val, rho0=rho0_val):
    """Integral de volumen de ∇·E usando tplquad."""

    def integrand(z, rho, phi):
        d = div_E(rho, z, rho0, a)
        return d * rho  # elemento de volumen: ρ dρ dφ dz

    val, err = tplquad(
        integrand,
        0,
        2 * np.pi,  # phi
        lambda phi: 0,
        lambda phi: R,  # rho
        lambda rho, phi: 0,
        lambda rho, phi: H,  # z
        epsabs=1e-10,
        epsrel=1e-10,
    )
    return val, err


# ─── (b) Flujo por las 3 superficies ──────────────────────────────


def E_field(rho, z, rho0=rho0_val, a=a_val):
    """Componentes de E en coordenadas cilindricas."""
    Erho = rho0 / (rho**2 + a**2)
    Ez = z * np.exp(-rho)
    return Erho, Ez


def flux_top(R=R_val, H=H_val, a=a_val, rho0=rho0_val):
    """Flujo a traves de la tapa superior (z=H, normal +k̂)."""
    # dS = ρ dρ dφ k̂,  E·k̂ = E_z = H e^{-ρ}
    # Flujo = ∫₀²π ∫₀^R H e^{-ρ} ρ dρ dφ = 2π H ∫₀^R ρ e^{-ρ} dρ
    # ∫ ρ e^{-ρ} dρ = -(ρ+1)e^{-ρ}
    # = 2π H [-(R+1)e^{-R} + 1]
    return 2 * np.pi * H * (-(R + 1) * np.exp(-R) + 1)


def flux_bottom(R=R_val, a=a_val, rho0=rho0_val):
    """Flujo a traves de la tapa inferior (z=0, normal -k̂)."""
    # dS = ρ dρ dφ (-k̂),  E·(-k̂) = -E_z = -0·e^{-ρ} = 0
    # E_z = 0·e^{-ρ} = 0 at z=0
    return 0.0


def flux_lateral(R=R_val, H=H_val, a=a_val, rho0=rho0_val):
    """Flujo a traves de la superficie lateral (ρ=R, normal +ρ̂)."""
    # dS = R dφ dz ρ̂,  E·ρ̂ = E_ρ|_ρ=R = ρ₀/(R²+a²)
    # Flujo = ∫₀^H ∫₀²π ρ₀/(R²+a²) · R dφ dz = 2π R H ρ₀/(R²+a²)
    return 2 * np.pi * R * H * rho0 / (R**2 + a**2)


# ─── Verificacion numerica con cuadratura ─────────────────────────


def flux_top_numerical(R=R_val, H=H_val, a=a_val, rho0=rho0_val):
    def integrand(phi, rho):
        _, Ez = E_field(rho, H, rho0, a)
        return Ez * rho  # ρ dρ

    val, err = dblquad(
        integrand, 0, R, lambda r: 0, lambda r: 2 * np.pi, epsabs=1e-12, epsrel=1e-12
    )
    return val


def flux_lateral_numerical(R=R_val, H=H_val, a=a_val, rho0=rho0_val):
    def integrand(phi, z):
        Erho, _ = E_field(R, z, rho0, a)
        return Erho * R  # R dφ dz

    val, err = dblquad(
        integrand, 0, H, lambda z: 0, lambda z: 2 * np.pi, epsabs=1e-12, epsrel=1e-12
    )
    return val


# ─── Ejecutar ─────────────────────────────────────────────────────
print("=" * 60)
print("Problema 2: Campo electrostatico en coordenadas cilindricas")
print("=" * 60)

vol_num, vol_err = vol_integral_numerical()
print(f"\nIntegral de volumen de ∇·E (numerica) = {vol_num:.12f}")
print(f"  (error estimado de integracion = {vol_err:.2e})")

ft = flux_top()
fb = flux_bottom()
fl = flux_lateral()
flux_total = ft + fb + fl
print(f"\nFlujo analitico (tapa sup + tapa inf + lateral):")
print(f"  Tapa superior  (z=H):  {ft:.12f}")
print(f"  Tapa inferior  (z=0):  {fb:.12f}")
print(f"  Superf. lateral (ρ=R): {fl:.12f}")
print(f"  TOTAL                  {flux_total:.12f}")

ftn = flux_top_numerical()
fln = flux_lateral_numerical()
flux_num_total = ftn + fb + fln
print(f"\nFlujo numerico:")
print(f"  Tapa superior  (z=H):  {ftn:.12f}")
print(f"  Tapa inferior  (z=0):  {fb:.12f}")
print(f"  Superf. lateral (ρ=R): {fln:.12f}")
print(f"  TOTAL                  {flux_num_total:.12f}")

print(f"\nComparacion Divergencia vs Flujo:")
print(f"  ∫_V ∇·E dV   = {vol_num:.12f}")
print(f"  ∮_S E·dS      = {flux_num_total:.12f}")
print(f"  Diferencia    = {abs(vol_num - flux_num_total):.2e}")
rel_err = abs(vol_num - flux_num_total) / max(1e-15, abs(vol_num)) * 100
print(f"  Error relativo = {rel_err:.4f} %")
