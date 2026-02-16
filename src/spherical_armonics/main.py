"""Plot spherical harmonics Y_l^m(theta, phi) for specified l and m."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import sph_harm_y


def plot_spherical_harmonic(l: int, m: int, real: bool = True, save: str | None = None) -> None:
    """
    Plot spherical harmonic Y_l^m on a sphere.
    Args:
        l: azimuthal quantum number
        m: magnetic quantum number (must satisfy |m| <= l)
        real: if True, plot Re(Y); if False, plot |Y|²
        save: optional path to save figure
    """
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")

    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)

    # sph_harm_y(n, m, theta, phi): theta=colatitude [0,π], phi=azimuth [0,2π]
    ylm = sph_harm_y(l, m, theta, phi)

    if real:
        z = np.real(ylm)
        cmap = "RdBu_r"
        vmax = np.abs(z).max()
        vmin = -vmax
    else:
        z = np.abs(ylm) ** 2
        cmap = "viridis"
        vmin, vmax = 0, z.max()

    # Radius deformed by harmonic value (positive = bulge, negative = indent)
    scale = 0.3 / (np.abs(z).max() + 1e-10)
    r = 1 + scale * z
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z_cart = r * np.cos(theta)
    colors = z

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        x, y, z_cart,
        facecolors=cm.get_cmap(cmap)(plt.Normalize(vmin, vmax)(colors)),
        rstride=1, cstride=1, antialiased=True,
    )
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    mappable = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    mappable.set_array(colors)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6)
    cbar.set_label(r"Re($Y_\ell^m$)" if real else r"$|Y_\ell^m|^2$")
    ax.set_title(rf"Spherical harmonic $Y_{{{l}}}^{{{m}}}$")
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    # plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot spherical harmonic Y_l^m")
    parser.add_argument("l", type=int, help="Azimuthal quantum number")
    parser.add_argument("m", type=int, help="Magnetic quantum number")
    parser.add_argument("--magnitude", action="store_true", help="Plot |Y|² instead of Re(Y)")
    parser.add_argument("--save", type=str, help="Path to save figure")
    args = parser.parse_args()

    plot_spherical_harmonic(args.l, args.m, real=not args.magnitude, save=args.save)


if __name__ == "__main__":
    main()
