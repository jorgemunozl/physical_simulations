"""Symbolic solutions and plots for linear second-order ODEs."""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def solution_1() -> sp.Eq:
    x = sp.symbols("x", real=True)
    y = sp.Function("y")
    ode = sp.Eq(y(x).diff(x, 2) + 7 * y(x).diff(x), sp.exp(-7 * x))
    return sp.dsolve(ode, y(x))


def solution_2() -> sp.Eq:
    x = sp.symbols("x", real=True)
    y = sp.Function("y")
    ode = sp.Eq(y(x).diff(x, 2) + 7 * y(x).diff(x), (x - 1) ** 2)
    return sp.dsolve(ode, y(x))


def _lambdify_general(sol_eq: sp.Eq, x_sym: sp.Symbol):
    y_expr = sol_eq.rhs
    consts = sorted(y_expr.free_symbols - {x_sym}, key=lambda s: s.name)
    fn = sp.lambdify([x_sym] + consts, y_expr, modules="numpy")
    return fn, consts


def plot_solutions(
    xlim: tuple[float, float] = (-0.5, 3.0),
    constant_pairs: list[tuple[float, float]] | None = None,
) -> None:
    """Plot sample integral curves for both ODE general solutions."""
    if constant_pairs is None:
        constant_pairs = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, -0.5),
        ]

    x_sym = sp.symbols("x", real=True)
    sol_a = solution_1()
    sol_b = solution_2()
    fn_a, _ = _lambdify_general(sol_a, x_sym)
    fn_b, _ = _lambdify_general(sol_b, x_sym)

    xs = np.linspace(xlim[0], xlim[1], 500)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for c1, c2 in constant_pairs:
        ya = fn_a(xs, c1, c2)
        yb = fn_b(xs, c1, c2)
        label = f"$C_1={c1}$, $C_2={c2}$"
        axes[0].plot(xs, ya, label=label)
        axes[1].plot(xs, yb, label=label)

    axes[0].set_title(r"$y'' + 7y' = e^{-7x}$")
    axes[1].set_title(r"$y'' + 7y' = (x-1)^2$")
    for ax in axes:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Solution 1:")
    sp.pprint(solution_1())
    print("Solution 2:")
    sp.pprint(solution_2())
    plot_solutions()
