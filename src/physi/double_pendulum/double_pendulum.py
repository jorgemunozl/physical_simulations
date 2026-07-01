from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

DOUBLE_PENDULUM_PATH = Path(__file__).parent

# Physical parameters for the double pendulum
M1 = 1.0
M2 = 1.0
L1 = 1.0
L2 = 1.0
GRAVITY = 9.81
TOTAL_MASS = M1 + M2


def d2theta1(theta_1, theta_2, omega_1, omega_2):
    diff = theta_1 - theta_2
    alpha = M1 + M2 * np.sin(diff) ** 2
    # θ₁'' = [m₂·g·sin(θ₂)·cos(Δ) - m₂·l₁·ω₁²·sin(Δ)·cos(Δ) - m₂·l₂·ω₂²·sin(Δ) - (m₁+m₂)·g·sin(θ₁)] / [l₁·(m₁ + m₂·sin²(Δ))]
    numerator = (
        M2 * GRAVITY * np.sin(theta_2) * np.cos(diff)
        - M2 * L1 * omega_1**2 * np.sin(diff) * np.cos(diff)
        - M2 * L2 * omega_2**2 * np.sin(diff)
        - TOTAL_MASS * GRAVITY * np.sin(theta_1)
    )
    return numerator / (alpha * L1)


def d2theta2(theta_1, theta_2, omega_1, omega_2):
    diff = theta_1 - theta_2
    alpha = M1 + M2 * np.sin(diff) ** 2
    first_term = np.sin(diff) * (
        TOTAL_MASS * L1 * omega_1**2 + M2 * L2 * omega_2**2 * np.cos(diff)
    )
    second_term = GRAVITY * (
        TOTAL_MASS * np.sin(theta_1) * np.cos(diff) - TOTAL_MASS * np.sin(theta_2)
    )
    return (first_term + second_term) / (alpha * L2)


def rk4_step(f, t, u, h):
    k1 = h * f(t, u, d2theta1, d2theta2)
    k2 = h * f(t + h / 2, u + k1 / 2, d2theta1, d2theta2)
    k3 = h * f(t + h / 2, u + k2 / 2, d2theta1, d2theta2)
    k4 = h * f(t + h, u + k3, d2theta1, d2theta2)
    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def adaptive_rk4_step(f, t, u, h):
    """
    Adaptive RK4 (RKF45) step using the Runge-Kutta-Fehlberg method.

    Computes both a 4th-order and a 5th-order solution from the same
    set of evaluations.  The difference gives an estimate of the local
    truncation error, which can be used for step-size control.

    Parameters
    ----------
    f : callable
        RHS function with signature f(t, u, d2theta1, d2theta2).
    t : float
        Current time.
    u : ndarray
        Current state vector.
    h : float
        Step size.

    Returns
    -------
    u5 : ndarray
        5th-order accurate solution at t + h.
    error : ndarray
        Element-wise estimated error (|u5 - u4|).
    """
    # RKF45 Butcher tableau                          # order
    #                                                # 4th    5th
    # 0      |
    # 1/4    | 1/4
    # 3/8    | 3/32      9/32
    # 12/13  | 1932/2197  -7200/2197  7296/2197
    # 1      | 439/216    -8          3680/513    -845/4104
    # 1/2    | -8/27      2           -3544/2565  1859/4104  -11/40
    # -----------------------------------------------------------
    #        | 25/216     0           1408/2565   2197/4104  -1/5      0
    #        | 16/135     0           6656/12825  28561/56430 -9/50    2/55

    # Stage coefficients
    a2, a3, a4, a5, a6 = 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2

    b21 = 1 / 4
    b31, b32 = 3 / 32, 9 / 32
    b41, b42, b43 = 1932 / 2197, -7200 / 2197, 7296 / 2197
    b51, b52, b53, b54 = 439 / 216, -8.0, 3680 / 513, -845 / 4104
    b61, b62, b63, b64, b65 = -8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40

    # 4th-order coefficients
    c1, c3, c4, c5 = 25 / 216, 1408 / 2565, 2197 / 4104, -1 / 5
    # 5th-order coefficients
    d1, d3, d4, d5, d6 = 16 / 135, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55

    k1 = h * f(t, u, d2theta1, d2theta2)
    k2 = h * f(t + a2 * h, u + b21 * k1, d2theta1, d2theta2)
    k3 = h * f(t + a3 * h, u + b31 * k1 + b32 * k2, d2theta1, d2theta2)
    k4 = h * f(t + a4 * h, u + b41 * k1 + b42 * k2 + b43 * k3, d2theta1, d2theta2)
    k5 = h * f(
        t + a5 * h,
        u + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4,
        d2theta1,
        d2theta2,
    )
    k6 = h * f(
        t + a6 * h,
        u + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5,
        d2theta1,
        d2theta2,
    )

    # 4th-order approximation (used for error estimation)
    u4 = u + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
    # 5th-order approximation (accepted solution)
    u5 = u + d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6

    error = np.abs(u5 - u4)
    return u5, error


# Vector State for the double pendulum
def F(t, u, second_theta_1, second_theta_2):
    # u = [theta_1, omega_1, theta_2, omega_2]
    theta_1, omega_1, theta_2, omega_2 = u

    d_omega1 = second_theta_1(theta_1, theta_2, omega_1, omega_2)
    d_omega2 = second_theta_2(theta_1, theta_2, omega_1, omega_2)

    return np.array([omega_1, d_omega1, omega_2, d_omega2])


def calc_kinetic_energy(m1, l1, omega_1, m2, l2, omega_2, diff) -> np.ndarray:
    """
    KE of the double pendulum.
    KE = 1/2 m1 l1^2 w1^2  +  1/2 m2 (l1^2 w1^2 + l2^2 w2^2 + 2 l1 l2 w1 w2 cos(dθ))
    """
    return 0.5 * m1 * l1**2 * omega_1**2 + 0.5 * m2 * (
        l1**2 * omega_1**2
        + l2**2 * omega_2**2
        + 2 * l1 * l2 * omega_1 * omega_2 * np.cos(diff)
    )


def calc_potential_energy(m1, l1, m2, l2, theta_1, theta_2) -> np.ndarray:
    """PE = -(m1+m2) g l1 cos theta1  -  m2 g l2 cos theta2"""
    return -(m1 + m2) * GRAVITY * l1 * np.cos(theta_1) - m2 * GRAVITY * l2 * np.cos(
        theta_2
    )


def rhs(t, u):
    """Right-hand side with the signature required by solve_ivp."""
    return F(t, u, d2theta1, d2theta2)


@dataclass
class DoublePendulumSolver:
    """
    present to dont change every time
    """

    time: float = field(
        default=10.0,
        metadata={"description": "Total simulation time in seconds"},
    )
    preset: str = field(
        default="hard",
        metadata={"description": "Preset to use for simulation"},
    )
    steps: int = field(
        default=int(10e2),
        metadata={"description": "Number of simulation steps"},
    )
    theta_1: float = field(
        default=np.pi / 2,
        metadata={"description": "Initial angle of the first pendulum"},
    )
    theta_2: float = field(
        default=np.pi / 2,
        metadata={"description": "Initial angle of the second pendulum"},
    )
    omega_1: float = field(
        default=1.0,
        metadata={"description": "Initial angular velocity of the first pendulum"},
    )
    omega_2: float = field(
        default=-1.0,
        metadata={"description": "Initial angular velocity of the second pendulum"},
    )
    time_str: str = field(
        default="10s",
        metadata={"description": "Total simulation time as a string"},
    )
    steps_str: str = field(
        default="10e2",
        metadata={"description": "Number of simulation steps as a string"},
    )
    path_numpy: Path = field(
        default=Path(""),
        metadata={"description": "Path to save the simulation results as a NumPy file"},
    )
    path_animation: Path = field(
        default=Path(""),
        metadata={"description": "Path to save the simulation results as a plot"},
    )
    method: Literal["rk4", "rk8", "rk4_adaptive"] = field(
        default="rk4",
        metadata={"description": "Solver method to use"},
    )

    def __post_init__(self):
        if self.preset:
            self.set_preset(self.preset)
        self._build_paths()

    def _build_paths(self):
        # map pi to symbol
        pi = "pi_2"
        prefix = f"{self.time_str}_{self.steps_str}_{pi}_{pi}_{self.omega_1}_{self.omega_2}_{self.method}"
        self.path_numpy = DOUBLE_PENDULUM_PATH / f"data/{prefix}.npy"
        self.path_animation = DOUBLE_PENDULUM_PATH / f"data/{prefix}.png"

    def solve(self):
        if self.preset:
            self.set_preset(self.preset)
            self._build_paths()

        if self.path_numpy.exists():
            print("Solution already exists!!", self.path_numpy)
        else:
            print("Computing solution on", self.path_numpy)
            if self.method == "rk4":
                self.solve_rk4()
            elif self.method == "rk8":
                self.solve_rk8()
            elif self.method == "rk4_adaptive":
                self.solve_adaptive_rk4()
            else:
                raise ValueError(f"Unknown method: {self.method}")

    def set_preset(self, preset: str):
        if preset == "testing":
            self.time = 10
            self.steps = 1000
            self.steps_str = "10e3"
            self.time_str = "10s"
        elif preset == "hard":
            self.time = 40
            self.steps = 100_000
            self.steps_str = "10e5"
            self.time_str = "40s"
        elif preset == "extreme":
            self.time = 100
            self.steps = 10_000_000
            self.steps_str = "10e6"
            self.time_str = "100s"
        else:
            raise ValueError(f"Unknown preset: {preset}")

    def build_times(self):
        return (0, self.time), np.linspace(0, self.time, self.steps)

    def build_initial_conditions(self):
        return np.array([self.theta_1, self.omega_1, self.theta_2, self.omega_2])

    def save_solution(self, u):
        np.save(self.path_numpy, u)

    def solve_rk4(self, dt=None):
        if dt is None:
            dt = self.time / self.steps
        u_0 = self.build_initial_conditions()
        u = np.zeros((self.steps + 1, 4))
        u[0] = u_0
        for i in range(self.steps):
            u[i + 1] = rk4_step(F, i * dt, u[i], dt)
        # u = [theta_1, omega_1, theta_2, omega_2]
        self.save_solution(u)
        print(f"Saved solution to {self.path_numpy}")

    def solve_rk8(self):
        from scipy.integrate import solve_ivp

        u_0 = self.build_initial_conditions()
        t_span, t_eval = self.build_times()
        sol = solve_ivp(
            rhs,
            t_span,
            u_0,
            method="DOP853",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-9,
        )
        self.save_solution(sol.y.transpose())
        print(f"Saved solution to {self.path_numpy}")

    def solve_adaptive_rk4(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        h0: float | None = None,
        h_min: float = 1e-12,
        h_max: float | None = None,
        max_steps: int = 1_000_000,
    ):
        """
        Solve the double pendulum using adaptive RK4 (RKF45).

        The step size is automatically adjusted so that the estimated local
        error stays within the prescribed tolerances.  At the end the
        solution is interpolated back onto the uniform time grid
        ``np.linspace(0, self.time, self.steps + 1)`` for full compatibility
        with the rest of the class.

        Parameters
        ----------
        rtol : float
            Relative tolerance for error control.
        atol : float
            Absolute tolerance for error control.
        h0 : float or None
            Initial step size (defaults to ``time / steps``).
        h_min : float
            Minimum allowed step size.
        h_max : float or None
            Maximum allowed step size (defaults to ``time / 10``).
        max_steps : int
            Maximum number of accepted steps before stopping.
        """
        u_0 = self.build_initial_conditions()
        tf = self.time

        # Uniform output grid (same shape as solve_rk4 produces)
        t_eval = np.linspace(0.0, tf, self.steps + 1)

        if h0 is None:
            h0 = tf / self.steps
        if h_max is None:
            h_max = tf / 10.0

        h = min(h0, h_max)
        t = 0.0
        u_current = u_0.copy()

        # Store every accepted step for later interpolation
        times = [t]
        solutions = [u_current.copy()]

        safety = 0.9
        max_factor = 5.0
        min_factor = 0.1

        accepted = 0
        rejected = 0

        while t < tf and accepted < max_steps:
            # Don't overshoot the final time
            if t + h > tf:
                h = tf - t

            u_next, error = adaptive_rk4_step(F, t, u_current, h)

            # Weighted error norm (mix of relative & absolute)
            scale = atol + rtol * np.maximum(np.abs(u_current), np.abs(u_next))
            error_ratio = np.max(error / scale)

            if error_ratio <= 1.0:
                # --- accept step ---
                t += h
                u_current = u_next.copy()
                times.append(t)
                solutions.append(u_current.copy())
                accepted += 1

                # Increase step size (exponent 1/5 for the 5th-order method)
                if error_ratio > 1e-15:
                    h *= min(
                        max_factor, max(min_factor, safety * error_ratio ** (-1 / 5))
                    )
                else:
                    h *= max_factor
                h = min(h, h_max)

            else:
                # --- reject step, reduce step size ---
                rejected += 1
                if error_ratio > 1e-15:
                    h *= max(min_factor, safety * error_ratio ** (-1 / 4))
                else:
                    h *= min_factor

                if h < h_min:
                    # Cannot shrink any further — accept the step anyway
                    t += h
                    u_current = u_next.copy()
                    times.append(t)
                    solutions.append(u_current.copy())
                    accepted += 1
                    h = max(h, h_min)

        # Interpolate the adaptive solution back onto the uniform grid
        times_arr = np.array(times)
        solutions_arr = np.array(solutions)

        u = np.zeros((self.steps + 1, 4))
        for i in range(4):
            u[:, i] = np.interp(t_eval, times_arr, solutions_arr[:, i])

        # Ensure the initial condition is exact
        u[0] = u_0

        self.save_solution(u)
        print(
            f"Adaptive RK4 completed: {accepted} accepted, {rejected} rejected "
            f"steps (target grid: {self.steps} points)"
        )

    def get_energy(self, u: np.ndarray) -> np.ndarray:
        # u = [theta_1, omega_1, theta_2, omega_2]
        if u.ndim == 1:
            # Single state vector: shape (4,)
            return calc_kinetic_energy(
                M1, L1, u[1], M2, L2, u[3], u[0] - u[2]
            ) + calc_potential_energy(M1, L1, M2, L2, u[0], u[2])
        # Batch of state vectors: shape (N, 4)
        return calc_kinetic_energy(
            M1, L1, u[:, 1], M2, L2, u[:, 3], u[:, 0] - u[:, 2]
        ) + calc_potential_energy(M1, L1, M2, L2, u[:, 0], u[:, 2])

    def get_delta_E(self) -> np.ndarray:
        u = np.load(self.path_numpy)
        u_0, u_f = u[0], u[-1]
        return self.get_energy(u_f) - self.get_energy(u_0)

    def plot_e_vs_time(self, method=None, preset=None):
        """
        Delta E against H
        """
        import matplotlib.pyplot as plt

        t = np.linspace(0, self.time, self.steps)
        u = np.load(self.path_numpy)
        e_self = self.get_energy(u)
        plt.plot(t, e_self, label=f"{self.method}")
        # plt.axhline(e_self[0], color="r", linestyle="--")
        if method is not None:
            solver = DoublePendulumSolver(method=method, preset=preset)
            solver.solve()
            u_odd = np.load(solver.path_numpy)
            t = np.linspace(0, self.time, self.steps + 1)
            plt.plot(t, self.get_energy(u_odd), label=f"{method}")
            plt.axhline(self.get_energy(u_odd)[0], color="r", linestyle="--")

        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.legend()
        plt.show()

    def plot_delta_e_vs_h(self, h_steps):
        e_s = []
        for h_step in h_steps:
            # 0, 1, 2, 3, 4 -> 1, 0.1, 0.01, 0.001, 0.0001
            u_h = self.solve_rk4(dt=(0.1) ** h_step)
            t = np.linspace(0, self.time, self.steps + 1)
            e_h = self.get_energy(u_h)
            plt.plot(t, e_h, label=f"h={h_step}")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.legend()
        plt.show()

    def compare_solutions(self, method: Literal["rk4", "rk8"], preset: str):
        solver_odd = DoublePendulumSolver(method=method, preset=preset)
        if not self.path_numpy.exists():
            self.solve()
        if not solver_odd.path_numpy.exists():
            solver_odd.solve()
        u_self = np.load(self.path_numpy)
        u_odd = np.load(solver_odd.path_numpy)

        diffs = np.zeros(u_self.shape)
        print("Resting", self.method, "against", solver_odd.method)
        for i in range(u_self.shape[0] - 1):
            diff = u_self[i] - u_odd[i]
            diffs[i] = diff

        u_0_self = u_self[0]
        u_0_odd = u_odd[0]

        u_f_self = u_self[-1]
        u_f_odd = u_odd[-1]

        e_0_self = self.get_energy(u_0_self)
        print(f"Energy at t=0 (self): {e_0_self}")
        e_0_odd = solver_odd.get_energy(u_0_odd)
        print(f"Energy at t=0 (odd): {e_0_odd}")
        e_0_diff = e_0_self - e_0_odd
        print(f"Energy difference at t=0: {e_0_diff}")

        e_f_self = self.get_energy(u_f_self)
        print(f"Energy at t=final (self): {e_f_self}")
        e_f_odd = solver_odd.get_energy(u_f_odd)
        print(f"Energy at t=final (odd): {e_f_odd}")
        e_self_f_diff = e_f_self - e_0_self
        e_odd_f_diff = e_f_odd - e_0_odd

        print(f"Energy difference at t=final (self): {e_self_f_diff}")
        print(f"Energy difference at t=final (odd): {e_odd_f_diff}")

        mse = np.mean(diffs**2)
        deviation = np.sqrt(mse)
        mean = np.mean(diffs)
        print(f"MSE: {mse}, deviation: {deviation}, mean: {mean}")
