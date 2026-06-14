from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PendulumState:
    m1: float
    m2: float
    theta_1: float
    theta_2: float
    omega_1: float
    omega_2: float
    M: float
    l1: float
    l2: float
    gravity: float
    h: float
    time: float


def d2theta1(theta_1, theta_2, omega_1, omega_2):

    diff = theta_1 - theta_2
    alpha = m1 + m2 * np.sin(diff) ** 2
    # θ₁'' = [m₂·g·sin(θ₂)·cos(Δ) - m₂·l₁·ω₁²·sin(Δ)·cos(Δ) - m₂·l₂·ω₂²·sin(Δ) - (m₁+m₂)·g·sin(θ₁)] / [l₁·(m₁ + m₂·sin²(Δ))]
    numerator = (
        m2 * gravity * np.sin(theta_2) * np.cos(diff)
        - m2 * l1 * omega_1**2 * np.sin(diff) * np.cos(diff)
        - m2 * l2 * omega_2**2 * np.sin(diff)
        - M * GRAVITY * np.sin(theta_1)
    )
    return numerator / (alpha * l1)


def d2theta2(theta_1, theta_2, omega_1, omega_2):
    diff = theta_1 - theta_2
    alpha = m1 + m2 * np.sin(diff) ** 2
    first_term = np.sin(diff) * (
        M * l1 * omega_1**2 + m2 * l2 * omega_2**2 * np.cos(diff)
    )
    second_term = GRAVITY * (M * np.sin(theta_1) * np.cos(diff) - M * np.sin(theta_2))
    return (first_term + second_term) / (alpha * l2)


def rk4_step(f, t, u, h):
    k1 = h * f(t, u, d2theta1, d2theta2)
    k2 = h * f(t + h / 2, u + k1 / 2, d2theta1, d2theta2)
    k3 = h * f(t + h / 2, u + k2 / 2, d2theta1, d2theta2)
    k4 = h * f(t + h, u + k3, d2theta1, d2theta2)
    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Theta update
def F(t, u, second_theta_1, second_theta_2):
    # u = [theta_1, omega_1, theta_2, omega_2]
    theta_1, omega_1, theta_2, omega_2 = u

    d_omega1 = second_theta_1(theta_1, theta_2, omega_1, omega_2)
    d_omega2 = second_theta_2(theta_1, theta_2, omega_1, omega_2)

    return np.array([omega_1, d_omega1, omega_2, d_omega2])


def main():
    steps = 100000
    time_steps = 100

    u_0 = np.array(
        [
            INITIAL_ANGLES[0],
            INITIAL_VELOCITIES[0],
            INITIAL_ANGLES[1],
            INITIAL_VELOCITIES[1],
        ]
    )

    u = np.zeros((steps + 1, 4))
    u[0] = u_0

    t = np.linspace(0, time_steps, steps + 1)
    h = t[1] - t[0]

    for i in range(steps):
        u[i + 1] = rk4_step(F, t[i], u[i], h)

    angles_1 = u[:, 0]
    angles_2 = u[:, 2]

    x_1 = l1 * np.sin(angles_1)
    y_1 = -l1 * np.cos(angles_1)
    x_2 = l2 * np.sin(angles_2) + x_1
    y_2 = -l2 * np.cos(angles_2) + y_1

    fig, ax = plt.subplots()
    # ax.plot(x_1, y_1, label="pendulum 1")
    ax.plot(x_2, y_2, label="pendulum 2")
    ax.legend()
    plt.show()
