import matplotlib.pyplot as plt
import numpy as np

MASSES = [1.0, 2.0]

m1 = MASSES[0]
m2 = MASSES[1]

M = m1 + m2

LENGTHS = [1.0, 2.0]

l1 = LENGTHS[0]
l2 = LENGTHS[1]


INITIAL_ANGLES = [0.0, 0.0]
INITIAL_VELOCITIES = [0.0, 0.0]
GRAVITY = 9.81


def double_derivative_theta_1(theta_1, theta_2, omega_1, omega_2):
    diff = theta_1 - theta_2
    alpha = m1 + m2 * np.sin(diff) ** 2
    first_term = -np.sin(diff) * (
        m2 * l2 * omega_1**2 * np.cos(diff)
        + m2 * l2 * omega_2**2 * np.cos(diff)
        + m1 * l1 * omega_2**2
    )
    second_term = -GRAVITY * (M * np.sin(diff) - m2 * np.sin(theta_2) * np.cos(diff))
    return (first_term + second_term) / (alpha * l1)


def double_derivative_theta_2(theta_1, theta_2, omega_1, omega_2):
    diff = theta_1 - theta_2
    alpha = m1 + m2 * np.sin(diff) ** 2
    first_term = np.sin(diff) * (
        M * l1 * omega_1**2 + m2 * l2 * omega_2**2 * np.cos(diff)
    )
    second_term = GRAVITY * (M * np.sin(theta_1) * np.cos(diff) - M * np.sin(theta_2))
    return (first_term + second_term) / (alpha * l2)
