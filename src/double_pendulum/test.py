import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

L1 = 4.0
m1 = 1.0
g = 9.81
w = np.sqrt(g / L1)


def F(t, u, gamma=0.1, omega0=1.0):
    u1, u2 = u
    du1 = u2
    du2 = -(w**2) * np.sin(u1)
    return np.array([du1, du2])


def rk4_step(f, t, u, h):
    k1 = h * f(t, u)
    k2 = h * f(t + h / 2, u + k1 / 2)
    k3 = h * f(t + h / 2, u + k2 / 2)
    k4 = h * f(t + h, u + k3)
    return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Initial conditions: y(0) = 1, y'(0) = 0
t0, tf, h = 0.0, 20.0, 0.01
t = np.arange(t0, tf, h)
u = np.zeros((len(t), 2))
u[0] = [4.0, 0.1]  # [y0, y'0]

for i in range(len(t) - 1):
    u[i + 1] = rk4_step(F, t[i], u[i], h)

## Angles and velocities
angles = u[:, 0]
velocities = u[:, 1]


x = np.sin(angles)
y = -np.cos(angles)

# Create pendulum animation GIF
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect("equal")
ax.grid(True)
ax.set_title("Simple Pendulum")

(line,) = ax.plot([], [], "o-", lw=2, markersize=8, color="C0")
time_text = ax.text(
    0.05, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment="top"
)


def init():
    line.set_data([], [])
    time_text.set_text("")
    return line, time_text


def animate(frame_idx):
    xi = x[frame_idx]
    yi = y[frame_idx]
    line.set_data([0, xi], [0, yi])
    time_text.set_text(f"t = {t[frame_idx]:.2f}s")
    return line, time_text


# Use every 5th frame to keep the GIF manageable
skip = 5
frames = range(0, len(t), skip)
ani = animation.FuncAnimation(
    fig, animate, frames=frames, init_func=init, interval=20, blit=True
)
ani.save("pendulum.gif", writer="pillow", fps=30)
print("Saved pendulum.gif")
