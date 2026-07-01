from physi.double_pendulum.double_pendulum import DoublePendulumSolver


def main():
    solver = DoublePendulumSolver(
        method="rk8",
        preset="hard",
    )
    solver.plot_e_vs_time(method="rk4", preset="hard")


if __name__ == "__main__":
    main()
