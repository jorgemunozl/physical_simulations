from pathlib import Path

import numpy as np

# Physical parameters — kept in sync with main.py
from main import L1, L2, M1, M2

from manim import (
    BLUE,
    GREEN,
    WHITE,
    YELLOW,
    Dot,
    Line,
    Scene,
    VGroup,
)

# ── Settings ──────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "data" / "double_pendulum.npy"
ANIM_DURATION = 10.0  # desired total playback duration in scene-seconds
RENDER_FPS = 60  # must match the -ql/--fps flag you pass to manim
#   -ql → 15 fps | -qm → 30 fps | -qh/-qk → 60 fps
SUBSAMPLING = 1  # extra trail thinning (1 = every rendered step)


# ══════════════════════════════════════════════════════════════════════
#  Manim scene
# ══════════════════════════════════════════════════════════════════════


class DoublePendulum(Scene):
    """Animate the double pendulum from saved simulation data."""

    def construct(self):
        # ── Load trajectory ─────────────────────────────────────────
        # Shape: (N, 4)  columns: [theta1, omega1, theta2, omega2]
        traj = np.load(DATA_PATH)
        num_frames = len(traj)

        # ── Cartesian coordinates ────────────────────────────────────
        # Pivot at origin; angles measured from the downward vertical.
        x1 = L1 * np.sin(traj[:, 0])
        y1 = -L1 * np.cos(traj[:, 0])
        x2 = x1 + L2 * np.sin(traj[:, 2])
        y2 = y1 - L2 * np.cos(traj[:, 2])

        # ── Scale so the pendulum fills ~3 Manim units ───────────────
        max_extent = max(np.abs(x2).max(), np.abs(y2).max(), 0.1) + 0.3
        scale = 3.0 / max_extent

        def pt(xx: float, yy: float) -> np.ndarray:
            """Physical (x, y)  →  Manim 3-D point."""
            return np.array([xx * scale, yy * scale, 0.0])

        pivot_pt = pt(0.0, 0.0)
        pts_p1 = [pt(x1[i], y1[i]) for i in range(num_frames)]
        pts_p2 = [pt(x2[i], y2[i]) for i in range(num_frames)]

        # ── Initial scene objects ─────────────────────────────────────
        pivot = Dot(pivot_pt, color=WHITE, radius=0.08)

        mass1 = Dot(pts_p1[0], color=BLUE, radius=0.15 * (M1 ** (1 / 3)))
        mass2 = Dot(pts_p2[0], color=GREEN, radius=0.15 * (M2 ** (1 / 3)))

        rod1 = Line(pivot_pt, pts_p1[0], color=BLUE, stroke_width=6)
        rod2 = Line(pts_p1[0], pts_p2[0], color=GREEN, stroke_width=5)

        trail = VGroup()

        self.add(pivot, trail, rod1, rod2, mass1, mass2)
        self.wait(0.3)

        # ── Frame-by-frame playback ─────────────────────────────────────
        # Each self.wait() must be >= 1/RENDER_FPS or manim silently rounds
        # it up, making the video much longer than intended.  We stride
        # through the data so that exactly one rendered frame passes per
        # loop iteration.
        min_wait = 1.0 / RENDER_FPS
        ideal_wait = ANIM_DURATION / max(num_frames - 1, 1)
        stride = max(1, int(np.ceil(min_wait / ideal_wait)))  # rows to skip
        indices = range(0, num_frames, stride)
        time_per_step = stride * ideal_wait  # now always >= min_wait

        for i in indices:
            p1 = pts_p1[i]
            p2 = pts_p2[i]

            # Move masses
            mass1.move_to(p1)
            mass2.move_to(p2)

            # Redraw rods
            rod1.become(Line(pivot_pt, p1, color=BLUE, stroke_width=6))
            rod2.become(Line(p1, p2, color=GREEN, stroke_width=5))

            # Trailing dot on mass 2
            if i % (stride * SUBSAMPLING) == 0:
                alpha = 0.15 + 0.65 * (i / max(num_frames - 1, 1))
                dot = Dot(p2, color=YELLOW, radius=0.025)
                dot.set_opacity(alpha)
                trail.add(dot)

            self.wait(time_per_step)

        self.wait(1.0)


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from manim import config

    config.progress_bar = "display"  # always show the progress bar
    config.quality = "low_quality"  # -ql  (change to medium/high as needed)
    config.preview = True  # open the video when done
    config.disable_caching = True  # always re-render, never use stale cache
    scene = DoublePendulum()
    scene.render()
