# graph.py
from __future__ import annotations
from typing import Callable, Optional, Tuple, List, Dict
import math
import numpy as np
import matplotlib.pyplot as plt

from .model import SimpleModel, Environment
from .frymodel import FryModel

# Optional presets/utilities if available:
try:
    from .environment_profiles import TankShape, FeedingSchedule
except Exception:
    TankShape = None
    FeedingSchedule = None


def _ensure_axes():
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    return fig, ax


def plot_over_time(model: FryModel,
                   init: Tuple[float, float, float, float] = (10.0, 2.0, 1.0, 1.0),
                   title: Optional[str] = None) -> None:
    """Run model.run(...) and plot Fry, Juveniles, AdultF, AdultM."""
    F, J, AF, AM = model.run(SimpleModel.State(*init))
    t = np.arange(len(F))
    fig, ax = _ensure_axes()
    ax.plot(t, F, label="Fry")
    ax.plot(t, J, label="Juveniles")
    ax.plot(t, AF, label="Adult ♀")
    ax.plot(t, AM, label="Adult ♂")
    ax.set_xlabel("Step")
    ax.set_ylabel("Abundance")
    ax.set_title(title or model.__class__.__name__)
    ax.legend()
    fig.tight_layout()
    plt.show()


# -----------------------------
# Temporal environment support
# -----------------------------
EnvUpdater = Callable[[int, Environment], None]

def simulate_with_env_profile(model: FryModel,
                              init: Tuple[float, float, float, float],
                              steps: int,
                              env_updater: Optional[EnvUpdater] = None):
    """
    Manual stepping loop so we can change Environment each step.
    Returns arrays (F, J, AF, AM) matching model.run() shape.
    """
    s = SimpleModel.State(*map(float, init))
    F, J, AF, AM = [s.fry], [s.juveniles], [s.adult_female], [s.adult_male]

    # keep a step index for feeding schedules, as in model.run()
    model.provenance.setdefault("step_index", 0)

    for k in range(steps):
        if env_updater is not None:
            env_updater(k, model.env)  # mutate in place

        s = model.step(s)
        F.append(max(0.0, float(s.fry)))
        J.append(max(0.0, float(s.juveniles)))
        AF.append(max(0.0, float(s.adult_female)))
        AM.append(max(0.0, float(s.adult_male)))
        model.provenance["step_index"] = int(model.provenance["step_index"]) + 1

    return np.array(F), np.array(J), np.array(AF), np.array(AM)


def seasonal_env_updater(period_steps: int = 52,
                         temp_C_base: float = 24.0,
                         temp_amp: float = 4.0,
                         o2_base: float = 6.0,
                         o2_amp: float = 1.0,
                         pred_base: float = 0.2,
                         pred_amp: float = 0.15,
                         refuge_base: float = 0.5,
                         refuge_amp: float = 0.2) -> EnvUpdater:
    """
    Build a simple sinusoidal seasonal updater for key env variables.
    Safe bounds applied inside Environment methods later.
    """
    def _update(t: int, env: Environment) -> None:
        phase = 2.0 * math.pi * (t % period_steps) / max(1, period_steps)
        env.temperature_C = temp_C_base + temp_amp * math.sin(phase)
        env.dissolved_oxygen_mgL = max(0.1, o2_base + o2_amp * math.cos(phase))
        env.predator_pressure = min(1.0, max(0.0, pred_base + pred_amp * math.sin(phase + math.pi / 4)))
        env.refuge_index = min(1.0, max(0.0, refuge_base + refuge_amp * math.cos(phase + math.pi / 6)))
    return _update


def demo_run(iterations: int = 60,
             init: Tuple[float, float, float, float] = (10.0, 2.0, 1.0, 1.0),
             with_tank: bool = False,
             with_feeding: bool = False,
             with_seasonal_env: bool = False):
    """
    Demo helper used by notebooks/tests. Returns (F,J,AF,AM).
    """
    m = FryModel(iterations=iterations)
    m.env = Environment(temperature_C=26.0, dissolved_oxygen_mgL=5.5,
                        salinity_ppt=0.5, ammonia_mgL=0.0,
                        pH=7.5, conductivity_uScm=300.0,
                        refuge_index=0.4, predator_pressure=0.15)

    # Optional tank shape & feeding schedule (safe if module missing)
    if with_tank and TankShape is not None:
        m.tank_shape = TankShape(length_m=0.6, width_m=0.3, depth_m=0.25)
    if with_feeding and FeedingSchedule is not None:
        m.feeding_schedule = FeedingSchedule(regime="pulsed", food_index=0.6,
                                             pulse_period_steps=7, pulse_duty_cycle=0.5,
                                             pulse_low=0.3, pulse_high=0.9)

    # pick runner
    if with_seasonal_env:
        env_fn = seasonal_env_updater(period_steps=52)
        return simulate_with_env_profile(m, init=init, steps=iterations, env_updater=env_fn)
    else:
        return m.run(SimpleModel.State(*init))


def plot_with_seasonal_env(iterations: int = 120):
    """Quick visual check for temporal env effects."""
    F, J, AF, AM = demo_run(iterations=iterations,
                            init=(10, 2, 1, 1),
                            with_tank=False,
                            with_feeding=True,
                            with_seasonal_env=True)
    t = np.arange(len(F))
    fig, ax = _ensure_axes()
    ax.plot(t, F, label="Fry")
    ax.plot(t, J, label="Juveniles")
    ax.plot(t, AF, label="Adult ♀")
    ax.plot(t, AM, label="Adult ♂")
    ax.set_xlabel("Step")
    ax.set_ylabel("Abundance")
    ax.set_title("Temporal environment demo (seasonal)")
    ax.legend()
    fig.tight_layout()
    plt.show()