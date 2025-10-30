# main.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional, Dict, Any

from .frymodel import FryModel
from .model import SimpleModel, Environment
from .graph import simulate_with_env_profile, seasonal_env_updater

# Optional imports; CLI works even if this module isn't present.
try:
    from .environment_profiles import TankShape, FeedingSchedule
except Exception:
    TankShape = None
    FeedingSchedule = None


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run 4-stage mosquitofish model with environment, cannibalism, optional tank/feeding, and temporal env support."
    )
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument(
        "--init", type=float, nargs=4, metavar=("F0", "J0", "AF0", "AM0"),
        default=(10.0, 2.0, 1.0, 1.0),
        help="Initial stocks for Fry, Juveniles, AdultFemale, AdultMale"
    )

    # ---- Environment ----
    ap.add_argument("--tempC", type=float, default=24.0)
    ap.add_argument("--o2", type=float, default=6.0)
    ap.add_argument("--sal", type=float, default=0.5)
    ap.add_argument("--nh3", type=float, default=0.0)
    ap.add_argument("--ph", type=float, default=7.5)
    ap.add_argument("--cond", type=float, default=300.0)
    ap.add_argument("--refuge", type=float, default=0.3)
    ap.add_argument("--pred", type=float, default=0.0)
    ap.add_argument("--profile", type=str, choices=["wild", "tank"])

    # ---- Cannibalism (Ricker-style) ----
    ap.add_argument("--c-strength", type=float, default=0.012,
                    help="Ricker cannibalism strength c in exp(-c * P_eff * encounter)")
    ap.add_argument("--male-fraction", type=float, default=0.5,
                    help="Fraction of adults that are male (used for P_eff)")
    ap.add_argument("--enable-nonfry-cannibalism", action="store_true",
                    help="Apply (tiny by default) cannibalism multipliers to J/AF/AM")

    # Optional fine-grained stage susceptibilities and predator weights
    ap.add_argument("--sus-fry", type=float, default=None)
    ap.add_argument("--sus-juvenile", type=float, default=None)
    ap.add_argument("--sus-adult-female", type=float, default=None)
    ap.add_argument("--sus-adult-male", type=float, default=None)
    ap.add_argument("--pred-w-female", type=float, default=None)
    ap.add_argument("--pred-w-male", type=float, default=None)

    # ---- Juvenile bottleneck ----
    ap.add_argument("--Kj", type=float, default=0.0, help="Juvenile capacity; 0 disables")
    ap.add_argument("--juv-density-mortality", type=float, default=0.0,
                    help="Quadratic loss coefficient on juveniles")

    # ---- Reproduction & sex ratio / endocrine ----
    ap.add_argument("--fec", type=float, default=0.08)
    ap.add_argument("--fec-mult", type=float, default=1.0)
    ap.add_argument("--p-male", type=float, default=0.5)
    ap.add_argument("--male-bias-mult", type=float, default=1.0)

    # ---- Tank shape (optional) ----
    ap.add_argument("--tank-length-m", type=float, default=0.0)
    ap.add_argument("--tank-width-m", type=float, default=0.0)
    ap.add_argument("--tank-depth-m", type=float, default=0.0)

    # ---- Feeding schedule (optional) ----
    ap.add_argument("--feed-regime", type=str, default="continuous",
                    choices=["continuous", "pulsed"])
    ap.add_argument("--food-index", type=float, default=0.7)
    ap.add_argument("--feed-period", type=int, default=7)
    ap.add_argument("--feed-duty", type=float, default=0.6)
    ap.add_argument("--feed-low", type=float, default=0.3)
    ap.add_argument("--feed-high", type=float, default=1.0)

    # ---- Temporal environment (optional) ----
    ap.add_argument("--env-trajectory", type=str, default="none",
                    choices=["none", "seasonal", "json"],
                    help="If 'seasonal', apply sinusoidal env; if 'json', read stepwise updates from --env-json.")
    ap.add_argument("--env-json", type=str, default="",
                    help="Path to JSON list of {'step': int, 'update': {field: value}} items.")

    # ---- Output ----
    ap.add_argument("--out", type=str, default="",
                    help="Write JSON to this path instead of printing")

    # ---- Deprecated Holling flags (parsed for back-compat; ignored) ----
    ap.add_argument("--c-kernel", type=str, default=None, choices=["male", "female", "both"])
    ap.add_argument("--c-type", type=str, default=None, choices=["H2", "H3"])
    ap.add_argument("--c-attack", type=float, default=None)
    ap.add_argument("--c-handle", type=float, default=None)

    return ap.parse_args()


def _build_json_updater(path: str):
    """
    Load a list of {'step': int, 'update': {field: value}} and return an updater.
    Fields must be attributes on Environment (e.g., temperature_C, dissolved_oxygen_mgL, refuge_index, predator_pressure, etc.)
    """
    data = json.loads(Path(path).read_text())
    # index by step for O(1) lookup
    by_step: Dict[int, Dict[str, Any]] = {}
    for item in data:
        step = int(item.get("step", 0))
        upd = dict(item.get("update", {}))
        by_step[step] = upd

    def updater(t: int, env: Environment) -> None:
        if t in by_step:
            for k, v in by_step[t].items():
                if hasattr(env, k):
                    setattr(env, k, v)
    return updater


def main():
    a = parse_args()
    m = FryModel(iterations=a.steps)

    # Environment (base)
    m.env = Environment(
        temperature_C=a.tempC,
        dissolved_oxygen_mgL=a.o2,
        salinity_ppt=a.sal,
        ammonia_mgL=a.nh3,
        pH=a.ph,
        conductivity_uScm=a.cond,
        refuge_index=a.refuge,
        predator_pressure=a.pred,
    )

    # Profile defaults that you can still override via flags
    if a.profile == "wild":
        m.tank_shape = None
        m.env.refuge_index = 0.7 if a.refuge == 0.3 else a.refuge  # respect explicit flags
        m.env.predator_pressure = 0.4 if a.pred == 0.0 else a.pred
        if FeedingSchedule is not None and a.feed_regime == "continuous" and a.food_index == 0.7:
            m.feeding_schedule = FeedingSchedule(regime="continuous", food_index=0.8)
    elif a.profile == "tank":
        if TankShape is not None and a.tank_length_m == 0.0 and a.tank_width_m == 0.0 and a.tank_depth_m == 0.0:
            m.tank_shape = TankShape(length_m=0.6, width_m=0.3, depth_m=0.25)
        m.env.refuge_index = 0.2 if a.refuge == 0.3 else a.refuge
        m.env.predator_pressure = 0.0 if a.pred == 0.0 else a.pred
        if FeedingSchedule is not None and a.feed_regime == "continuous":
            m.feeding_schedule = FeedingSchedule(regime="pulsed", food_index=max(0.0, min(1.0, a.food_index)),
                                                 pulse_period_steps=a.feed_period, pulse_duty_cycle=a.feed_duty,
                                                 pulse_low=a.feed_low, pulse_high=a.feed_high)

    # Cannibalism (Ricker-style)
    m.strength_cannibalism = float(a.c_strength)
    m.male_fraction = max(0.0, min(1.0, float(a.male_fraction)))
    m.enable_nonfry_cannibalism = bool(a.enable_nonfry_cannibalism)

    # Optional overrides
    if a.sus_fry is not None:           m.susceptibility_fry = float(a.sus_fry)
    if a.sus_juvenile is not None:      m.susceptibility_juvenile = float(a.sus_juvenile)
    if a.sus_adult_female is not None:  m.susceptibility_adult_female = float(a.sus_adult_female)
    if a.sus_adult_male is not None:    m.susceptibility_adult_male = float(a.sus_adult_male)
    if a.pred_w_female is not None:     m.predator_weight_female = float(a.pred_w_female)
    if a.pred_w_male is not None:       m.predator_weight_male = float(a.pred_w_male)

    # Juvenile bottleneck
    m.K_j = None if a.Kj <= 0 else float(a.Kj)
    m.juv_density_mortality = float(a.juv_density_mortality)

    # Reproduction & sex ratio
    m.fecundity_per_female = float(a.fec)
    m.fecundity_multiplier = float(a.fec_mult)
    m.sex_ratio_at_birth_male = max(0.0, min(1.0, float(a.p_male)))
    m.male_bias_multiplier = float(a.male_bias_mult)

    # Tank shape (manual override)
    if (
        TankShape is not None
        and a.tank_length_m > 0.0
        and a.tank_width_m > 0.0
        and a.tank_depth_m > 0.0
    ):
        m.tank_shape = TankShape(
            length_m=a.tank_length_m, width_m=a.tank_width_m, depth_m=a.tank_depth_m
        )

    # Feeding schedule (manual override)
    if FeedingSchedule is not None:
        m.feeding_schedule = FeedingSchedule(
            regime=a.feed_regime, food_index=max(0.0, min(1.0, a.food_index)),
            pulse_period_steps=a.feed_period, pulse_duty_cycle=a.feed_duty,
            pulse_low=a.feed_low, pulse_high=a.feed_high,
        )

    # Back-compat note if deprecated flags were supplied
    deprecated_used = any(
        x is not None for x in (a.c_kernel, a.c_type, a.c_attack, a.c_handle)
    )
    if deprecated_used:
        m.provenance.setdefault("notes", []).append(
            "Deprecated Holling CLI flags were provided and ignored; using Ricker cannibalism."
        )

    # -------- Run with or without temporal env --------
    if a.env_trajectory == "none":
        F, J, AF, AM = m.run(SimpleModel.State(*a.init))
    else:
        if a.env_trajectory == "seasonal":
            updater = seasonal_env_updater(period_steps=max(8, a.steps))
        elif a.env_trajectory == "json":
            if not a.env_json:
                raise SystemExit("--env-trajectory=json requires --env-json path")
            updater = _build_json_updater(a.env_json)
        else:
            updater = None

        F, J, AF, AM = simulate_with_env_profile(
            m, init=tuple(map(float, a.init)), steps=a.steps, env_updater=updater
        )

    out = {
        "fry": F.tolist(),
        "juvenile": J.tolist(),
        "adult_female": AF.tolist(),
        "adult_male": AM.tolist(),
        "provenance": m.provenance,
    }

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()