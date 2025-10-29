# model.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import math
import numpy as np


@dataclass
class Environment:
    """
    Minimal environment; maps physical drivers to multipliers used in dynamics.
    Defaults are chosen so that, if you ignore env in your step(), you retain
    prior behavior. See methods for the exact (soft) mappings.
    """
    temperature_C: float = 24.0
    dissolved_oxygen_mgL: float = 6.0
    salinity_ppt: float = 0.5
    ammonia_mgL: float = 0.0
    pH: float = 7.5
    conductivity_uScm: float = 300.0
    refuge_index: float = 0.3       # 0=open water, 1=dense refuge
    predator_pressure: float = 0.0  # exogenous predators (0..1)

    def clamp(self) -> "Environment":
        self.refuge_index = min(max(self.refuge_index, 0.0), 1.0)
        self.predator_pressure = min(max(self.predator_pressure, 0.0), 1.0)
        return self

    # ---- helper squashing ----
    def _logistic(self, x: float, x0: float, k: float = 0.28, lo: float = 0.7, hi: float = 1.2) -> float:
        s = 1.0 / (1.0 + math.exp(-k * (x - x0)))
        return lo + (hi - lo) * s

    # ---- survival multipliers (stage-specific) ----
    def stage_survival_mult(self, stage: str) -> float:
        T, O2, S, NH3 = self.temperature_C, self.dissolved_oxygen_mgL, self.salinity_ppt, self.ammonia_mgL
        temp = self._logistic(T, 24.0, k=0.25, lo=0.75, hi=1.15)
        o2   = 1.0 if O2 >= 1.3 else max(0.6, O2 / 1.3)          # mosquitofish tolerate hypoxia
        sal  = 1.0 if S <= 10 else max(0.75, 1.0 - 0.02 * (S - 10))  # mild penalty at higher salinity
        amm  = 1.0 if NH3 <= 0.5 else max(0.7, 1.0 - 0.1 * (NH3 - 0.5))
        base = temp * o2 * sal * amm
        # predators primarily affect early stages
        if stage == "fry":
            base *= max(0.6, 1.0 - 0.5 * self.predator_pressure)
        elif stage == "juvenile":
            base *= max(0.7, 1.0 - 0.35 * self.predator_pressure)
        else:
            base *= max(0.8, 1.0 - 0.2 * self.predator_pressure)
        return float(base)

    # ---- maturation multipliers (edge-specific) ----
    def maturation_mult(self, src_stage: str, dst_stage: str) -> float:
        T, O2, S, NH3 = self.temperature_C, self.dissolved_oxygen_mgL, self.salinity_ppt, self.ammonia_mgL
        temp = self._logistic(T, 24.0, k=0.30, lo=0.8, hi=1.25)   # warmer -> faster development
        o2   = 1.0 if O2 >= 2.0 else max(0.6, O2 / 2.0)
        sal  = 1.0 if S <= 15 else max(0.8, 1.0 - 0.01 * (S - 15))
        amm  = 1.0 if NH3 <= 0.5 else max(0.85, 1.0 - 0.05 * (NH3 - 0.5))
        return float(temp * o2 * sal * amm)

    # ---- encounter multiplier for predation/cannibalism ----
    def encounter_mult(self) -> float:
        # Refuge reduces encounters; external predators can increase risky movement.
        # Fixed stray ')' from earlier draft.
        return float(max(0.1, (1.0 - self.refuge_index) * (1.0 + 0.5 * self.predator_pressure)))


@dataclass
class SimpleModel:
    """
    Base model with 4-stage state: Fry, Juvenile, AdultFemale, AdultMale.
    Derived models implement .step().
    """
    time_step_weeks: float = 1.0
    iterations: int = 60
    env: Environment = field(default_factory=Environment)

    # Baseline survival (before env)
    fry_survival_probability: float = 0.92
    juvenile_survival_probability: float = 0.95
    adult_f_survival_probability: float = 0.97
    adult_m_survival_probability: float = 0.97

    # Baseline maturation (before env)
    mat_fj_scale: float = 0.25      # F->J per step
    mat_ja_scale: float = 0.18      # J->A per step (split to m/f internally via sex ratio)

    # Juvenile bottleneck controls
    K_j: Optional[float] = None           # capacity; if None, off
    juv_density_mortality: float = 0.0    # extra loss ~ c * J^2

    # Reproduction
    fecundity_per_female: float = 0.08    # recruits per adult female per step (before env)
    sex_ratio_at_birth_male: float = 0.5  # fraction male among newborn fry

    # Cannibalism (functional response in subclass): parameters are “nominal” before refuge
    cannibalism_attack_rate: float = 0.01
    cannibalism_handling_time: float = 0.0
    cannibalism_kernel: str = "male"      # "male" | "female" | "both"
    cannibalism_type: str = "H2"          # "H2" or "H3" (Holling)

    # Optional endocrine/toxicant effects on fecundity and sex ratio
    fecundity_multiplier: float = 1.0     # <1 reduces brood due to toxins etc.
    male_bias_multiplier: float = 1.0     # >1 biases sex ratio toward males at birth

    # NEW (optional): experimental design hooks (no behavior change unless set)
    tank_shape: Optional[object] = None        # e.g., environment_profiles.TankShape
    feeding_schedule: Optional[object] = None  # e.g., environment_profiles.FeedingSchedule

    # Provenance / diagnostics
    provenance: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class State:
        fry: float
        juveniles: float
        adult_female: float
        adult_male: float

    def step(self, s: "SimpleModel.State") -> "SimpleModel.State":
        raise NotImplementedError

    def run(self, s0: "SimpleModel.State") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the model for `iterations` steps, calling .step() each time.
        Returns arrays for Fry, Juveniles, AdultFemale, AdultMale.
        """
        F = [float(s0.fry)]
        J = [float(s0.juveniles)]
        AF = [float(s0.adult_female)]
        AM = [float(s0.adult_male)]

        # local copy of state (floats only)
        s = SimpleModel.State(*map(float, (s0.fry, s0.juveniles, s0.adult_female, s0.adult_male)))

        # maintain a step index for schedules; harmless if unused
        step_idx = int(self.provenance.get("step_index", 0))

        for _ in range(self.iterations):
            s = self.step(s)
            F.append(max(0.0, float(s.fry)))
            J.append(max(0.0, float(s.juveniles)))
            AF.append(max(0.0, float(s.adult_female)))
            AM.append(max(0.0, float(s.adult_male)))
            step_idx += 1

        self.provenance["step_index"] = step_idx
        return np.array(F), np.array(J), np.array(AF), np.array(AM)