# environment_profiles.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict, Any, Optional
import math

@dataclass
class TankShape:
    """Geometric descriptors for encounter scaling."""
    length_m: float        # tank long side
    width_m: float         # tank short side
    depth_m: float         # water depth

    @property
    def volume_L(self) -> float:
        return 1000.0 * self.length_m * self.width_m * self.depth_m

    @property
    def footprint_m2(self) -> float:
        return self.length_m * self.width_m

    @property
    def boundary_m(self) -> float:
        # perimeter that can cause wall-following / encounter biases
        return 2.0 * (self.length_m + self.width_m)

    @property
    def aspect(self) -> float:
        return max(self.length_m, self.width_m) / max(1e-9, min(self.length_m, self.width_m))

@dataclass
class FeedingSchedule:
    """
    Feeding regimes: continuous vs pulses (simulate fasting periods).
    food_index scales fecundity; fasting pulses temporarily reduce it.
    """
    regime: Literal["continuous", "pulsed"] = "continuous"
    food_index: float = 0.7            # 0..1; maps to fecundity multiplier
    pulse_period_steps: int = 7        # for 'pulsed', days/weeks per your time step
    pulse_duty_cycle: float = 0.6      # fraction of time with food available (0..1)
    pulse_low: float = 0.3             # fecundity scale during low-food window (<= food_index)
    pulse_high: float = 1.0            # during high-food window (>= food_index)

    def fecundity_multiplier_at(self, step: int) -> float:
        if self.regime == "continuous":
            return max(0.0, min(1.0, self.food_index))
        # pulsed: square-wave between low and high around base food_index
        T = max(1, self.pulse_period_steps)
        phase = (step % T) / T
        on = phase < self.pulse_duty_cycle
        target = self.pulse_high if on else self.pulse_low
        # bias toward configured food_index
        return max(0.0, min(1.0, 0.5 * (self.food_index + target)))

@dataclass
class EncounterScales:
    """Multiplicative factors for encounter/attack from geometry/density."""
    boundary_factor: float = 1.0
    density_factor: float = 1.0
    aspect_factor: float = 1.0

def encounter_scaling_from_shape(n_fish: float, shape: TankShape) -> EncounterScales:
    # Density in the footprint plane encourages encounters; boundary can increase wall-following
    # Long narrow tanks (high aspect) also bias movement paths.
    density = n_fish / max(1e-9, shape.footprint_m2)
    density_factor = 1.0 + 0.02 * min(200.0, density)      # modest linear boost
    boundary_factor = 1.0 + 0.001 * min(2000.0, shape.boundary_m)  # small bump
    aspect_factor = 1.0 + 0.1 * max(0.0, shape.aspect - 1.0)
    return EncounterScales(boundary_factor, density_factor, aspect_factor)

# Presets that fill vendor model fields (Environment + cannibalism + bottleneck)
def wild_pond_preset() -> Dict[str, Any]:
    return {
        "env.temperature_C": 24.0, "env.dissolved_oxygen_mgL": 6.0,
        "env.salinity_ppt": 0.5, "env.ammonia_mgL": 0.0, "env.pH": 7.6,
        "env.conductivity_uScm": 300.0, "env.refuge_index": 0.6, "env.predator_pressure": 0.3,
        # literature: field/mesocosm → cannibals predominantly female; FR leaning Type III with refuge
        "cannibalism_kernel": "female", "cannibalism_type": "H3",
        "cannibalism_attack_rate": 0.006, "cannibalism_handling_time": 0.07,
        "K_j": 60.0, "juv_density_mortality": 2e-4,
        "fecundity_multiplier": 1.0, "male_bias_multiplier": 1.0
    }

def lab_tank_preset() -> Dict[str, Any]:
    return {
        "env.temperature_C": 25.0, "env.dissolved_oxygen_mgL": 6.0,
        "env.salinity_ppt": 0.5, "env.ammonia_mgL": 0.0, "env.pH": 7.5,
        "env.conductivity_uScm": 300.0, "env.refuge_index": 0.15, "env.predator_pressure": 0.0,
        # lab reports of male cannibalism; simple habitat → Type II
        "cannibalism_kernel": "male", "cannibalism_type": "H2",
        "cannibalism_attack_rate": 0.012, "cannibalism_handling_time": 0.05,
        "K_j": 40.0, "juv_density_mortality": 2e-4,
        "fecundity_multiplier": 1.0, "male_bias_multiplier": 1.0
    }