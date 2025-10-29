# cannibalism_models.py
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any
from .model import SimpleModel


# ----------------------------
# Helpers
# ----------------------------

def _clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _effective_predators(
    state: SimpleModel.State,
    kernel: Literal["male", "female", "both"],
    w_male: float,
    w_female: float,
) -> float:
    """
    Effective predator pressure from adults.
    - kernel selects which adult pools contribute.
    - w_male / w_female are weights (defaults make females ~0 by default).
    """
    am = max(0.0, float(state.adult_male))
    af = max(0.0, float(state.adult_female))
    if kernel == "male":
        return w_male * am
    if kernel == "female":
        return w_female * af
    # both
    return w_male * am + w_female * af


def _shape_attack_multiplier(owner: SimpleModel, predators_visible: float) -> float:
    """
    Optional: scale attack rate slightly based on tank geometry/density.
    Bounded in [0.5, 2.0] to avoid blowups. No effect if tank_shape is None.
    """
    # If no shape info on the owning model, do nothing.
    tank_shape = getattr(owner, "tank_shape", None)
    if tank_shape is None:
        return 1.0

    # Loose, conservative mapping (length, width, depth attributes expected).
    L = float(getattr(tank_shape, "length_m", 0.0) or 0.0)
    W = float(getattr(tank_shape, "width_m", 0.0) or 0.0)
    D = float(getattr(tank_shape, "depth_m", 0.0) or 0.0)
    if L <= 0.0 or W <= 0.0 or D <= 0.0:
        return 1.0

    footprint = L * W
    boundary  = 2.0 * (L + W)
    aspect    = (max(L, W) / min(L, W)) if min(L, W) > 0 else 1.0
    density   = predators_visible / max(1e-9, footprint)

    density_factor  = 1.0 + 0.02  * min(200.0, density)
    boundary_factor = 1.0 + 0.001 * min(2000.0, boundary)
    aspect_factor   = 1.0 + 0.10  * max(0.0, aspect - 1.0)

    scale = density_factor * boundary_factor * aspect_factor
    return float(min(2.0, max(0.5, scale)))


# ----------------------------
# Multi-stage Holling cannibalism with refuge (default behavior ~fry-only)
# ----------------------------

class RefugeHolling(SimpleModel):
    """
    Refuge-aware Holling functional response, generalized to any stage.

    Provides:
      - cannibalism_survival(stage, state): per-stage survival multiplier in (0, 1].
        Stages: "fry", "juvenile", "adult_male", "adult_female".
      - fry_cannibalism_survival(state): convenience wrapper for existing code paths.

    Defaults make non-fry predation *extremely rare*, nearly nil on adult females.
    You can override the susceptibility or predator weights to explore lab vs wild.
    """

    # --- Predator kernel weights (who does the eating) ---
    # Defaults: males are the predators; females are ~0 (rare in wild).
    predator_weight_male: float = 1.0
    predator_weight_female: float = 0.02

    # --- Prey susceptibility (who gets eaten) ---
    # Defaults: fry = 1.0 (target), others very small; adult females ~impossible.
    prey_susc_fry: float = 1.0
    prey_susc_juvenile: float = 0.02
    prey_susc_adult_male: float = 0.005
    prey_susc_adult_female: float = 0.001

    # Optional global scale to uniformly raise/lower all non-fry susceptibilities
    non_fry_susc_scale: float = 1.0

    # If True, clamp survival to [eps, 1], avoiding exact 0 even at extreme params.
    clamp_eps: float = 1e-6

    def _stage_susceptibility(self, stage: str) -> float:
        if stage == "fry":
            return float(self.prey_susc_fry)
        if stage == "juvenile":
            return float(self.prey_susc_juvenile) * float(self.non_fry_susc_scale)
        if stage == "adult_male":
            return float(self.prey_susc_adult_male) * float(self.non_fry_susc_scale)
        if stage == "adult_female":
            return float(self.prey_susc_adult_female) * float(self.non_fry_susc_scale)
        # unknown stage -> no cannibalism effect
        return 0.0

    def cannibalism_survival(self, stage: str, state: SimpleModel.State) -> float:
        """
        Per-stage cannibalism survival (0,1]; applies Holling Type II/III with
        environment encounter scaling and optional tank-shape tweak.

        NOTE: This returns *only* the cannibalism term; multiply by your
              stage_survival and other terms in your step() as usual.
        """
        # If susceptibility is (nearly) zero, skip the math for speed.
        susc = max(0.0, float(self._stage_susceptibility(stage)))
        if susc <= 0.0:
            return 1.0

        # Effective predator pressure
        kernel = getattr(self, "cannibalism_kernel", "male")
        w_m = float(getattr(self, "predator_weight_male", 1.0))
        w_f = float(getattr(self, "predator_weight_female", 0.02))
        P   = _effective_predators(state, kernel, w_m, w_f)

        # Encounter multiplier from environment (refuge, predator pressure)
        E = self.env.encounter_mult()

        # Attack/handling; allow tank geometry to tweak attack rate slightly
        a0 = float(getattr(self, "cannibalism_attack_rate", 0.01))
        h0 = float(getattr(self, "cannibalism_handling_time", 0.0))

        # Refuge reduces attack; increases effective handling
        r  = float(self.env.refuge_index)
        a  = max(1e-12, a0 * (1.0 - 0.8 * r))
        h  = max(0.0,    h0 * (1.0 + 1.5 * r))

        # Optional: tank shape multiplier
        a *= _shape_attack_multiplier(self, predators_visible=max(1.0, P))

        # Holling type
        ctype = str(getattr(self, "cannibalism_type", "H2")).upper()
        if ctype == "H3":
            denom = 1.0 + a * (P ** 2) * E * h
            attack = (a * (P ** 2) * E) / denom
        else:  # "H2" by default
            denom = 1.0 + a * P * E * h
            attack = (a * P * E) / denom

        # Convert to survival via exponential kill; scale by susceptibility
        S = float(math.exp(-susc * attack))
        if self.clamp_eps > 0.0:
            S = float(min(1.0, max(self.clamp_eps, S)))

        # provenance (last step)
        self.provenance.setdefault("last_step", {})
        self.provenance["last_step"].update({
            f"cann_{stage}_predators": float(P),
            f"cann_{stage}_attack_eff": float(a),
            f"cann_{stage}_handling_eff": float(h),
            f"cann_{stage}_E": float(E),
            f"cann_{stage}_susc": float(susc),
            f"cann_{stage}_survival": float(S),
            "cann_type": ctype,
        })
        return S

    # Backwards-compatible helper used by existing FryModel code
    def fry_cannibalism_survival(self, state: SimpleModel.State) -> float:
        return self.cannibalism_survival("fry", state)


# ----------------------------
# Lightweight alternatives: Exponential / Beverton–Holt forms
# (These apply to FRY by default but can be called per-stage if desired)
# ----------------------------

class RickerCannibalism(SimpleModel):
    """
    Survival(stage) = exp(- c * susc(stage) * P_eff * E)
    Where P_eff combines adult males/females with optional weights and kernel.
    Default susceptibilities mirror RefugeHolling: fry high, others ~0.
    """
    predator_weight_male: float = 1.0
    predator_weight_female: float = 0.02

    prey_susc_fry: float = 1.0
    prey_susc_juvenile: float = 0.02
    prey_susc_adult_male: float = 0.005
    prey_susc_adult_female: float = 0.001
    non_fry_susc_scale: float = 1.0
    clamp_eps: float = 1e-6

    strength_cannibalism: float = 0.01  # c

    def _susc(self, stage: str) -> float:
        if stage == "fry":
            return float(self.prey_susc_fry)
        if stage == "juvenile":
            return float(self.prey_susc_juvenile) * float(self.non_fry_susc_scale)
        if stage == "adult_male":
            return float(self.prey_susc_adult_male) * float(self.non_fry_susc_scale)
        if stage == "adult_female":
            return float(self.prey_susc_adult_female) * float(self.non_fry_susc_scale)
        return 0.0

    def cannibalism_survival(self, stage: str, state: SimpleModel.State) -> float:
        susc = max(0.0, self._susc(stage))
        if susc <= 0.0:
            return 1.0
        kernel = getattr(self, "cannibalism_kernel", "male")
        P = _effective_predators(
            state,
            kernel,
            float(getattr(self, "predator_weight_male", 1.0)),
            float(getattr(self, "predator_weight_female", 0.02)),
        )
        E = self.env.encounter_mult()
        c = float(getattr(self, "strength_cannibalism", 0.01))
        S = float(math.exp(-c * susc * P * E))
        if self.clamp_eps > 0.0:
            S = float(min(1.0, max(self.clamp_eps, S)))
        return S

    # Convenience for fry-only paths
    def fry_cannibalism_survival(self, state: SimpleModel.State) -> float:
        return self.cannibalism_survival("fry", state)


class BevertonHoltCannibalism(SimpleModel):
    """
    Survival(stage) = 1 / (1 + c * susc(stage) * P_eff * E)
    Saturates at high predator pressure; defaults keep non-fry impacts tiny.
    """
    predator_weight_male: float = 1.0
    predator_weight_female: float = 0.02

    prey_susc_fry: float = 1.0
    prey_susc_juvenile: float = 0.02
    prey_susc_adult_male: float = 0.005
    prey_susc_adult_female: float = 0.001
    non_fry_susc_scale: float = 1.0
    clamp_eps: float = 1e-6

    strength_cannibalism: float = 0.01  # c

    def _susc(self, stage: str) -> float:
        if stage == "fry":
            return float(self.prey_susc_fry)
        if stage == "juvenile":
            return float(self.prey_susc_juvenile) * float(self.non_fry_susc_scale)
        if stage == "adult_male":
            return float(self.prey_susc_adult_male) * float(self.non_fry_susc_scale)
        if stage == "adult_female":
            return float(self.prey_susc_adult_female) * float(self.non_fry_susc_scale)
        return 0.0

    def cannibalism_survival(self, stage: str, state: SimpleModel.State) -> float:
        susc = max(0.0, self._susc(stage))
        if susc <= 0.0:
            return 1.0
        kernel = getattr(self, "cannibalism_kernel", "male")
        P = _effective_predators(
            state,
            kernel,
            float(getattr(self, "predator_weight_male", 1.0)),
            float(getattr(self, "predator_weight_female", 0.02)),
        )
        E = self.env.encounter_mult()
        c = float(getattr(self, "strength_cannibalism", 0.01))
        denom = 1.0 + c * susc * max(0.0, P) * E
        S = float(1.0 / max(1e-12, denom))
        if self.clamp_eps > 0.0:
            S = float(min(1.0, max(self.clamp_eps, S)))
        return S

    def fry_cannibalism_survival(self, state: SimpleModel.State) -> float:
        return self.cannibalism_survival("fry", state)