# frymodel.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .model import SimpleModel
from .cannibalism_models import RickerCannibalism


@dataclass
class FryModel(RickerCannibalism):
    """
    Four-stage model:
      Fry (F) -> Juvenile (J) -> Adult Female (AF) / Adult Male (AM)

    Features
    --------
    - Env-modulated survival per stage (Environment.stage_survival_mult).
    - Env-modulated maturation (Environment.maturation_mult).
    - Fry-only cannibalism by default via Ricker-style survival:
        S_fry = exp(-c * P_eff * E) with refuge/encounter in E.
      (Non-fry cannibalism hooks exist in the base class but are ~0 by default.)
    - Juvenile bottleneck:
        * optional capacity (K_j) applied to F->J inflow
        * optional quadratic density loss on juveniles
    - Reproduction: recruits per adult FEMALE per step,
        fecundity_per_female * fecundity_multiplier * env.maturation_mult("adult","fry")
      Optionally scaled by a feeding schedule if attached.

    Notes
    -----
    - Default parameters preserve earlier behavior unless you set tank shape,
      feeding schedule, or change susceptibility/weights in the base class.
    """

    # If True, apply (very small by default) non-fry cannibalism multipliers too.
    # Defaults keep behavior identical to "fry-only".
    enable_nonfry_cannibalism: bool = False

    def step(self, s: SimpleModel.State) -> SimpleModel.State:
        env = self.env.clamp()

        # ---------- Survival (stage-specific) ----------
        F_surv  = max(0.0, s.fry)          * self.fry_survival_probability       * env.stage_survival_mult("fry")
        J_surv  = max(0.0, s.juveniles)    * self.juvenile_survival_probability  * env.stage_survival_mult("juvenile")
        AF_surv = max(0.0, s.adult_female) * self.adult_f_survival_probability   * env.stage_survival_mult("adult")
        AM_surv = max(0.0, s.adult_male)   * self.adult_m_survival_probability   * env.stage_survival_mult("adult")

        # ---------- Cannibalism survival (fry required; others optional & tiny by default) ----------
        S_fry = self.cannibalism_survival("fry", s)

        if self.enable_nonfry_cannibalism:
            # These default to ~1.0 because susceptibilities are tiny in the base class.
            S_juv = self.cannibalism_survival("juvenile", s)
            S_af  = self.cannibalism_survival("adult_female", s)
            S_am  = self.cannibalism_survival("adult_male", s)
        else:
            S_juv = S_af = S_am = 1.0

        # ---------- Maturation (env-adjusted) ----------
        pf = max(0.0, min(1.0, self.mat_fj_scale * env.maturation_mult("fry", "juvenile")))
        pj = max(0.0, min(1.0, self.mat_ja_scale * env.maturation_mult("juvenile", "adult")))

        # ---------- F -> J (fry survivors gated by cannibalism) ----------
        flow_FJ = F_surv * S_fry * pf

        # Juvenile capacity limits inflow; density mortality acts on stock
        cap_mult = 1.0
        if self.K_j is not None and self.K_j > 0.0:
            # cap based on current (survived) juvenile level
            cap_mult = max(0.0, 1.0 - (J_surv / float(self.K_j)))
        flow_FJ *= cap_mult

        # Optional extra juvenile density loss
        J_density_loss = self.juv_density_mortality * (J_surv ** 2)

        # Update juveniles before J->A; include optional (tiny) juv cannibalism
        J_next = max(0.0, J_surv * S_juv + flow_FJ - J_density_loss)

        # ---------- J -> A split by sex ratio at adulthood ----------
        pjF = pj * (1.0 - self.sex_ratio_at_birth_male)
        pjM = pj * self.sex_ratio_at_birth_male
        flow_JA_F = J_next * pjF
        flow_JA_M = J_next * pjM

        # Adults next; include optional (tiny) adult cannibalism multipliers
        AF_next = max(0.0, AF_surv * S_af + flow_JA_F)
        AM_next = max(0.0, AM_surv * S_am + flow_JA_M)

        # ---------- Recruitment (per adult FEMALE) ----------
        # Base fecundity scaling with environment (temperature etc.)
        fec_env  = env.maturation_mult("adult", "fry")
        fec_base = max(0.0, self.fecundity_per_female * self.fecundity_multiplier * fec_env)

        # Optional: time-varying feeding schedule multiplier (safe if not attached)
        fec_scale = 1.0
        if getattr(self, "feeding_schedule", None) is not None:
            try:
                step_idx = int(self.provenance.get("step_index", 0))
                fec_scale = float(self.feeding_schedule.fecundity_multiplier_at(step_idx))
            except Exception:
                fec_scale = 1.0

        fec = max(0.0, fec_base * fec_scale)
        recruits = AF_surv * fec  # births enter as fry (sex split applied at J->A, not at birth)

        # Fry next: survivors that did not progress + new recruits
        F_next = max(0.0, F_surv * (1.0 - pf) + recruits)

        # ---------- Diagnostics ----------
        self.provenance.setdefault("last_step", {})
        self.provenance["last_step"].update({
            "S_fry": float(S_fry),
            "S_juv": float(S_juv),
            "S_af": float(S_af),
            "S_am": float(S_am),
            "pf": float(pf),
            "pj": float(pj),
            "flow_FJ": float(flow_FJ),
            "flow_JA_F": float(flow_JA_F),
            "flow_JA_M": float(flow_JA_M),
            "cap_mult": float(cap_mult),
            "J_density_loss": float(J_density_loss),
            "fec_env": float(fec_env),
            "fec_scale": float(fec_scale),
            "fec": float(fec),
        })

        return SimpleModel.State(F_next, J_next, AF_next, AM_next)