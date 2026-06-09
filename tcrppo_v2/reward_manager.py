"""Reward manager: combine all scorers into a single reward signal."""

from collections import deque
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RewardManager:
    """Combine multiple scorer signals into reward.

    R_t = w1 * affinity - w2 * R_decoy + w3 * P_naturalness + w4 * P_diversity

    For tFold V3.4 scorers, affinity is the pre-sigmoid binding logit
    (-gate_logit), not sigmoid(-gate_logit). Larger is better; very low or
    failed tFold samples receive strongly negative affinity.

    Naturalness/diversity scorers return non-positive penalty terms, where
    0.0 means acceptable and negative values mean worse. Adding those terms
    ensures auxiliary objectives can only reduce reward; affinity controls the
    reward upside.
    """

    def __init__(
        self,
        affinity_scorer=None,
        decoy_scorer=None,
        naturalness_scorer=None,
        diversity_scorer=None,
        w_affinity: float = 1.0,
        w_decoy: float = 0.8,
        w_naturalness: float = 0.5,
        w_diversity: float = 0.2,
        norm_window: int = 10000,
        norm_warmup: int = 1000,
        use_delta_reward: bool = True,
        reward_mode: str = "v2_full",
        n_contrast_decoys: int = 4,
        convex_alpha: float = 3.0,
        contrastive_agg: str = "mean",
        ood_threshold: float = 0.15,
        ood_penalty_weight: float = 1.0,
        ood_penalty_mode: str = "soft",
        w_absolute_affinity: float = 0.25,
        affinity_ref_logit: float = -4.5,
        delta_amp_thresholds: Optional[List[float]] = None,
        delta_amp_slopes: Optional[List[float]] = None,
        delta_negative_scale: float = 1.0,
        delta_deadband: float = 0.0,
        delta_deadband_penalty: float = 0.0,
        delta_amp_clip: Optional[float] = None,
        naturalness_gate: bool = False,
        naturalness_gate_threshold: float = 0.0,
        affinity_guard_logit: float = -3.0,
        affinity_guard_tolerance: float = 0.35,
        affinity_guard_weight: float = 4.0,
        specificity_margin: float = 1.0,
        decoy_drop_weight: float = 0.25,
        target_gate_temperature: float = 0.25,
        decoy_affinity_ceiling: float = -4.5,
        target_surplus_cap: float = 2.0,
        target_decoy_gate_logit: float = -2.0,
        target_pass_bonus: float = 1.0,
        decoy_affinity_center: float = -3.0,
        decoy_fixed_tiers: Optional[List[str]] = None,
        decoy_k_per_tier: int = 1,
        hybrid_delta_weight: float = 0.25,
        # Curriculum climbing parameters
        curriculum_gates: Optional[List[float]] = None,
        curriculum_bonuses: Optional[List[float]] = None,
        gap_margin: float = 1.0,
        decoy_activation_threshold: float = 0.5,
        w_gap: float = 1.0,
        soft_gate_affinity: float = 0.5,
        soft_gate_temperature: float = 0.12,
        soft_decoy_min_gate: float = 0.02,
        decoy_topk: int = 2,
        w_decoy_mean: float = 0.15,
        decoy_margin_clip: Optional[float] = 3.0,
        preserve_high_init_threshold: Optional[float] = None,
        preserve_high_init_tolerance: float = 0.10,
        preserve_high_init_weight: float = 0.0,
        improve_low_init_threshold: Optional[float] = None,
        improve_low_init_min_delta: float = 0.05,
        improve_low_init_weight: float = 0.0,
        improve_low_init_max_penalty: Optional[float] = None,
        pretrain_naturalness_only: bool = False,
    ):
        self.pretrain_naturalness_only = pretrain_naturalness_only
        self.affinity_scorer = affinity_scorer
        self.decoy_scorer = decoy_scorer
        self.naturalness_scorer = naturalness_scorer
        self.diversity_scorer = diversity_scorer

        self.weights = {
            "affinity": w_affinity,
            "decoy": w_decoy,
            "naturalness": w_naturalness,
            "diversity": w_diversity,
        }
        self.use_delta_reward = use_delta_reward
        self.reward_mode = reward_mode
        self.n_contrast_decoys = n_contrast_decoys
        self.convex_alpha = convex_alpha
        self.contrastive_agg = contrastive_agg  # "mean" or "max"
        self.ood_threshold = ood_threshold
        self.ood_penalty_weight = ood_penalty_weight
        self.ood_penalty_mode = ood_penalty_mode  # "soft" or "hard"
        self.w_absolute_affinity = w_absolute_affinity
        self.affinity_ref_logit = affinity_ref_logit
        self.delta_amp_thresholds = self._normalize_thresholds(
            delta_amp_thresholds or [0.5, 1.0, 2.0]
        )
        self.delta_amp_slopes = self._normalize_slopes(
            delta_amp_slopes or [1.0, 2.0, 4.0, 8.0],
            len(self.delta_amp_thresholds) + 1,
        )
        self.delta_negative_scale = float(delta_negative_scale)
        self.delta_deadband = max(0.0, float(delta_deadband))
        self.delta_deadband_penalty = float(delta_deadband_penalty)
        self.delta_amp_clip = None if delta_amp_clip is None else abs(float(delta_amp_clip))
        self.naturalness_gate = naturalness_gate
        self.naturalness_gate_threshold = naturalness_gate_threshold
        self.naturalness_scorer = naturalness_scorer
        self.affinity_guard_logit = float(affinity_guard_logit)
        self.affinity_guard_tolerance = max(0.0, float(affinity_guard_tolerance))
        self.affinity_guard_weight = max(0.0, float(affinity_guard_weight))
        self.specificity_margin = max(0.0, float(specificity_margin))
        self.decoy_drop_weight = max(0.0, float(decoy_drop_weight))
        self.target_gate_temperature = max(1e-6, float(target_gate_temperature))
        self.decoy_affinity_ceiling = float(decoy_affinity_ceiling)
        self.target_surplus_cap = max(0.0, float(target_surplus_cap))
        self.target_decoy_gate_logit = float(target_decoy_gate_logit)
        self.target_pass_bonus = float(target_pass_bonus)
        self.decoy_affinity_center = float(decoy_affinity_center)
        # Allow empty decoy_fixed_tiers to truly disable decoy sampling
        if decoy_fixed_tiers is None:
            self.decoy_fixed_tiers = ["A", "B"]
        else:
            self.decoy_fixed_tiers = list(decoy_fixed_tiers)
        # Allow decoy_k_per_tier=0 to disable decoy sampling
        self.decoy_k_per_tier = int(decoy_k_per_tier) if decoy_k_per_tier is not None else 1
        self.hybrid_delta_weight = max(0.0, float(hybrid_delta_weight))
        # Curriculum climbing
        self.curriculum_gates = curriculum_gates or [-0.2, 0.0, 0.2, 0.5]
        self.curriculum_bonuses = curriculum_bonuses or [0.5, 1.0, 1.5, 2.0]
        self.gap_margin = max(0.0, float(gap_margin))
        self.decoy_activation_threshold = float(decoy_activation_threshold)
        self.weights["gap"] = max(0.0, float(w_gap))
        self.soft_gate_affinity = float(soft_gate_affinity)
        self.soft_gate_temperature = max(1e-6, float(soft_gate_temperature))
        self.soft_decoy_min_gate = max(0.0, float(soft_decoy_min_gate))
        self.decoy_topk = max(1, int(decoy_topk))
        self.weights["decoy_mean"] = max(0.0, float(w_decoy_mean))
        self.decoy_margin_clip = None if decoy_margin_clip is None else abs(float(decoy_margin_clip))
        self.preserve_high_init_threshold = (
            None if preserve_high_init_threshold is None else float(preserve_high_init_threshold)
        )
        self.preserve_high_init_tolerance = max(0.0, float(preserve_high_init_tolerance))
        self.preserve_high_init_weight = max(0.0, float(preserve_high_init_weight))
        self.improve_low_init_threshold = (
            None if improve_low_init_threshold is None else float(improve_low_init_threshold)
        )
        self.improve_low_init_min_delta = max(0.0, float(improve_low_init_min_delta))
        self.improve_low_init_weight = max(0.0, float(improve_low_init_weight))
        self.improve_low_init_max_penalty = (
            None if improve_low_init_max_penalty is None else max(0.0, float(improve_low_init_max_penalty))
        )
        # OOD stats tracking
        self._ood_triggered = 0
        self._ood_total = 0

    @staticmethod
    def _normalize_thresholds(thresholds: List[float]) -> List[float]:
        """Return sorted, positive, unique amplification thresholds."""
        cleaned = sorted({float(t) for t in thresholds if float(t) > 0.0})
        return cleaned

    @staticmethod
    def _normalize_slopes(slopes: List[float], expected_len: int) -> List[float]:
        """Return one positive slope per amplification segment."""
        cleaned = [max(0.0, float(s)) for s in slopes]
        if not cleaned:
            cleaned = [1.0]
        while len(cleaned) < expected_len:
            cleaned.append(cleaned[-1])
        return cleaned[:expected_len]

    @staticmethod
    def _finite_scalar(value: float, default: float, label: str) -> float:
        """Clamp non-finite scorer outputs before they reach PPO."""
        try:
            value = float(value)
        except Exception:
            logger.warning("Non-scalar reward component %s=%r; using %.3f", label, value, default)
            return default
        if not np.isfinite(value):
            logger.warning("Non-finite reward component %s=%r; using %.3f", label, value, default)
            return default
        return value

    @staticmethod
    def _sigmoid_scalar(value: float) -> float:
        """Numerically stable scalar sigmoid."""
        value = float(value)
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def _amplify_delta(self, delta: float) -> float:
        """Piecewise-linear convex shaping for terminal affinity deltas."""
        delta = float(delta)
        if delta < 0.0:
            shaped = self.delta_negative_scale * delta
        else:
            shaped = 0.0
            prev = 0.0
            for threshold, slope in zip(self.delta_amp_thresholds, self.delta_amp_slopes):
                if delta <= prev:
                    break
                span = max(0.0, min(delta, threshold) - prev)
                shaped += span * slope
                if delta <= threshold:
                    prev = delta
                    break
                prev = threshold

            if delta > prev:
                shaped += (delta - prev) * self.delta_amp_slopes[-1]
            if 0.0 <= delta < self.delta_deadband:
                shaped += self.delta_deadband_penalty

        if self.delta_amp_clip is not None:
            shaped = float(np.clip(shaped, -self.delta_amp_clip, self.delta_amp_clip))
        return shaped

    def _apply_naturalness_gate(
        self,
        total: float,
        nat_score: float,
        components: Dict[str, float],
    ) -> float:
        """Optionally reject positive reward for sequences failing naturalness.
        
        If naturalness_gate is enabled and the naturalness penalty is below threshold,
        the affinity reward component is banned (set to 0), keeping only the naturalness penalty.
        """
        if self.naturalness_gate and nat_score < self.naturalness_gate_threshold:
            components["naturalness_gate"] = 1.0
            # Ban affinity reward, keep only naturalness penalty
            return self.weights["naturalness"] * nat_score
        components["naturalness_gate"] = 0.0
        return total

    def _target_guard_floor(self, initial_affinity: float) -> float:
        """Return the per-episode target affinity floor for guarded specificity."""
        return max(
            self.affinity_guard_logit,
            float(initial_affinity) - self.affinity_guard_tolerance,
        )

    def _target_gate(self, affinity: float, floor: float) -> float:
        """Smoothly unlock decoy-drop bonus only when target affinity is preserved."""
        return self._sigmoid_scalar((float(affinity) - float(floor)) / self.target_gate_temperature)

    def _guarded_decoy_reward(
        self,
        aff_score: float,
        initial_affinity: float,
        aff_delta: float,
        nat_score: float,
        div_score: float,
        components: Dict[str, float],
    ) -> float:
        """Target-first specificity reward.

        The target term preserves/improves target affinity. The decoy term
        penalizes decoys that are too close to the target logit, and only grants
        a decoy-lowering bonus when the target affinity remains above its guard.
        """
        floor = self._target_guard_floor(initial_affinity)
        target_shortfall = max(0.0, floor - aff_score)
        target_gate = self._target_gate(aff_score, floor)

        if components.get("decoy_n", 0.0) <= 0.0:
            decoy_final = 0.0
            decoy_initial = 0.0
            decoy_delta = 0.0
            margin_violation = 0.0
            decoy_drop = 0.0
        else:
            agg_key = "max" if self.contrastive_agg == "max" else "mean"
            decoy_final = components.get(f"decoy_final_{agg_key}", components.get("decoy_final_mean", 0.0))
            decoy_initial = components.get(f"decoy_initial_{agg_key}", components.get("decoy_initial_mean", 0.0))
            decoy_delta = decoy_final - decoy_initial

            # Larger logits bind more strongly. A decoy should sit at least
            # specificity_margin below the target logit.
            margin_violation = max(0.0, decoy_final - aff_score + self.specificity_margin)
            decoy_drop = max(0.0, decoy_initial - decoy_final)
        guard_penalty = self.affinity_guard_weight * target_shortfall * (1.0 + target_shortfall)
        target_abs_centered = aff_score - floor
        target_term = aff_delta + self.w_absolute_affinity * target_abs_centered
        decoy_term = (
            -self.weights["decoy"] * margin_violation
            + self.decoy_drop_weight * target_gate * decoy_drop
        )

        components["affinity_guard_floor"] = floor
        components["affinity_guard_shortfall"] = target_shortfall
        components["target_gate"] = target_gate
        components["affinity_abs_centered"] = target_abs_centered
        components["decoy_margin_violation"] = margin_violation
        components["decoy_drop"] = decoy_drop
        components["decoy_delta"] = decoy_delta
        components["target_guard_penalty"] = guard_penalty
        components["target_term"] = target_term
        components["decoy_term"] = decoy_term

        return (
            self.weights["affinity"] * target_term
            - guard_penalty
            + decoy_term
            + self.weights["naturalness"] * nat_score
            + self.weights["diversity"] * div_score
        )

    def _absolute_specificity_reward(
        self,
        aff_score: float,
        nat_score: float,
        div_score: float,
        components: Dict[str, float],
    ) -> float:
        """Absolute binding/non-binding reward.

        The target is rewarded for clearing an absolute binding floor. Decoys
        are penalized for exceeding an absolute non-binding ceiling. This
        avoids rewarding a merely relative target-vs-decoy ordering when both
        logits are biologically unacceptable.
        """
        target_floor = self.affinity_guard_logit
        target_surplus = aff_score - target_floor
        target_shortfall = max(0.0, -target_surplus)
        target_satisfied = max(0.0, min(target_surplus, self.target_surplus_cap))
        target_penalty = self.affinity_guard_weight * target_shortfall * (1.0 + target_shortfall)

        if components.get("decoy_n", 0.0) <= 0.0:
            decoy_final = 0.0
            decoy_violation = 0.0
        else:
            agg_key = "max" if self.contrastive_agg == "max" else "mean"
            decoy_final = components.get(f"decoy_final_{agg_key}", components.get("decoy_final_mean", 0.0))
            decoy_violation = max(0.0, decoy_final - self.decoy_affinity_ceiling)
        decoy_penalty = self.weights["decoy"] * decoy_violation * (1.0 + decoy_violation)

        components["target_affinity_floor"] = target_floor
        components["target_affinity_surplus"] = target_surplus
        components["target_affinity_satisfied"] = target_satisfied
        components["target_affinity_shortfall"] = target_shortfall
        components["target_affinity_penalty"] = target_penalty
        components["decoy_affinity_ceiling"] = self.decoy_affinity_ceiling
        components["decoy_affinity_for_penalty"] = decoy_final
        components["decoy_affinity_violation"] = decoy_violation
        components["decoy_affinity_penalty"] = decoy_penalty

        return (
            self.weights["affinity"] * target_satisfied
            - target_penalty
            - decoy_penalty
            + self.weights["naturalness"] * nat_score
            + self.weights["diversity"] * div_score
        )

    def _simple_target_gated_decoy_reward(
        self,
        aff_score: float,
        nat_score: float,
        div_score: float,
        components: Dict[str, float],
        aff_delta: float = None,
    ) -> float:
        """Simple target-first absolute specificity reward.

        Below the target gate, reward is pure target affinity. Once target
        affinity reaches the gate, a pass bonus is added and the mean fixed
        A/B decoy affinity is centered at ``decoy_affinity_center``.

        When use_delta_reward is True, the affinity component uses aff_delta
        (improvement over initial) instead of absolute aff_score. The gate
        still uses aff_score (absolute) since we care about absolute quality.
        """
        target_gate = self.target_decoy_gate_logit
        target_passed = 1.0 if aff_score >= target_gate else 0.0
        decoy_n = components.get("decoy_n", 0.0)
        decoy_final = components.get("decoy_final_mean", 0.0)
        decoy_active = target_passed and decoy_n > 0.0
        decoy_centered = decoy_final - self.decoy_affinity_center if decoy_active else 0.0
        decoy_term = -self.weights["decoy"] * decoy_centered if target_passed else 0.0

        components["target_decoy_gate_logit"] = target_gate
        components["target_decoy_gate_passed"] = target_passed
        components["target_pass_bonus"] = self.target_pass_bonus if target_passed else 0.0
        components["target_affinity_shortfall"] = max(0.0, target_gate - aff_score)
        components["target_affinity_satisfied"] = max(0.0, aff_score - target_gate)
        components["decoy_affinity_center"] = self.decoy_affinity_center
        components["decoy_affinity_for_penalty"] = decoy_final
        components["decoy_affinity_violation"] = max(0.0, decoy_centered) if decoy_active else 0.0
        components["decoy_centered"] = decoy_centered
        components["decoy_term"] = decoy_term

        aff_for_reward = aff_delta if (self.use_delta_reward and aff_delta is not None) else aff_score
        total = self.weights["affinity"] * aff_for_reward
        if target_passed:
            total += self.target_pass_bonus + decoy_term
        preserve_penalty = 0.0
        init_affinity = components.get("initial_affinity")
        if (
            self.preserve_high_init_threshold is not None
            and self.preserve_high_init_weight > 0.0
            and init_affinity is not None
            and float(init_affinity) >= self.preserve_high_init_threshold
        ):
            preserve_floor = float(init_affinity) - self.preserve_high_init_tolerance
            preserve_shortfall = max(0.0, preserve_floor - aff_score)
            preserve_penalty = self.preserve_high_init_weight * preserve_shortfall
            total -= preserve_penalty
        components["preserve_high_init_penalty"] = preserve_penalty
        improve_penalty = 0.0
        if (
            self.improve_low_init_threshold is not None
            and self.improve_low_init_weight > 0.0
            and init_affinity is not None
            and float(init_affinity) < self.improve_low_init_threshold
        ):
            aff_delta = float(components.get("affinity_step_delta", aff_score - float(init_affinity)))
            improve_shortfall = max(0.0, self.improve_low_init_min_delta - aff_delta)
            improve_penalty = self.improve_low_init_weight * improve_shortfall
            if self.improve_low_init_max_penalty is not None:
                improve_penalty = min(improve_penalty, self.improve_low_init_max_penalty)
            total -= improve_penalty
        components["improve_low_init_penalty"] = improve_penalty
        total += self.weights["naturalness"] * nat_score
        total += self.weights["diversity"] * div_score
        
        # Apply naturalness gating if enabled
        total = self._apply_naturalness_gate(total, nat_score, components)
        
        return total

    def _curriculum_climbing_reward(
        self,
        aff_score: float,
        nat_score: float,
        div_score: float,
        components: Dict[str, float],
    ) -> float:
        """Curriculum climbing reward: 逐步爬升到 A > 0.5，然后优化 specificity.

        Phase 1-4 (A < decoy_activation_threshold):
            Reward = w_affinity * A + stage_bonus + auxiliaries
            stage_bonus 根据 A 跨过 curriculum_gates 中的哪些阈值累计

        Phase 5 (A >= decoy_activation_threshold):
            gap = A - DecA_mean
            gap_bonus = w_gap * max(0, gap - gap_margin)  # 只奖励不惩罚
            Reward = w_affinity * A + max_stage_bonus + gap_bonus + auxiliaries

        Key: gap < margin 时 gap_bonus = 0，不会因 decoy 而惩罚模型。
        """
        # 计算 stage bonus: 取 aff_score 跨过的最高 gate 对应的 bonus
        stage_bonus = 0.0
        stage_idx = 0
        for i, (gate, bonus) in enumerate(zip(self.curriculum_gates, self.curriculum_bonuses)):
            if aff_score >= gate:
                stage_bonus = bonus
                stage_idx = i + 1

        components["curriculum_stage"] = float(stage_idx)
        components["stage_bonus"] = stage_bonus

        # Base reward (target affinity + stage bonus)
        total = self.weights["affinity"] * aff_score + stage_bonus

        # Phase 5: specificity gap (仅在 A >= threshold 时激活)
        decoy_n = components.get("decoy_n", 0.0)
        decoy_final = components.get("decoy_final_mean", 0.0)
        gap_active = (aff_score >= self.decoy_activation_threshold) and (decoy_n > 0.0)

        if gap_active:
            gap = aff_score - decoy_final
            gap_bonus = self.weights.get("gap", 1.0) * max(0.0, gap - self.gap_margin)
            total += gap_bonus
            components["gap"] = gap
            components["gap_bonus"] = gap_bonus
            components["gap_active"] = 1.0
            components["decoy_affinity_for_penalty"] = decoy_final
        else:
            components["gap"] = 0.0
            components["gap_bonus"] = 0.0
            components["gap_active"] = 0.0
            components["decoy_affinity_for_penalty"] = decoy_final

        # Auxiliary terms (naturalness, diversity)
        total += self.weights["naturalness"] * nat_score
        total += self.weights["diversity"] * div_score
        return total

    def _smooth_gate_reward(
        self,
        aff_score: float,
        nat_score: float,
        div_score: float,
        components: Dict[str, float],
    ) -> float:
        """Smooth gate reward with sigmoid transition.
        
        Instead of hard gate (if A > gate: bonus else 0), use smooth sigmoid:
            bonus_factor = sigmoid((A - gate) / temperature)
            reward = A + bonus * bonus_factor + auxiliaries
        
        This provides smooth gradient and encourages continuous improvement.
        
        Args:
            aff_score: Target affinity
            nat_score: Naturalness score
            div_score: Diversity score
            components: Dict to store reward components
            
        Config params (from __init__):
            target_decoy_gate_logit: Gate threshold (e.g., -1.0)
            target_pass_bonus: Maximum bonus (e.g., 1.0)
            target_gate_temperature: Sigmoid steepness (e.g., 0.5)
        """
        gate = self.target_decoy_gate_logit
        bonus = self.target_pass_bonus
        temperature = self.target_gate_temperature
        
        # Compute sigmoid bonus factor
        # sigmoid((A - gate) / temp) gives smooth 0→1 transition around gate
        bonus_factor = self._sigmoid_scalar((aff_score - gate) / temperature)
        smooth_bonus = bonus * bonus_factor
        
        # Store components for logging
        components["smooth_gate"] = gate
        components["smooth_bonus_factor"] = bonus_factor
        components["smooth_bonus"] = smooth_bonus
        components["affinity_above_gate"] = max(0.0, aff_score - gate)
        
        # Total reward
        total = (
            self.weights["affinity"] * aff_score
            + smooth_bonus
            + self.weights["naturalness"] * nat_score
            + self.weights["diversity"] * div_score
        )
        return total

    def _soft_target_decoy_gap_reward(
        self,
        aff_score: float,
        nat_score: float,
        div_score: float,
        components: Dict[str, float],
    ) -> float:
        """Soft target-first reward for one-peptide decoy finetuning.

        Target affinity is always rewarded. Decoy specificity is smoothly
        activated around ``soft_gate_affinity`` and uses the strongest decoys
        more heavily than the raw mean, so negative/easy decoys do not dominate.
        """
        gate = self._sigmoid_scalar(
            (aff_score - self.soft_gate_affinity) / self.soft_gate_temperature
        )
        decoy_n = components.get("decoy_n", 0.0)
        decoy_mean = components.get("decoy_final_mean", 0.0)
        decoy_topk_mean = components.get("decoy_final_topk_mean", components.get("decoy_final_max", decoy_mean))

        if decoy_n <= 0.0:
            margin_topk = 0.0
            margin_mean = 0.0
            decoy_term = 0.0
        else:
            margin_topk = aff_score - decoy_topk_mean
            margin_mean = aff_score - decoy_mean
            if self.decoy_margin_clip is not None:
                margin_topk = float(np.clip(margin_topk, -self.decoy_margin_clip, self.decoy_margin_clip))
                margin_mean = float(np.clip(margin_mean, -self.decoy_margin_clip, self.decoy_margin_clip))
            decoy_term = gate * (
                self.weights["decoy"] * margin_topk
                + self.weights["decoy_mean"] * margin_mean
            )

        components["soft_gate_affinity"] = self.soft_gate_affinity
        components["soft_gate_temperature"] = self.soft_gate_temperature
        components["target_decoy_soft_gate"] = gate
        components["decoy_final_topk_mean"] = decoy_topk_mean
        components["target_decoy_margin_topk"] = margin_topk
        components["target_decoy_margin_mean"] = margin_mean
        components["decoy_term"] = decoy_term
        components["decoy_affinity_for_penalty"] = decoy_topk_mean
        components["decoy_affinity_violation"] = max(0.0, decoy_topk_mean - aff_score)

        return (
            self.weights["affinity"] * aff_score
            + decoy_term
            + self.weights["naturalness"] * nat_score
            + self.weights["diversity"] * div_score
        )

    def _hybrid_abs_delta_gated_decoy_reward(
        self,
        aff_score: float,
        aff_step_delta: float,
        nat_score: float,
        div_score: float,
        components: Dict[str, float],
    ) -> float:
        """Trace29 absolute reward plus a small positive improvement bonus."""
        target_gate = self.target_decoy_gate_logit
        target_passed = 1.0 if aff_score >= target_gate else 0.0
        positive_delta = max(0.0, aff_step_delta)

        decoy_n = components.get("decoy_n", 0.0)
        decoy_final = components.get("decoy_final_mean", 0.0)
        decoy_active = target_passed and decoy_n > 0.0
        decoy_centered = decoy_final - self.decoy_affinity_center if decoy_active else 0.0
        decoy_term = -self.weights["decoy"] * decoy_centered if target_passed else 0.0

        components["target_decoy_gate_logit"] = target_gate
        components["target_decoy_gate_passed"] = target_passed
        components["target_pass_bonus"] = self.target_pass_bonus if target_passed else 0.0
        components["target_affinity_shortfall"] = max(0.0, target_gate - aff_score)
        components["target_affinity_satisfied"] = max(0.0, aff_score - target_gate)
        components["positive_affinity_delta"] = positive_delta
        components["hybrid_delta_weight"] = self.hybrid_delta_weight
        components["hybrid_delta_bonus"] = self.hybrid_delta_weight * positive_delta
        components["decoy_affinity_center"] = self.decoy_affinity_center
        components["decoy_affinity_for_penalty"] = decoy_final
        components["decoy_affinity_violation"] = max(0.0, decoy_centered) if decoy_active else 0.0
        components["decoy_centered"] = decoy_centered
        components["decoy_term"] = decoy_term

        total = self.weights["affinity"] * aff_score
        total += self.hybrid_delta_weight * positive_delta
        if target_passed:
            total += self.target_pass_bonus + decoy_term
        total += self.weights["naturalness"] * nat_score
        total += self.weights["diversity"] * div_score
        return total

    def compute_reward(
        self,
        tcr: str,
        peptide: str,
        initial_affinity: float = 0.0,
        initial_tcr: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute combined reward for a TCR-peptide pair."""
        target = target or peptide
        components = {}

        # ---- Pretrain mode: reward = raw naturalness score (0~1) ----
        if self.pretrain_naturalness_only and self.naturalness_scorer is not None:
            raw = self.naturalness_scorer.score_raw(tcr)
            components["naturalness_raw_combined"] = raw
            components["affinity_raw"] = 0.0
            components["initial_affinity"] = 0.0
            components["affinity_step_delta"] = 0.0
            components["affinity_sigmoid"] = 0.5
            components["decoy_raw"] = 0.0
            components["naturalness_raw"] = raw
            components["diversity_raw"] = 0.0
            return raw, components

        # Affinity
        if self.affinity_scorer is not None and self.reward_mode != "disabled":
            # For OOD penalty mode, use score_batch to get uncertainty
            if self.reward_mode == "v1_ergo_ood_penalty":
                if hasattr(self.affinity_scorer, 'score_batch'):
                    scores, confidences = self.affinity_scorer.score_batch([tcr], [peptide])
                    aff_score = scores[0]
                    confidence = confidences[0]
                    uncertainty = 1.0 - confidence
                    components["uncertainty"] = uncertainty

                    # Apply OOD penalty
                    self._ood_total += 1
                    if self.ood_penalty_mode == "soft":
                        # Soft penalty: only penalize the excess beyond threshold
                        if uncertainty > self.ood_threshold:
                            penalty = (uncertainty - self.ood_threshold) * self.ood_penalty_weight
                            aff_score = aff_score - penalty
                            components["ood_penalty"] = penalty
                            self._ood_triggered += 1
                        else:
                            components["ood_penalty"] = 0.0
                    else:  # hard
                        # Hard penalty: penalize full uncertainty
                        if uncertainty > self.ood_threshold:
                            penalty = uncertainty * self.ood_penalty_weight
                            aff_score = aff_score - penalty
                            components["ood_penalty"] = penalty
                            self._ood_triggered += 1
                        else:
                            components["ood_penalty"] = 0.0
                else:
                    # Fallback if scorer doesn't support score_batch
                    aff_score, _ = self.affinity_scorer.score(tcr, peptide)
                    components["uncertainty"] = 0.0
                    components["ood_penalty"] = 0.0
            elif hasattr(self.affinity_scorer, 'score_batch_fast'):
                preds = self.affinity_scorer.score_batch_fast([tcr], [peptide])
                aff_score = preds[0]
            else:
                aff_score, _ = self.affinity_scorer.score(tcr, peptide)
            aff_score = self._finite_scalar(aff_score, 0.0, "affinity_raw")
            initial_affinity = self._finite_scalar(initial_affinity, 0.0, "initial_affinity")
            aff_step_delta = aff_score - initial_affinity
            aff_delta = aff_step_delta if self.use_delta_reward else aff_score
            components["affinity_raw"] = aff_score
            components["initial_affinity"] = initial_affinity
            components["affinity_delta"] = aff_delta
            components["affinity_step_delta"] = aff_step_delta
            components["affinity_sigmoid"] = self._sigmoid_scalar(aff_score)
        else:
            aff_score = 0.0
            aff_delta = 0.0
            aff_step_delta = 0.0
            components["affinity_raw"] = 0.0
            components["initial_affinity"] = 0.0
            components["affinity_step_delta"] = 0.0
            components["affinity_sigmoid"] = 0.5

        # Decoy penalty — always computed (no frequency gating)
        if (self.decoy_scorer is not None
                and self.reward_mode in ("v2_full", "v2_decoy_only", "raw_decoy", "raw_multi_penalty", "threshold_penalty")):
            decoy_score, _ = self.decoy_scorer.score(tcr, peptide, target=target)
            decoy_score = self._finite_scalar(decoy_score, 0.0, "decoy_raw")
            components["decoy_raw"] = decoy_score
        else:
            decoy_score = 0.0
            components["decoy_raw"] = 0.0

        decoy_delta = 0.0
        if self.reward_mode in (
            "v2_delta_minus_decoy",
            "v2_target_guarded_decoy",
            "v2_absolute_specificity",
        ):
            decoy_delta = self._compute_decoy_delta(
                final_tcr=tcr,
                initial_tcr=initial_tcr,
                target=target,
                components=components,
            )
        elif self.reward_mode in ("v2_simple_target_gated_decoy", "v2_hybrid_abs_delta_gated_decoy"):
            if aff_score >= self.target_decoy_gate_logit:
                self._compute_decoy_final(
                    final_tcr=tcr,
                    target=target,
                    components=components,
                )
            else:
                self._finish_decoy_final([], components)
        elif self.reward_mode == "v2_soft_target_decoy_gap":
            soft_gate = self._sigmoid_scalar(
                (aff_score - self.soft_gate_affinity) / self.soft_gate_temperature
            )
            if soft_gate >= self.soft_decoy_min_gate:
                self._compute_decoy_final(
                    final_tcr=tcr,
                    target=target,
                    components=components,
                )
            else:
                self._finish_decoy_final([], components)
            components["target_decoy_soft_gate"] = soft_gate
        elif self.reward_mode == "v2_curriculum_climbing":
            if aff_score >= self.decoy_activation_threshold:
                self._compute_decoy_final(
                    final_tcr=tcr,
                    target=target,
                    components=components,
                )
            else:
                self._finish_decoy_final([], components)

        # Naturalness penalty — always computed (no frequency gating)
        if (self.naturalness_scorer is not None
                and self.reward_mode in (
                    "v2_full", "v2_no_decoy", "v2_no_decoy_delta",
                    "v2_no_decoy_delta_calibrated", "v2_no_decoy_sigmoid_delta",
                    "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty",
                    "contrastive_ergo", "tfold_delta_amplified",
                    "v2_delta_minus_decoy", "v2_target_guarded_decoy", "v2_absolute_specificity",
                    "v2_simple_target_gated_decoy", "v2_curriculum_climbing",
                    "v2_hybrid_abs_delta_gated_decoy", "v2_soft_target_decoy_gap",
                )):
            nat_score, _ = self.naturalness_scorer.score(tcr)
            nat_score = self._finite_scalar(nat_score, 0.0, "naturalness_raw")
            components["naturalness_raw"] = nat_score
        else:
            nat_score = 0.0
            components["naturalness_raw"] = 0.0

        # Diversity penalty
        if (self.diversity_scorer is not None
                and self.reward_mode in (
                    "v2_full", "v2_no_decoy", "v2_no_decoy_delta",
                    "v2_no_decoy_delta_calibrated", "v2_no_decoy_sigmoid_delta",
                    "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty",
                    "v2_delta_minus_decoy", "v2_target_guarded_decoy", "v2_absolute_specificity",
                    "v2_simple_target_gated_decoy", "v2_curriculum_climbing",
                    "v2_hybrid_abs_delta_gated_decoy", "v2_soft_target_decoy_gap",
                )):
            div_score, _ = self.diversity_scorer.score(tcr)
            div_score = self._finite_scalar(div_score, 0.0, "diversity_raw")
            components["diversity_raw"] = div_score
        else:
            div_score = 0.0
            components["diversity_raw"] = 0.0

        aff_weight = self.weights["affinity"]

        # Compute total reward — ALL modes use raw scores, NO z-norm
        if self.reward_mode == "v1_ergo_only":
            total = aff_weight * aff_score
        elif self.reward_mode == "v1_ergo_ood_penalty":
            # OOD penalty already applied to aff_score above
            total = aff_weight * aff_score
        elif self.reward_mode == "v1_ergo_convex":
            # Convex reward: ERGO^alpha — amplifies gradient at high scores
            # alpha=3: 0.5→0.125, 0.7→0.343, 0.8→0.512, 0.9→0.729, 0.95→0.857
            total = aff_weight * (aff_score ** self.convex_alpha)
        elif self.reward_mode == "v1_ergo_squared":
            total = aff_weight * (aff_score ** 2)
        elif self.reward_mode == "v1_ergo_delta":
            total = aff_weight * aff_delta
        elif self.reward_mode == "v1_ergo_stepwise":
            total = aff_weight * aff_score
        elif self.reward_mode == "tfold_stepwise":
            total = aff_weight * (aff_score - initial_affinity)
        elif self.reward_mode == "tfold_delta_calibrated":
            abs_centered = aff_score - self.affinity_ref_logit
            components["affinity_abs_centered"] = abs_centered
            total = aff_weight * aff_delta + self.w_absolute_affinity * abs_centered
        elif self.reward_mode == "tfold_delta_amplified":
            amplified_delta = self._amplify_delta(aff_step_delta)
            components["affinity_delta_amplified"] = amplified_delta
            components["affinity_delta_amplification"] = amplified_delta - aff_step_delta
            total = (aff_weight * amplified_delta
                    + self.weights["naturalness"] * nat_score
                    + self.weights["diversity"] * div_score)
            total = self._apply_naturalness_gate(total, nat_score, components)
        elif self.reward_mode == "v1_ergo_shaped":
            # Shaped reward: intermediate steps get 0.1 * delta, terminal gets full score
            # Note: is_terminal flag must be passed via kwargs
            total = aff_weight * aff_score  # Default to full score (will be overridden in env for intermediate)
        elif self.reward_mode in ("v2_decoy_only", "raw_decoy"):
            total = aff_weight * aff_score - self.weights["decoy"] * decoy_score
        elif self.reward_mode in ("raw_multi_penalty", "v2_full", "v2_no_decoy", "v2_no_curriculum"):
            total = (aff_weight * aff_score
                    - self.weights["decoy"] * decoy_score
                    + self.weights["naturalness"] * nat_score
                    + self.weights["diversity"] * div_score)
        elif self.reward_mode == "v2_no_decoy_delta":
            total = (aff_weight * aff_delta
                    + self.weights["naturalness"] * nat_score
                    + self.weights["diversity"] * div_score)
        elif self.reward_mode == "v2_delta_minus_decoy":
            total = (aff_weight * aff_delta
                    - self.weights["decoy"] * decoy_delta
                    + self.weights["naturalness"] * nat_score
                    + self.weights["diversity"] * div_score)
        elif self.reward_mode == "v2_target_guarded_decoy":
            total = self._guarded_decoy_reward(
                aff_score=aff_score,
                initial_affinity=initial_affinity,
                aff_delta=aff_delta,
                nat_score=nat_score,
                div_score=div_score,
                components=components,
            )
        elif self.reward_mode == "v2_absolute_specificity":
            total = self._absolute_specificity_reward(
                aff_score=aff_score,
                nat_score=nat_score,
                div_score=div_score,
                components=components,
            )
        elif self.reward_mode in ("v2_simple_target_gated_decoy", "v2_hybrid_abs_delta_gated_decoy"):
            total = self._simple_target_gated_decoy_reward(
                aff_score=aff_score,
                nat_score=nat_score,
                div_score=div_score,
                components=components,
                aff_delta=aff_delta,
            )
        elif self.reward_mode == "v2_hybrid_abs_delta_gated_decoy":
            total = self._hybrid_abs_delta_gated_decoy_reward(
                aff_score=aff_score,
                aff_step_delta=aff_step_delta,
                nat_score=nat_score,
                div_score=div_score,
                components=components,
            )
        elif self.reward_mode == "v2_curriculum_climbing":
            total = self._curriculum_climbing_reward(
                aff_score=aff_score,
                nat_score=nat_score,
                div_score=div_score,
                components=components,
            )
        elif self.reward_mode == "v2_soft_target_decoy_gap":
            total = self._soft_target_decoy_gap_reward(
                aff_score=aff_score,
                nat_score=nat_score,
                div_score=div_score,
                components=components,
            )
        elif self.reward_mode == "v2_no_decoy_delta_calibrated":
            # Delta alone can reward "less bad" samples. Add an absolute logit
            # term centered on known-binder calibration so poor terminal binders
            # remain unattractive even if they improved from a worse seed.
            abs_centered = aff_score - self.affinity_ref_logit
            components["affinity_abs_centered"] = abs_centered
            total = (aff_weight * aff_delta
                    + self.w_absolute_affinity * abs_centered
                    + self.weights["naturalness"] * nat_score
                    + self.weights["diversity"] * div_score)
        elif self.reward_mode == "v2_no_decoy_sigmoid_delta":
            sig_aff = self._sigmoid_scalar(aff_score)
            sig_init = self._sigmoid_scalar(initial_affinity)
            components["affinity_sigmoid"] = sig_aff
            components["affinity_sigmoid_delta"] = sig_aff - sig_init
            total = (aff_weight * (sig_aff - sig_init)
                    + self.weights["naturalness"] * nat_score
                    + self.weights["diversity"] * div_score)
        elif self.reward_mode == "threshold_penalty":
            if aff_score < 0.5:
                total = aff_weight * aff_score
            else:
                total = (aff_weight * aff_score
                        - self.weights["decoy"] * decoy_score
                        + self.weights["naturalness"] * nat_score
                        + self.weights["diversity"] * div_score)
        elif self.reward_mode == "contrastive_ergo":
            # Contrastive: reward = ERGO(target) - agg(ERGO(decoys))
            # agg = "mean" (original) or "max" (worst-case specificity)
            if self.decoy_scorer is not None:
                decoy_peptides = self.decoy_scorer.sample_decoys(target, k=self.n_contrast_decoys)
                if not decoy_peptides:
                    total = aff_weight * aff_score  # No decoys available, fallback
                else:
                    # Score TCR against decoys using ERGO
                    if hasattr(self.affinity_scorer, 'score_batch_fast'):
                        decoy_scores = self.affinity_scorer.score_batch_fast(
                            [tcr] * len(decoy_peptides), decoy_peptides
                        )
                    else:
                        decoy_scores = [self.affinity_scorer.score(tcr, d)[0] for d in decoy_peptides]
                    if self.contrastive_agg == "max":
                        agg_decoy_score = float(np.max(decoy_scores))
                    else:
                        agg_decoy_score = float(np.mean(decoy_scores))
                    # Apply convex transformation if alpha != 1
                    if self.convex_alpha != 1.0 and self.convex_alpha > 0:
                        total = aff_weight * (aff_score ** self.convex_alpha - agg_decoy_score ** self.convex_alpha)
                    else:
                        total = aff_weight * (aff_score - agg_decoy_score)
                    components["decoy_mean"] = float(np.mean(decoy_scores))
                    components["decoy_max"] = float(np.max(decoy_scores))
                    components["contrast_margin"] = total
                # Add naturalness bonus if available
                if self.naturalness_scorer is not None and self.weights["naturalness"] > 0:
                    total = total + self.weights["naturalness"] * nat_score
            else:
                total = aff_weight * aff_score  # Fallback if no decoy_scorer
        else:
            total = aff_weight * aff_score  # Fallback: raw affinity

        total = self._finite_scalar(total, 0.0, "reward_total")
        components["total"] = total
        return total, components

    def compute_reward_batch(
        self,
        tcrs: List[str],
        peptides: List[str],
        initial_affinities: List[float],
        initial_tcrs: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """Compute rewards for a batch of TCR-peptide pairs."""
        n = len(tcrs)
        targets = targets or peptides
        initial_tcrs = initial_tcrs or [None] * n

        # ---- Pretrain mode: reward = raw naturalness score (0~1) ----
        if self.pretrain_naturalness_only and self.naturalness_scorer is not None:
            if hasattr(self.naturalness_scorer, 'score_raw_batch'):
                raws = self.naturalness_scorer.score_raw_batch(tcrs)
            else:
                raws = [self.naturalness_scorer.score_raw(t) for t in tcrs]
            all_rewards = raws
            all_components = []
            for raw in raws:
                all_components.append({
                    "naturalness_raw_combined": raw,
                    "affinity_raw": 0.0,
                    "initial_affinity": 0.0,
                    "affinity_step_delta": 0.0,
                    "affinity_sigmoid": 0.5,
                    "decoy_raw": 0.0,
                    "naturalness_raw": raw,
                    "diversity_raw": 0.0,
                })
            return all_rewards, all_components

        all_rewards = []
        all_components = []

        # Batch affinity scoring
        if self.affinity_scorer is not None and self.reward_mode != "disabled":
            # For OOD penalty mode, use score_batch to get uncertainty
            if self.reward_mode == "v1_ergo_ood_penalty":
                if hasattr(self.affinity_scorer, 'score_batch'):
                    aff_scores, confidences = self.affinity_scorer.score_batch(tcrs, peptides)
                    uncertainties = [1.0 - c for c in confidences]
                else:
                    # Fallback
                    aff_scores = []
                    uncertainties = [0.0] * n
                    for tcr, pep in zip(tcrs, peptides):
                        s, _ = self.affinity_scorer.score(tcr, pep)
                        aff_scores.append(s)
            elif hasattr(self.affinity_scorer, 'score_batch_fast'):
                aff_scores = self.affinity_scorer.score_batch_fast(tcrs, peptides)
                uncertainties = [0.0] * n
            else:
                aff_scores = []
                uncertainties = [0.0] * n
                for tcr, pep in zip(tcrs, peptides):
                    s, _ = self.affinity_scorer.score(tcr, pep)
                    aff_scores.append(s)
        else:
            aff_scores = [0.0] * n
            uncertainties = [0.0] * n

        precomputed_decoy_components: List[Optional[Dict[str, float]]] = [None] * n
        precomputed_decoy_deltas = [0.0] * n
        if self.reward_mode in ("v2_delta_minus_decoy", "v2_target_guarded_decoy", "v2_absolute_specificity"):
            decoy_lists = [self._sample_delta_decoys(targets[i]) for i in range(n)]
            flat_final_tcrs: List[str] = []
            flat_initial_tcrs: List[str] = []
            flat_decoy_peptides: List[str] = []
            offsets: List[Tuple[int, int]] = []
            for i, decoy_peptides in enumerate(decoy_lists):
                start = len(flat_decoy_peptides)
                init_tcr = initial_tcrs[i] or tcrs[i]
                flat_final_tcrs.extend([tcrs[i]] * len(decoy_peptides))
                flat_initial_tcrs.extend([init_tcr] * len(decoy_peptides))
                flat_decoy_peptides.extend(decoy_peptides)
                offsets.append((start, len(flat_decoy_peptides)))

            final_decoy_scores = self._score_affinity_batch(flat_final_tcrs, flat_decoy_peptides)
            initial_decoy_scores = self._score_affinity_batch(flat_initial_tcrs, flat_decoy_peptides)
            for i, (start, end) in enumerate(offsets):
                decoy_components: Dict[str, float] = {
                    "decoy_delta_missing_initial_tcr": 0.0 if initial_tcrs[i] else 1.0,
                    "decoy_n": float(end - start),
                }
                precomputed_decoy_deltas[i] = self._finish_decoy_delta(
                    final_decoy_scores[start:end],
                    initial_decoy_scores[start:end],
                    decoy_components,
                )
                precomputed_decoy_components[i] = decoy_components
        elif self.reward_mode == "v2_simple_target_gated_decoy":
            decoy_lists = [
                self._sample_simple_decoys(targets[i])
                if self._finite_scalar(aff_scores[i], 0.0, "affinity_raw") >= self.target_decoy_gate_logit
                else []
                for i in range(n)
            ]
            flat_final_tcrs: List[str] = []
            flat_decoy_peptides: List[str] = []
            offsets: List[Tuple[int, int]] = []
            for i, decoy_peptides in enumerate(decoy_lists):
                start = len(flat_decoy_peptides)
                flat_final_tcrs.extend([tcrs[i]] * len(decoy_peptides))
                flat_decoy_peptides.extend(decoy_peptides)
                offsets.append((start, len(flat_decoy_peptides)))

            final_decoy_scores = self._score_affinity_batch(flat_final_tcrs, flat_decoy_peptides)
            for i, (start, end) in enumerate(offsets):
                decoy_components = {"decoy_n": float(end - start)}
                self._finish_decoy_final(
                    final_decoy_scores[start:end],
                    decoy_components,
                )
                precomputed_decoy_components[i] = decoy_components
        elif self.reward_mode == "v2_curriculum_climbing":
            decoy_lists = [
                self._sample_simple_decoys(targets[i])
                if self._finite_scalar(aff_scores[i], 0.0, "affinity_raw") >= self.decoy_activation_threshold
                else []
                for i in range(n)
            ]
            flat_final_tcrs: List[str] = []
            flat_decoy_peptides: List[str] = []
            offsets: List[Tuple[int, int]] = []
            for i, decoy_peptides in enumerate(decoy_lists):
                start = len(flat_decoy_peptides)
                flat_final_tcrs.extend([tcrs[i]] * len(decoy_peptides))
                flat_decoy_peptides.extend(decoy_peptides)
                offsets.append((start, len(flat_decoy_peptides)))

            final_decoy_scores = self._score_affinity_batch(flat_final_tcrs, flat_decoy_peptides)
            for i, (start, end) in enumerate(offsets):
                decoy_components = {"decoy_n": float(end - start)}
                self._finish_decoy_final(
                    final_decoy_scores[start:end],
                    decoy_components,
                )
                precomputed_decoy_components[i] = decoy_components
        elif self.reward_mode == "v2_soft_target_decoy_gap":
            decoy_lists = []
            for i in range(n):
                aff_i = self._finite_scalar(aff_scores[i], 0.0, "affinity_raw")
                gate_i = self._sigmoid_scalar(
                    (aff_i - self.soft_gate_affinity) / self.soft_gate_temperature
                )
                decoy_lists.append(
                    self._sample_simple_decoys(targets[i])
                    if gate_i >= self.soft_decoy_min_gate
                    else []
                )
            flat_final_tcrs: List[str] = []
            flat_decoy_peptides: List[str] = []
            offsets: List[Tuple[int, int]] = []
            for i, decoy_peptides in enumerate(decoy_lists):
                start = len(flat_decoy_peptides)
                flat_final_tcrs.extend([tcrs[i]] * len(decoy_peptides))
                flat_decoy_peptides.extend(decoy_peptides)
                offsets.append((start, len(flat_decoy_peptides)))

            final_decoy_scores = self._score_affinity_batch(flat_final_tcrs, flat_decoy_peptides)
            for i, (start, end) in enumerate(offsets):
                decoy_components = {"decoy_n": float(end - start)}
                self._finish_decoy_final(
                    final_decoy_scores[start:end],
                    decoy_components,
                )
                precomputed_decoy_components[i] = decoy_components

        # Process each sample — always compute all scorers (no frequency gating)
        for i in range(n):
            components = {}
            aff_score = self._finite_scalar(aff_scores[i], 0.0, "affinity_raw")
            init_aff = self._finite_scalar(initial_affinities[i], 0.0, "initial_affinity")

            # Apply OOD penalty if in OOD mode
            if self.reward_mode == "v1_ergo_ood_penalty":
                uncertainty = self._finite_scalar(uncertainties[i], 0.0, "uncertainty")
                components["uncertainty"] = uncertainty
                self._ood_total += 1

                if self.ood_penalty_mode == "soft":
                    if uncertainty > self.ood_threshold:
                        penalty = (uncertainty - self.ood_threshold) * self.ood_penalty_weight
                        aff_score = aff_score - penalty
                        components["ood_penalty"] = penalty
                        self._ood_triggered += 1
                    else:
                        components["ood_penalty"] = 0.0
                else:  # hard
                    if uncertainty > self.ood_threshold:
                        penalty = uncertainty * self.ood_penalty_weight
                        aff_score = aff_score - penalty
                        components["ood_penalty"] = penalty
                        self._ood_triggered += 1
                    else:
                        components["ood_penalty"] = 0.0

            aff_step_delta = aff_score - init_aff
            aff_delta = aff_step_delta if self.use_delta_reward else aff_score
            components["affinity_raw"] = aff_score
            components["initial_affinity"] = init_aff
            components["affinity_delta"] = aff_delta
            components["affinity_step_delta"] = aff_step_delta
            components["affinity_sigmoid"] = self._sigmoid_scalar(aff_score)

            # Decoy — always computed for every sample
            if (self.decoy_scorer is not None
                    and self.reward_mode in ("v2_full", "v2_decoy_only", "raw_decoy", "raw_multi_penalty", "threshold_penalty")):
                decoy_score, _ = self.decoy_scorer.score(tcrs[i], peptides[i], target=targets[i])
                decoy_score = self._finite_scalar(decoy_score, 0.0, "decoy_raw")
                components["decoy_raw"] = decoy_score
            else:
                decoy_score = 0.0
                components["decoy_raw"] = 0.0

            decoy_delta = 0.0
            if self.reward_mode in ("v2_delta_minus_decoy", "v2_target_guarded_decoy", "v2_absolute_specificity"):
                if precomputed_decoy_components[i] is not None:
                    components.update(precomputed_decoy_components[i])
                    decoy_delta = precomputed_decoy_deltas[i]
                else:
                    decoy_delta = self._compute_decoy_delta(
                        final_tcr=tcrs[i],
                        initial_tcr=initial_tcrs[i],
                        target=targets[i],
                        components=components,
                    )
            elif self.reward_mode in ("v2_simple_target_gated_decoy", "v2_hybrid_abs_delta_gated_decoy"):
                if precomputed_decoy_components[i] is not None:
                    components.update(precomputed_decoy_components[i])
                elif aff_score < self.target_decoy_gate_logit:
                    self._finish_decoy_final([], components)
                else:
                    self._compute_decoy_final(
                        final_tcr=tcrs[i],
                        target=targets[i],
                        components=components,
                    )
            elif self.reward_mode == "v2_curriculum_climbing":
                if precomputed_decoy_components[i] is not None:
                    components.update(precomputed_decoy_components[i])
                elif aff_score < self.decoy_activation_threshold:
                    self._finish_decoy_final([], components)
                else:
                    self._compute_decoy_final(
                        final_tcr=tcrs[i],
                        target=targets[i],
                        components=components,
                    )
            elif self.reward_mode == "v2_soft_target_decoy_gap":
                soft_gate = self._sigmoid_scalar(
                    (aff_score - self.soft_gate_affinity) / self.soft_gate_temperature
                )
                if precomputed_decoy_components[i] is not None:
                    components.update(precomputed_decoy_components[i])
                elif soft_gate < self.soft_decoy_min_gate:
                    self._finish_decoy_final([], components)
                else:
                    self._compute_decoy_final(
                        final_tcr=tcrs[i],
                        target=targets[i],
                        components=components,
                    )
                components["target_decoy_soft_gate"] = soft_gate

            # Naturalness — always computed for every sample
            if (self.naturalness_scorer is not None
                    and self.reward_mode in (
                        "v2_full", "v2_no_decoy", "v2_no_decoy_delta",
                        "v2_no_decoy_delta_calibrated", "v2_no_decoy_sigmoid_delta",
                        "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty",
                        "contrastive_ergo", "tfold_delta_amplified",
                        "v2_delta_minus_decoy", "v2_target_guarded_decoy", "v2_absolute_specificity",
                        "v2_simple_target_gated_decoy", "v2_curriculum_climbing",
                        "v2_hybrid_abs_delta_gated_decoy", "v2_soft_target_decoy_gap",
                    )):
                nat_score, _ = self.naturalness_scorer.score(tcrs[i])
                nat_score = self._finite_scalar(nat_score, 0.0, "naturalness_raw")
                components["naturalness_raw"] = nat_score
            else:
                nat_score = 0.0
                components["naturalness_raw"] = 0.0

            # Diversity
            if (self.diversity_scorer is not None
                    and self.reward_mode in (
                        "v2_full", "v2_no_decoy", "v2_no_decoy_delta",
                        "v2_no_decoy_delta_calibrated", "v2_no_decoy_sigmoid_delta",
                        "v2_no_curriculum", "raw_multi_penalty", "threshold_penalty",
                        "v2_delta_minus_decoy", "v2_target_guarded_decoy", "v2_absolute_specificity",
                        "v2_simple_target_gated_decoy", "v2_curriculum_climbing",
                        "v2_hybrid_abs_delta_gated_decoy", "v2_soft_target_decoy_gap",
                    )):
                div_score, _ = self.diversity_scorer.score(tcrs[i])
                div_score = self._finite_scalar(div_score, 0.0, "diversity_raw")
                components["diversity_raw"] = div_score
            else:
                div_score = 0.0
                components["diversity_raw"] = 0.0

            aff_weight = self.weights["affinity"]

            # Total reward — NO z-norm, all raw
            if self.reward_mode == "v1_ergo_only":
                total = aff_weight * aff_score
            elif self.reward_mode == "v1_ergo_ood_penalty":
                # OOD penalty already applied to aff_score above
                total = aff_weight * aff_score
            elif self.reward_mode == "v1_ergo_convex":
                total = aff_weight * (aff_score ** self.convex_alpha)
            elif self.reward_mode == "v1_ergo_squared":
                total = aff_weight * (aff_score ** 2)
            elif self.reward_mode == "v1_ergo_delta":
                total = aff_weight * aff_delta
            elif self.reward_mode == "v1_ergo_stepwise":
                total = aff_weight * aff_score
            elif self.reward_mode == "tfold_stepwise":
                total = aff_weight * (aff_score - init_aff)
            elif self.reward_mode == "tfold_delta_calibrated":
                abs_centered = aff_score - self.affinity_ref_logit
                components["affinity_abs_centered"] = abs_centered
                total = aff_weight * aff_delta + self.w_absolute_affinity * abs_centered
            elif self.reward_mode == "tfold_delta_amplified":
                amplified_delta = self._amplify_delta(aff_step_delta)
                components["affinity_delta_amplified"] = amplified_delta
                components["affinity_delta_amplification"] = amplified_delta - aff_step_delta
                total = (aff_weight * amplified_delta
                        + self.weights["naturalness"] * nat_score
                        + self.weights["diversity"] * div_score)
                total = self._apply_naturalness_gate(total, nat_score, components)
            elif self.reward_mode == "v1_ergo_shaped":
                total = aff_weight * aff_score  # env handles shaped vs terminal split
            elif self.reward_mode in ("v2_decoy_only", "raw_decoy"):
                total = aff_weight * aff_score - self.weights["decoy"] * decoy_score
            elif self.reward_mode in ("raw_multi_penalty", "v2_full", "v2_no_decoy", "v2_no_curriculum"):
                total = (aff_weight * aff_score
                        - self.weights["decoy"] * decoy_score
                        + self.weights["naturalness"] * nat_score
                        + self.weights["diversity"] * div_score)
            elif self.reward_mode == "v2_no_decoy_delta":
                total = (aff_weight * aff_delta
                        + self.weights["naturalness"] * nat_score
                        + self.weights["diversity"] * div_score)
            elif self.reward_mode == "v2_delta_minus_decoy":
                total = (aff_weight * aff_delta
                        - self.weights["decoy"] * decoy_delta
                        + self.weights["naturalness"] * nat_score
                        + self.weights["diversity"] * div_score)
            elif self.reward_mode == "v2_target_guarded_decoy":
                total = self._guarded_decoy_reward(
                    aff_score=aff_score,
                    initial_affinity=init_aff,
                    aff_delta=aff_delta,
                    nat_score=nat_score,
                    div_score=div_score,
                    components=components,
                )
            elif self.reward_mode == "v2_absolute_specificity":
                total = self._absolute_specificity_reward(
                    aff_score=aff_score,
                    nat_score=nat_score,
                    div_score=div_score,
                    components=components,
                )
            elif self.reward_mode == "v2_simple_target_gated_decoy":
                total = self._simple_target_gated_decoy_reward(
                    aff_score=aff_score,
                    nat_score=nat_score,
                    div_score=div_score,
                    components=components,
                    aff_delta=aff_delta,
                )
            elif self.reward_mode == "v2_hybrid_abs_delta_gated_decoy":
                total = self._hybrid_abs_delta_gated_decoy_reward(
                    aff_score=aff_score,
                    aff_step_delta=aff_step_delta,
                    nat_score=nat_score,
                    div_score=div_score,
                    components=components,
                )
            elif self.reward_mode == "v2_curriculum_climbing":
                total = self._curriculum_climbing_reward(
                    aff_score=aff_score,
                    nat_score=nat_score,
                    div_score=div_score,
                    components=components,
                )
            elif self.reward_mode == "v2_smooth_gate_reward":
                total = self._smooth_gate_reward(
                    aff_score=aff_score,
                    nat_score=nat_score,
                    div_score=div_score,
                    components=components,
                )
            elif self.reward_mode == "v2_soft_target_decoy_gap":
                total = self._soft_target_decoy_gap_reward(
                    aff_score=aff_score,
                    nat_score=nat_score,
                    div_score=div_score,
                    components=components,
                )
            elif self.reward_mode == "v2_no_decoy_delta_calibrated":
                abs_centered = aff_score - self.affinity_ref_logit
                components["affinity_abs_centered"] = abs_centered
                total = (aff_weight * aff_delta
                        + self.w_absolute_affinity * abs_centered
                        + self.weights["naturalness"] * nat_score
                        + self.weights["diversity"] * div_score)
            elif self.reward_mode == "v2_no_decoy_sigmoid_delta":
                sig_aff = self._sigmoid_scalar(aff_score)
                sig_init = self._sigmoid_scalar(init_aff)
                components["affinity_sigmoid"] = sig_aff
                components["affinity_sigmoid_delta"] = sig_aff - sig_init
                total = (aff_weight * (sig_aff - sig_init)
                        + self.weights["naturalness"] * nat_score
                        + self.weights["diversity"] * div_score)
            elif self.reward_mode == "threshold_penalty":
                if aff_score < 0.5:
                    total = aff_weight * aff_score
                else:
                    total = (aff_weight * aff_score
                            - self.weights["decoy"] * decoy_score
                            + self.weights["naturalness"] * nat_score
                            + self.weights["diversity"] * div_score)
            elif self.reward_mode == "contrastive_ergo":
                if self.decoy_scorer is not None:
                    decoy_peptides = self.decoy_scorer.sample_decoys(targets[i], k=self.n_contrast_decoys)
                    if not decoy_peptides:
                        total = aff_weight * aff_score
                        components["decoy_mean"] = 0.0
                        components["decoy_max"] = 0.0
                        components["contrast_margin"] = aff_score
                    else:
                        if hasattr(self.affinity_scorer, 'score_batch_fast'):
                            decoy_scores = self.affinity_scorer.score_batch_fast(
                                [tcrs[i]] * len(decoy_peptides), decoy_peptides
                            )
                        else:
                            decoy_scores = [self.affinity_scorer.score(tcrs[i], d)[0] for d in decoy_peptides]
                        if self.contrastive_agg == "max":
                            agg_decoy_score = self._finite_scalar(np.max(decoy_scores), 0.0, "decoy_max")
                        else:
                            agg_decoy_score = self._finite_scalar(np.mean(decoy_scores), 0.0, "decoy_mean")
                        total = aff_weight * (aff_score - agg_decoy_score)
                        components["decoy_mean"] = self._finite_scalar(np.mean(decoy_scores), 0.0, "decoy_mean")
                        components["decoy_max"] = self._finite_scalar(np.max(decoy_scores), 0.0, "decoy_max")
                        components["contrast_margin"] = total
                    # Add naturalness bonus if available
                    if self.naturalness_scorer is not None and self.weights["naturalness"] > 0:
                        total = total + self.weights["naturalness"] * nat_score
                else:
                    total = aff_weight * aff_score
            else:
                total = aff_weight * aff_score  # Fallback

            total = self._finite_scalar(total, 0.0, "reward_total")
            components["total"] = total
            all_rewards.append(total)
            all_components.append(components)

        return all_rewards, all_components

    def _sample_delta_decoys(self, target: str) -> List[str]:
        """Sample unique decoy peptides for delta-minus-decoy reward."""
        if self.decoy_scorer is None:
            return []
        k = self.n_contrast_decoys
        if k <= 0:
            k = getattr(self.decoy_scorer, "K", 0)
        if k <= 0 or not hasattr(self.decoy_scorer, "sample_decoys"):
            return []
        decoys = self.decoy_scorer.sample_decoys(target, k=k)
        unique = []
        seen = set()
        for decoy in decoys:
            if not decoy or decoy == target or decoy in seen:
                continue
            unique.append(decoy)
            seen.add(decoy)
        return unique

    def _score_affinity_batch(self, tcrs: List[str], peptides: List[str]) -> List[float]:
        """Score affinity pairs using the fastest scorer API available."""
        if self.affinity_scorer is None or not tcrs:
            return []
        if hasattr(self.affinity_scorer, "score_batch_fast"):
            scores = self.affinity_scorer.score_batch_fast(tcrs, peptides)
        elif hasattr(self.affinity_scorer, "score_batch"):
            scores, _ = self.affinity_scorer.score_batch(tcrs, peptides)
        else:
            scores = [self.affinity_scorer.score(tcr, pep)[0] for tcr, pep in zip(tcrs, peptides)]
        return [self._finite_scalar(s, 0.0, "decoy_affinity") for s in scores]

    def _sample_simple_decoys(self, target: str) -> List[str]:
        """Sample fixed A/B decoys for target-gated specificity reward."""
        if self.decoy_scorer is None:
            return []
        # Allow truly disabling decoy sampling
        if not self.decoy_fixed_tiers or self.decoy_k_per_tier == 0:
            return []
        if hasattr(self.decoy_scorer, "sample_decoys_by_difficulty"):
            decoys = self.decoy_scorer.sample_decoys_by_difficulty(
                target,
                tiers=self.decoy_fixed_tiers,
                k_per_tier=self.decoy_k_per_tier,
            )
        elif hasattr(self.decoy_scorer, "sample_decoys"):
            k = max(1, len(self.decoy_fixed_tiers) * self.decoy_k_per_tier)
            decoys = self.decoy_scorer.sample_decoys(target, k=k)
        else:
            decoys = []

        unique: List[str] = []
        seen = {target}
        for decoy in decoys:
            if not decoy or decoy in seen:
                continue
            unique.append(decoy)
            seen.add(decoy)
        return unique

    def _finish_decoy_final(
        self,
        final_scores: List[float],
        components: Dict[str, float],
    ) -> float:
        """Aggregate final decoy scores and record diagnostics."""
        if not final_scores:
            components["decoy_n"] = 0.0
            components["decoy_final_mean"] = 0.0
            components["decoy_final_max"] = 0.0
            components["decoy_raw"] = 0.0
            return 0.0

        scores = np.asarray(final_scores, dtype=np.float64)
        mean_score = self._finite_scalar(np.mean(scores), 0.0, "decoy_final_mean")
        max_score = self._finite_scalar(np.max(scores), 0.0, "decoy_final_max")
        agg_score = max_score if self.contrastive_agg == "max" else mean_score
        components["decoy_n"] = float(len(final_scores))
        components["decoy_final_mean"] = mean_score
        components["decoy_final_max"] = max_score
        topk = min(self.decoy_topk, len(final_scores))
        topk_mean = self._finite_scalar(
            np.mean(np.sort(scores)[-topk:]), 0.0, "decoy_final_topk_mean"
        )
        components["decoy_final_topk_mean"] = topk_mean
        components["decoy_raw"] = mean_score
        return agg_score

    def _compute_decoy_final(
        self,
        final_tcr: str,
        target: str,
        components: Dict[str, float],
    ) -> float:
        """Return aggregate final decoy affinity for simple specificity reward."""
        decoy_peptides = self._sample_simple_decoys(target)
        components["decoy_n"] = float(len(decoy_peptides))
        if not decoy_peptides:
            return self._finish_decoy_final([], components)
        final_scores = self._score_affinity_batch(
            [final_tcr] * len(decoy_peptides),
            decoy_peptides,
        )
        return self._finish_decoy_final(final_scores, components)

    def _finish_decoy_delta(
        self,
        final_scores: List[float],
        initial_scores: List[float],
        components: Dict[str, float],
    ) -> float:
        """Aggregate decoy final-initial scores and record diagnostics."""
        if not final_scores:
            components["decoy_initial_mean"] = 0.0
            components["decoy_final_mean"] = 0.0
            components["decoy_delta"] = 0.0
            components["decoy_delta_mean"] = 0.0
            components["decoy_delta_max"] = 0.0
            components["decoy_raw"] = 0.0
            return 0.0

        deltas = np.asarray(final_scores, dtype=np.float64) - np.asarray(initial_scores, dtype=np.float64)
        delta_mean = self._finite_scalar(np.mean(deltas), 0.0, "decoy_delta_mean")
        delta_max = self._finite_scalar(np.max(deltas), 0.0, "decoy_delta_max")
        decoy_delta = delta_max if self.contrastive_agg == "max" else delta_mean

        components["decoy_initial_mean"] = self._finite_scalar(
            np.mean(initial_scores), 0.0, "decoy_initial_mean"
        )
        components["decoy_final_mean"] = self._finite_scalar(
            np.mean(final_scores), 0.0, "decoy_final_mean"
        )
        components["decoy_initial_max"] = self._finite_scalar(
            np.max(initial_scores), 0.0, "decoy_initial_max"
        )
        components["decoy_final_max"] = self._finite_scalar(
            np.max(final_scores), 0.0, "decoy_final_max"
        )
        components["decoy_raw"] = components["decoy_final_mean"]
        components["decoy_delta"] = decoy_delta
        components["decoy_delta_mean"] = delta_mean
        components["decoy_delta_max"] = delta_max
        return decoy_delta

    def _compute_decoy_delta(
        self,
        final_tcr: str,
        initial_tcr: Optional[str],
        target: str,
        components: Dict[str, float],
    ) -> float:
        """Return aggregate decoy delta for specificity shaping.

        Positive decoy delta means the edit increased predicted binding to
        off-target decoys, so the reward subtracts it. Negative decoy delta
        means the edit reduced off-target binding, which increases reward.
        """
        if initial_tcr is None:
            initial_tcr = final_tcr
            components["decoy_delta_missing_initial_tcr"] = 1.0
        else:
            components["decoy_delta_missing_initial_tcr"] = 0.0

        decoy_peptides = self._sample_delta_decoys(target)
        components["decoy_n"] = float(len(decoy_peptides))
        if not decoy_peptides:
            return self._finish_decoy_delta([], [], components)

        final_scores = self._score_affinity_batch(
            [final_tcr] * len(decoy_peptides),
            decoy_peptides,
        )
        initial_scores = self._score_affinity_batch(
            [initial_tcr] * len(decoy_peptides),
            decoy_peptides,
        )
        return self._finish_decoy_delta(final_scores, initial_scores, components)

    def get_ood_stats(self) -> Dict[str, float]:
        """Get OOD penalty statistics."""
        if self._ood_total == 0:
            return {"ood_trigger_rate": 0.0, "ood_triggered": 0, "ood_total": 0}
        return {
            "ood_trigger_rate": self._ood_triggered / self._ood_total,
            "ood_triggered": self._ood_triggered,
            "ood_total": self._ood_total,
        }

    def reset_ood_stats(self):
        """Reset OOD statistics counters."""
        self._ood_triggered = 0
        self._ood_total = 0
