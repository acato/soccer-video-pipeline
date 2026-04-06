"""
Tier Router — decides which Tier 1 VLM verdicts escalate to Tier 2.

Escalation criteria:
  1. Low confidence (below threshold)
  2. High-value event types (goals, penalties, 1-on-1)
  3. Save/shot ambiguity (hard distinction)
  4. Distribution anomaly (broken Tier 1 detection)
  5. Random spot-check (calibration)
  6. High-motion rejections (motion says event, VLM disagrees)

Usage::

    router = TierRouter()
    routing = router.route(tier1_verdicts)
    # routing.escalated   → list of verdicts to send to Tier 2
    # routing.kept        → list of verdicts to keep from Tier 1
    # routing.tier1_broken → True if Tier 1 model has collapsed
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

from src.detection.models import EventType
from src.detection.vlm_verifier import VLMVerdict, VerificationResult

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Expected distribution — derived from 19-game dataset
# ---------------------------------------------------------------------------

# Approximate expected fraction of each event type among VLM-confirmed
# candidates.  Used to detect label collapse in Tier 1.
_EXPECTED_DISTRIBUTION: dict[str, float] = {
    "shot_stop_diving": 0.13,
    "shot_on_target": 0.10,
    "shot_off_target": 0.08,
    "goal_kick": 0.14,
    "corner_kick": 0.08,
    "goal": 0.04,
    "throw_in": 0.06,
    "free_kick_shot": 0.05,
    "catch": 0.05,
    "penalty": 0.02,
    "kickoff": 0.03,
    "one_on_one": 0.01,
    # "none"/rejected verdicts are handled separately
}


class EscalationReason(str, Enum):
    """Why a verdict was escalated to Tier 2."""
    LOW_CONFIDENCE = "low_confidence"
    HIGH_VALUE = "high_value_event"
    SAVE_SHOT_AMBIGUITY = "save_shot_ambiguity"
    DISTRIBUTION_ANOMALY = "distribution_anomaly"
    SPOT_CHECK = "spot_check"
    HIGH_MOTION_REJECTION = "high_motion_rejection"
    TIER1_BROKEN = "tier1_broken"


@dataclass(frozen=True)
class RoutingDecision:
    """Routing decision for a single verdict."""
    verdict: VLMVerdict
    escalate: bool
    reasons: tuple[EscalationReason, ...] = ()


@dataclass
class RoutingResult:
    """Complete routing result for all Tier 1 verdicts."""
    decisions: list[RoutingDecision]
    tier1_broken: bool = False
    anomaly_details: dict = field(default_factory=dict)

    @property
    def escalated(self) -> list[VLMVerdict]:
        """Verdicts that should go to Tier 2."""
        return [d.verdict for d in self.decisions if d.escalate]

    @property
    def kept(self) -> list[VLMVerdict]:
        """Verdicts to keep from Tier 1 (not escalated)."""
        return [d.verdict for d in self.decisions if not d.escalate]

    @property
    def escalation_rate(self) -> float:
        """Fraction of verdicts escalated."""
        if not self.decisions:
            return 0.0
        return len(self.escalated) / len(self.decisions)

    @property
    def reason_counts(self) -> dict[str, int]:
        """Count of each escalation reason."""
        counts: dict[str, int] = {}
        for d in self.decisions:
            for r in d.reasons:
                counts[r.value] = counts.get(r.value, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class TierRouter:
    """Route Tier 1 VLM verdicts to Tier 2 based on escalation criteria."""

    # Event types that are too important to trust Tier 1 alone
    HIGH_VALUE_TYPES = frozenset({
        EventType.GOAL,
        EventType.PENALTY,
        EventType.ONE_ON_ONE,
    })

    # Event types where save/shot distinction is ambiguous
    AMBIGUOUS_TYPES = frozenset({
        EventType.SHOT_STOP_DIVING,
        EventType.SHOT_ON_TARGET,
        EventType.SHOT_OFF_TARGET,
        EventType.CATCH,
    })

    def __init__(
        self,
        *,
        min_confidence: float = 0.6,
        ambiguity_confidence: float = 0.75,
        broken_threshold: float = 3.0,
        spot_check_rate: float = 0.10,
        escalation_cap: float = 0.50,
        motion_percentile: float = 0.95,
        seed: Optional[int] = None,
    ):
        """
        Args:
            min_confidence: Verdicts below this confidence get escalated.
            ambiguity_confidence: For save/shot types, escalate below this.
            broken_threshold: If any label exceeds expected_rate × this
                multiplier, declare Tier 1 broken.
            spot_check_rate: Fraction of confident verdicts to spot-check.
            escalation_cap: Max fraction of verdicts to escalate (safety cap).
                If exceeded, route ALL to Tier 2 (assumes Tier 1 is broken).
            motion_percentile: Rejected verdicts with motion above this
                percentile get escalated.
            seed: Random seed for reproducible spot-checks.
        """
        self._min_conf = min_confidence
        self._ambiguity_conf = ambiguity_confidence
        self._broken_threshold = broken_threshold
        self._spot_check_rate = spot_check_rate
        self._escalation_cap = escalation_cap
        self._motion_pct = motion_percentile
        self._rng = random.Random(seed)

    def route(self, verdicts: list[VLMVerdict]) -> RoutingResult:
        """Evaluate all Tier 1 verdicts and decide which escalate to Tier 2.

        Returns a RoutingResult with per-verdict decisions and aggregate info.
        """
        if not verdicts:
            return RoutingResult(decisions=[], tier1_broken=False)

        # Step 1: Check for distribution anomaly (broken Tier 1)
        anomaly_broken, anomaly_details = self._check_distribution_anomaly(
            verdicts
        )

        # Step 2: Compute motion threshold for high-motion rejection check
        motion_threshold = self._compute_motion_threshold(verdicts)

        # Step 3: Per-verdict routing decisions
        decisions: list[RoutingDecision] = []
        for v in verdicts:
            reasons = self._evaluate_verdict(
                v,
                anomaly_broken=anomaly_broken,
                motion_threshold=motion_threshold,
            )
            decisions.append(RoutingDecision(
                verdict=v,
                escalate=len(reasons) > 0,
                reasons=tuple(reasons),
            ))

        # Step 4: Check escalation cap
        result = RoutingResult(
            decisions=decisions,
            tier1_broken=anomaly_broken,
            anomaly_details=anomaly_details,
        )

        if result.escalation_rate > self._escalation_cap and not anomaly_broken:
            # Too many escalations — Tier 1 is likely broken
            log.warning("tier_router.escalation_cap_exceeded",
                        rate=round(result.escalation_rate, 3),
                        cap=self._escalation_cap)
            # Escalate everything
            decisions = [
                RoutingDecision(
                    verdict=d.verdict,
                    escalate=True,
                    reasons=d.reasons + (EscalationReason.TIER1_BROKEN,)
                    if d.escalate
                    else (EscalationReason.TIER1_BROKEN,),
                )
                for d in decisions
            ]
            result = RoutingResult(
                decisions=decisions,
                tier1_broken=True,
                anomaly_details={
                    **anomaly_details,
                    "cap_exceeded": True,
                    "escalation_rate": result.escalation_rate,
                },
            )

        # Log summary
        log.info("tier_router.routing_complete",
                 total=len(verdicts),
                 escalated=len(result.escalated),
                 kept=len(result.kept),
                 tier1_broken=result.tier1_broken,
                 escalation_rate=round(result.escalation_rate, 3),
                 reasons=result.reason_counts)

        return result

    # ------------------------------------------------------------------
    # Internal — per-verdict evaluation
    # ------------------------------------------------------------------

    def _evaluate_verdict(
        self,
        verdict: VLMVerdict,
        *,
        anomaly_broken: bool,
        motion_threshold: float,
    ) -> list[EscalationReason]:
        """Evaluate a single verdict against all escalation criteria.

        Returns list of reasons (empty = keep Tier 1 verdict).
        """
        reasons: list[EscalationReason] = []

        # Criterion 0: Tier 1 is broken → escalate everything
        if anomaly_broken:
            reasons.append(EscalationReason.TIER1_BROKEN)
            return reasons  # No need to check other criteria

        # Criterion 1: Low confidence
        if verdict.confidence < self._min_conf:
            reasons.append(EscalationReason.LOW_CONFIDENCE)

        # Criterion 2: High-value event type
        if (verdict.event_type in self.HIGH_VALUE_TYPES
                and verdict.result == VerificationResult.CONFIRMED):
            reasons.append(EscalationReason.HIGH_VALUE)

        # Criterion 3: Save/shot ambiguity
        if (verdict.event_type in self.AMBIGUOUS_TYPES
                and verdict.confidence < self._ambiguity_conf):
            reasons.append(EscalationReason.SAVE_SHOT_AMBIGUITY)

        # Criterion 4: High-motion rejection
        if (verdict.result == VerificationResult.REJECTED
                and verdict.candidate.context.motion_magnitude > motion_threshold):
            reasons.append(EscalationReason.HIGH_MOTION_REJECTION)

        # Criterion 5: Random spot-check (only for confident, non-escalated)
        if not reasons and verdict.result == VerificationResult.CONFIRMED:
            if verdict.confidence >= self._ambiguity_conf:
                if self._rng.random() < self._spot_check_rate:
                    reasons.append(EscalationReason.SPOT_CHECK)

        return reasons

    # ------------------------------------------------------------------
    # Internal — distribution anomaly detection
    # ------------------------------------------------------------------

    def _check_distribution_anomaly(
        self,
        verdicts: list[VLMVerdict],
    ) -> tuple[bool, dict]:
        """Check if Tier 1 label distribution indicates a broken model.

        A model is considered broken if any single event type appears at
        more than `broken_threshold` times its expected rate.

        Returns (is_broken, details_dict).
        """
        # Count confirmed verdicts by event type
        confirmed = [v for v in verdicts if v.result == VerificationResult.CONFIRMED]
        if len(confirmed) < 10:
            # Too few verdicts to judge distribution
            return False, {"skipped": True, "reason": "too_few_verdicts",
                           "confirmed_count": len(confirmed)}

        type_counts = Counter(
            v.event_type.value for v in confirmed if v.event_type
        )
        total_confirmed = len(confirmed)

        anomalies: dict[str, dict] = {}
        is_broken = False

        for event_type_str, count in type_counts.items():
            observed_rate = count / total_confirmed
            expected_rate = _EXPECTED_DISTRIBUTION.get(event_type_str, 0.03)

            if expected_rate > 0 and observed_rate > expected_rate * self._broken_threshold:
                ratio = observed_rate / expected_rate
                anomalies[event_type_str] = {
                    "observed_rate": round(observed_rate, 3),
                    "expected_rate": round(expected_rate, 3),
                    "ratio": round(ratio, 2),
                    "count": count,
                }
                is_broken = True
                log.warning("tier_router.distribution_anomaly",
                            event_type=event_type_str,
                            observed_rate=round(observed_rate, 3),
                            expected_rate=round(expected_rate, 3),
                            ratio=round(ratio, 2),
                            count=count,
                            total=total_confirmed)

        # Also check: if all confirmed verdicts have identical confidence
        # (e.g., all 0.95), the model is not discriminating
        confidences = [v.confidence for v in confirmed]
        if confidences:
            unique_confs = len(set(round(c, 2) for c in confidences))
            if unique_confs <= 2 and len(confirmed) > 20:
                log.warning("tier_router.uniform_confidence",
                            unique_values=unique_confs,
                            total=len(confirmed),
                            sample=round(confidences[0], 3))
                anomalies["__uniform_confidence"] = {
                    "unique_values": unique_confs,
                    "total": len(confirmed),
                    "dominant_confidence": round(confidences[0], 3),
                }
                is_broken = True

        if is_broken:
            log.warning("tier_router.tier1_broken",
                        anomalies=list(anomalies.keys()),
                        total_confirmed=total_confirmed)

        return is_broken, {
            "confirmed_count": total_confirmed,
            "type_distribution": {k: v for k, v in type_counts.most_common()},
            "anomalies": anomalies,
            "is_broken": is_broken,
        }

    # ------------------------------------------------------------------
    # Internal — motion threshold
    # ------------------------------------------------------------------

    def _compute_motion_threshold(self, verdicts: list[VLMVerdict]) -> float:
        """Compute the motion magnitude threshold for the given percentile."""
        magnitudes = sorted(
            v.candidate.context.motion_magnitude for v in verdicts
        )
        if not magnitudes:
            return 0.0
        idx = min(int(len(magnitudes) * self._motion_pct), len(magnitudes) - 1)
        return magnitudes[idx]
