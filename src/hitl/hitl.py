"""
Lab 11 — Part 4: Human-in-the-Loop Design
  TODO 12: Confidence Router
  TODO 13: Design 3 HITL decision points
"""
from dataclasses import dataclass


# ============================================================
# TODO 12: Implement ConfidenceRouter
#
# Route agent responses based on confidence scores:
#   - HIGH (>= 0.9): Auto-send to user
#   - MEDIUM (0.7 - 0.9): Queue for human review
#   - LOW (< 0.7): Escalate to human immediately
#
# Special case: if the action is HIGH_RISK (e.g., money transfer,
# account deletion), ALWAYS escalate regardless of confidence.
#
# Implement the route() method.
# ============================================================

HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]

UNCERTAINTY_CUES = [
    "not sure",
    "probably",
    "maybe",
    "i think",
    "unclear",
    "khong chac",
    "co le",
]


@dataclass
class RoutingDecision:
    """Result of the confidence router."""
    action: str          # "auto_send", "queue_review", "escalate"
    hitl_model: str      # human-on-the-loop / human-in-the-loop / human-as-tiebreaker
    confidence: float
    reason: str
    priority: str        # "low", "normal", "high"
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level.

    Thresholds:
        HIGH:   confidence >= 0.9 -> auto-send
        MEDIUM: 0.7 <= confidence < 0.9 -> queue for review
        LOW:    confidence < 0.7 -> escalate to human

    High-risk actions require human approval before the agent acts.
    """

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(self, response: str, confidence: float,
              action_type: str = "general") -> RoutingDecision:
        """Route a response based on confidence score and action type.

        Args:
            response: The agent's response text
            confidence: Confidence score between 0.0 and 1.0
            action_type: Type of action (e.g., "general", "transfer_money")

        Returns:
            RoutingDecision with routing action and metadata
        """
        normalized_response = response.lower()
        effective_confidence = max(0.0, min(confidence, 1.0))

        if any(cue in normalized_response for cue in UNCERTAINTY_CUES):
            effective_confidence = min(effective_confidence, 0.65)

        if action_type in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="queue_review",
                hitl_model="human-in-the-loop",
                confidence=effective_confidence,
                reason=f"High-risk action '{action_type}' requires human approval before execution.",
                priority="high",
                requires_human=True,
            )

        if effective_confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                hitl_model="human-on-the-loop",
                confidence=effective_confidence,
                reason="High confidence and low operational risk.",
                priority="low",
                requires_human=False,
            )

        if effective_confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                hitl_model="human-in-the-loop",
                confidence=effective_confidence,
                reason="Medium confidence response should be reviewed before it reaches the customer.",
                priority="normal",
                requires_human=True,
            )

        return RoutingDecision(
            action="escalate",
            hitl_model="human-as-tiebreaker",
            confidence=effective_confidence,
            reason="Low confidence or ambiguous output requires a human final decision.",
            priority="high",
            requires_human=True,
        )


# ============================================================
# TODO 13: Design 3 HITL decision points
#
# For each decision point, define:
# - trigger: What condition activates this HITL check?
# - hitl_model: Which model? (human-in-the-loop, human-on-the-loop,
#   human-as-tiebreaker)
# - context_needed: What info does the human reviewer need?
# - example: A concrete scenario
#
# Think about real banking scenarios where human judgment is critical.
# ============================================================

hitl_decision_points = [
    {
        "id": 1,
        "name": "Large External Transfer Approval",
        "trigger": "Transfer above 50,000,000 VND, first-time beneficiary, or unusual device/location risk.",
        "hitl_model": "human-in-the-loop",
        "context_needed": "Customer KYC status, account balance, beneficiary history, device fingerprint, fraud score, and recent transaction timeline.",
        "example": "A customer asks the assistant to transfer 75,000,000 VND to a new external account late at night.",
        "expected_response_time": "< 5 minutes",
    },
    {
        "id": 2,
        "name": "Profile Change Post-Action Review",
        "trigger": "Low-risk profile updates such as mailing address, statement email, or card delivery preference that pass automated checks.",
        "hitl_model": "human-on-the-loop",
        "context_needed": "Previous profile values, verification method used, device/session metadata, and an audit trail of the assistant steps taken.",
        "example": "The assistant updates the customer statement email after OTP verification, then queues the case for back-office review.",
        "expected_response_time": "< 30 minutes",
    },
    {
        "id": 3,
        "name": "Policy Conflict or Fraud Dispute",
        "trigger": "The model is below 0.7 confidence, safety rules disagree with business rules, or the customer contests a fraud lock or account restriction.",
        "hitl_model": "human-as-tiebreaker",
        "context_needed": "Conversation transcript, confidence score, guardrail verdicts, fraud alerts, account restrictions, and supporting documents from the customer.",
        "example": "The assistant cannot determine whether to unlock an account because the customer claims a legitimate overseas login while fraud systems mark the session as suspicious.",
        "expected_response_time": "< 15 minutes",
    },
]


# ============================================================
# Quick tests
# ============================================================

def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("I am not sure about this fee policy", 0.88, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 110)
    print(f"{'Scenario':<30} {'Conf':<6} {'Action Type':<20} {'Decision':<15} {'HITL Model':<24} {'Priority'}")
    print("-" * 110)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<30} {decision.confidence:<6.2f} {action_type:<20} "
            f"{decision.action:<15} {decision.hitl_model:<24} {decision.priority}"
        )

    print("=" * 110)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed']}")
        print(f"    SLA:      {point['expected_response_time']}")
        print(f"    Example:  {point['example']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
