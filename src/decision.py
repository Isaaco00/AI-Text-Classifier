HIGH_CONFIDENCE_THRESHOLD = 0.65
REVIEW_THRESHOLD = 0.40


def get_decision(confidence):
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return {
            "status": "HIGH_CONFIDENCE",
            "action": "AUTO_ROUTE",
            "message": "The model is confident enough to route this ticket automatically."
        }

    if confidence >= REVIEW_THRESHOLD:
        return {
            "status": "REVIEW",
            "action": "HUMAN_REVIEW",
            "message": "The model has a prediction, but human review is recommended."
        }

    return {
        "status": "UNCERTAIN",
        "action": "CLARIFY",
        "message": "The model is not confident enough. Additional information is recommended."
    }