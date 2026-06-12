"""Severity classification based on the number of reported symptoms."""


def get_severity(symptom_count: int) -> str:
    """
    Classify severity from symptom count.

    - 1-2 symptoms: Mild
    - 3-4 symptoms: Moderate
    - 5+ symptoms: Severe
    """
    if symptom_count <= 0:
        raise ValueError("Symptom count must be at least 1.")
    if symptom_count <= 2:
        return "Mild"
    if symptom_count <= 4:
        return "Moderate"
    return "Severe"
