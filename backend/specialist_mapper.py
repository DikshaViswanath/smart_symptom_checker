"""Map predicted diseases to recommended medical specialists."""

DISEASE_TO_SPECIALIST = {
    "Flu": "General Physician",
    "Common Cold": "General Physician",
    "Pneumonia": "Pulmonologist",
    "Asthma": "Pulmonologist",
    "Bronchitis": "Pulmonologist",
    "Migraine": "Neurologist",
    "Hypertension": "Cardiologist",
    "Diabetes": "Endocrinologist",
    "Dengue": "Infectious Disease Specialist",
    "Malaria": "Infectious Disease Specialist",
    "Typhoid": "Infectious Disease Specialist",
    "GERD": "Gastroenterologist",
    "Urinary Tract Infection": "Urologist",
    "Anemia": "Hematologist",
    "Chickenpox": "Dermatologist",
    "Allergic Rhinitis": "Allergist",
    "Sinusitis": "ENT Specialist",
    "Appendicitis": "General Surgeon",
    "Arthritis": "Rheumatologist",
    "Depression": "Psychiatrist",
}

DEFAULT_SPECIALIST = "General Physician"


def get_specialist(disease: str) -> str:
    """Return the specialist recommended for a predicted disease."""
    if not disease or not str(disease).strip():
        return DEFAULT_SPECIALIST
    return DISEASE_TO_SPECIALIST.get(str(disease).strip(), DEFAULT_SPECIALIST)
