"""Deterministic nutrition calculators (no API calls)."""

from langchain_core.tools import tool

ACTIVITY_FACTORS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}
GOAL_DELTAS = {"lose": -500, "maintain": 0, "gain": 300}
CALORIE_FLOORS = {"male": 1500, "female": 1200}


def bmr(weight_kg: float, height_cm: float, age: int, sex: str) -> int:
    """Basal metabolic rate, Mifflin-St Jeor (kcal/day)."""
    if sex not in CALORIE_FLOORS:
        raise ValueError(f"sex must be one of {sorted(CALORIE_FLOORS)}, got {sex!r}")
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return round(base + (5 if sex == "male" else -161))


def tdee(bmr_kcal: float, activity: str) -> int:
    """Total daily energy expenditure: BMR x activity factor."""
    if activity not in ACTIVITY_FACTORS:
        raise ValueError(f"activity must be one of {sorted(ACTIVITY_FACTORS)}, got {activity!r}")
    return round(bmr_kcal * ACTIVITY_FACTORS[activity])


def goal_calories(tdee_kcal: float, goal: str, sex: str) -> tuple[int, bool]:
    """Daily calorie target for a goal. Returns (kcal, floored); floored means the
    safety floor (1500 male / 1200 female) raised the target."""
    if goal not in GOAL_DELTAS:
        raise ValueError(f"goal must be one of {sorted(GOAL_DELTAS)}, got {goal!r}")
    if sex not in CALORIE_FLOORS:
        raise ValueError(f"sex must be one of {sorted(CALORIE_FLOORS)}, got {sex!r}")
    target = round(tdee_kcal + GOAL_DELTAS[goal])
    floor = CALORIE_FLOORS[sex]
    return (floor, True) if target < floor else (target, False)


@tool
def calculate_macros(
    calories: int,
    protein_pct: float = 30,
    carbs_pct: float = 40,
    fat_pct: float = 30,
) -> str:
    """Calculate macronutrient gram targets from a daily calorie goal.
    Protein and carbs = 4 kcal/g; fat = 9 kcal/g. Percentages should sum to 100."""
    protein_g = round(calories * protein_pct / 100 / 4)
    carbs_g = round(calories * carbs_pct / 100 / 4)
    fat_g = round(calories * fat_pct / 100 / 9)
    return (
        f"{calories} kcal/day → "
        f"Protein: {protein_g}g ({protein_pct:.0f}%) | "
        f"Carbs: {carbs_g}g ({carbs_pct:.0f}%) | "
        f"Fat: {fat_g}g ({fat_pct:.0f}%)"
    )
