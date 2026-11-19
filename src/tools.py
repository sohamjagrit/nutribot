"""Macro calculator used by the graph's macros route."""

from langchain_core.tools import tool


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
