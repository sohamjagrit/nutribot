import pytest

from src.tools import bmr, calculate_macros, goal_calories, tdee


def test_bmr_male_and_female():
    # 10*80 + 6.25*180 - 5*30 = 1775; +5 / -161
    assert bmr(80, 180, 30, "male") == 1780
    assert bmr(80, 180, 30, "female") == 1614


def test_bmr_rejects_bad_sex():
    with pytest.raises(ValueError):
        bmr(80, 180, 30, "other")


@pytest.mark.parametrize(
    "activity,expected",
    [("sedentary", 2400), ("light", 2750), ("moderate", 3100), ("active", 3450), ("very_active", 3800)],
)
def test_tdee_factors(activity, expected):
    assert tdee(2000, activity) == expected


def test_tdee_rejects_bad_activity():
    with pytest.raises(ValueError):
        tdee(1700, "extreme")


def test_goal_deltas():
    assert goal_calories(2500, "lose", "male") == (2000, False)
    assert goal_calories(2500, "maintain", "male") == (2500, False)
    assert goal_calories(2500, "gain", "male") == (2800, False)


def test_goal_floor_applied_and_flagged():
    assert goal_calories(1800, "lose", "male") == (1500, True)
    assert goal_calories(1600, "lose", "female") == (1200, True)


def test_goal_at_floor_not_flagged():
    assert goal_calories(2000, "lose", "male") == (1500, False)


def test_goal_rejects_bad_input():
    with pytest.raises(ValueError):
        goal_calories(2000, "bulk", "male")
    with pytest.raises(ValueError):
        goal_calories(2000, "lose", "x")


def test_macros_split():
    out = calculate_macros.invoke({"calories": 2000})
    assert "Protein: 150g (30%)" in out
    assert "Carbs: 200g (40%)" in out
    assert "Fat: 67g (30%)" in out


def test_macros_custom_split():
    out = calculate_macros.invoke(
        {"calories": 1800, "protein_pct": 40, "carbs_pct": 40, "fat_pct": 20}
    )
    assert "Protein: 180g" in out and "Carbs: 180g" in out and "Fat: 40g" in out
