from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.tools import (
    ask_user,
    bmr,
    calculate_macros,
    calorie_plan,
    goal_calories,
    meal_macros,
    tdee,
)


def test_bmr_male_and_female():
    # 10*80 + 6.25*180 - 5*30 = 1775; +5 / -161
    assert bmr(80, 180, 30, "male") == 1780
    assert bmr(80, 180, 30, "female") == 1614


def test_bmr_rejects_bad_sex():
    with pytest.raises(ValueError):
        bmr(80, 180, 30, "other")


@pytest.mark.parametrize(
    "activity,expected",
    [
        ("sedentary", 2400),
        ("light", 2750),
        ("moderate", 3100),
        ("active", 3450),
        ("very_active", 3800),
    ],
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


def test_meal_macros_scales_and_sums():
    out = meal_macros.invoke(
        {
            "items": [
                {
                    "name": "chicken",
                    "grams": 200,
                    "calories_100g": 151,
                    "protein_100g": 30.5,
                    "carbs_100g": 0,
                    "fat_100g": 3.2,
                },
                {
                    "name": "rice",
                    "grams": 150,
                    "calories_100g": 130,
                    "protein_100g": 2.7,
                    "carbs_100g": 28,
                    "fat_100g": 0.3,
                },
            ]
        }
    )
    assert "chicken (200 g): 302 kcal, P 61.0 g, C 0.0 g, F 6.4 g" in out
    assert "Total: 497 kcal, Protein 65.0 g, Carbs 42.0 g, Fat 6.9 g" in out


class _S(TypedDict, total=False):
    out: str


def _run(tool, args, resumes):
    """Run a tool inside a one-node graph; feed each resume value in turn."""
    g = StateGraph(_S)
    g.add_node("n", lambda s: {"out": tool.invoke(args)})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    graph = g.compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}
    result = graph.invoke({}, cfg)
    interrupts = []
    for r in resumes:
        interrupts.append(result["__interrupt__"][0].value)
        result = graph.invoke(Command(resume=r), cfg)
    return result, interrupts


PLAN = {"weight_kg": 80, "height_cm": 180, "age": 30, "sex": "male", "activity": "moderate"}


def test_calorie_plan_no_interrupt_without_floor():
    # BMR 1780, TDEE 2759, lose -> 2259
    out = calorie_plan.invoke({**PLAN, "goal": "lose"})
    assert out == "BMR 1780 kcal, TDEE 2759 kcal, lose target: 2259 kcal/day"


def test_calorie_plan_floor_asks_confirmation():
    small = {"weight_kg": 50, "height_cm": 160, "age": 40, "sex": "female", "activity": "sedentary"}
    result, interrupts = _run(calorie_plan, {**small, "goal": "lose"}, [True])
    assert "confirm" in interrupts[0] and "1200" in interrupts[0]["confirm"]
    assert "lose target: 1200 kcal/day (raised to the safety floor" in result["out"]


def test_calorie_plan_floor_declined():
    small = {"weight_kg": 50, "height_cm": 160, "age": 40, "sex": "female", "activity": "sedentary"}
    result, _ = _run(calorie_plan, {**small, "goal": "lose"}, [False])
    assert "declined" in result["out"]


def test_ask_user_interrupts_with_fields_and_returns_answers():
    result, interrupts = _run(ask_user, {"fields": ["age", "sex"]}, [{"age": 30, "sex": "male"}])
    assert interrupts == [{"fields": ["age", "sex"]}]
    assert "'age': 30" in result["out"]
