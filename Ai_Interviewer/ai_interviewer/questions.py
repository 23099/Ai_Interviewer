"""Question-bank loading and adaptive interview flow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .config import DEFAULT_ROLE_KEY, QUESTION_BANK_PATH


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    skill: str
    level: str
    keywords: tuple[str, ...]
    supportive_prompt: str


@lru_cache(maxsize=1)
def load_question_bank() -> dict[str, list[Question]]:
    with QUESTION_BANK_PATH.open("r", encoding="utf-8") as file:
        raw_bank = json.load(file)

    parsed: dict[str, list[Question]] = {}
    for role_key, role_questions in raw_bank.items():
        parsed[role_key] = [
            Question(
                id=item["id"],
                text=item["text"],
                skill=item["skill"],
                level=item["level"],
                keywords=tuple(item.get("keywords", [])),
                supportive_prompt=item.get("supportive_prompt", item["text"]),
            )
            for item in role_questions
        ]
    return parsed


def get_questions_for_role(role_key: str) -> list[Question]:
    bank = load_question_bank()
    return bank.get(role_key, bank[DEFAULT_ROLE_KEY])


def first_question(role_key: str) -> Question:
    return get_questions_for_role(role_key)[0]


def question_from_dict(item: dict) -> Question:
    return Question(
        id=item["id"],
        text=item["text"],
        skill=item["skill"],
        level=item["level"],
        keywords=tuple(item.get("keywords", [])),
        supportive_prompt=item.get("supportive_prompt", item["text"]),
    )


def questions_from_template(template: dict) -> list[Question]:
    return [question_from_dict(item) for item in template.get("questions", [])]


def choose_next_question(
    role_key: str,
    answered_history: Iterable[dict],
    max_questions: int = 5,
    question_pool: list[Question] | None = None,
) -> Question | None:
    """Pick the next question based on score, stress, and previous coverage."""
    history = list(answered_history)
    if len(history) >= max_questions:
        return None

    role_questions = question_pool or get_questions_for_role(role_key)
    if not role_questions:
        return None
    asked_ids = {item["question"]["id"] for item in history}

    if not history:
        return role_questions[0]

    last = history[-1]
    last_score = float(last["evaluation"]["final_score"])
    last_stress = float(last["behavior"]["stress_score"])
    last_skill = last["question"]["skill"]

    if last_stress >= 68 or last_score < 45:
        supportive = _first_unasked(
            role_questions,
            asked_ids,
            levels=("supportive",),
            preferred_skill=last_skill,
        )
        if supportive:
            return supportive

    if last_score >= 76 and last_stress < 62:
        advanced = _first_unasked(
            role_questions,
            asked_ids,
            levels=("advanced",),
            preferred_skill=last_skill,
        )
        if advanced:
            return advanced

    return _first_unasked(
        role_questions,
        asked_ids,
        levels=("core", "advanced", "supportive"),
    )


def serialize_question(question: Question) -> dict:
    return {
        "id": question.id,
        "text": question.text,
        "skill": question.skill,
        "level": question.level,
        "keywords": list(question.keywords),
        "supportive_prompt": question.supportive_prompt,
    }


def _first_unasked(
    questions: list[Question],
    asked_ids: set[str],
    levels: tuple[str, ...],
    preferred_skill: str | None = None,
) -> Question | None:
    if preferred_skill:
        for question in questions:
            if (
                question.id not in asked_ids
                and question.level in levels
                and question.skill == preferred_skill
            ):
                return question

    for level in levels:
        for question in questions:
            if question.id not in asked_ids and question.level == level:
                return question
    return None
