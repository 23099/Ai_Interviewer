"""Answer scoring for skill, communication, and thinking indicators."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .questions import Question


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.-]*")


@dataclass(frozen=True)
class EvaluationResult:
    final_score: float
    skill_score: float
    keyword_score: float
    communication_score: float
    thinking_score: float
    eq_score: float
    label: str
    feedback: str
    evidence: list[str]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["evidence"] = list(self.evidence)
        return data


def evaluate_answer(
    answer: str,
    question: Question,
    response_seconds: float,
) -> EvaluationResult:
    """Score a candidate answer with transparent, non-demographic factors."""
    clean_answer = answer.strip()
    if not clean_answer:
        return EvaluationResult(
            final_score=0.0,
            skill_score=0.0,
            keyword_score=0.0,
            communication_score=0.0,
            thinking_score=_thinking_score(response_seconds),
            eq_score=0.0,
            label="No response",
            feedback="No response was detected, so this answer needs another attempt.",
            evidence=["Empty answer"],
        )

    skill_score = _svm_skill_score(clean_answer)
    keyword_score = _keyword_score(clean_answer, question.keywords)
    communication_score = _communication_score(clean_answer)
    thinking_score = _thinking_score(response_seconds)
    eq_score = _eq_score(clean_answer)

    final_score = round(
        0.36 * skill_score
        + 0.24 * keyword_score
        + 0.18 * communication_score
        + 0.12 * thinking_score
        + 0.10 * eq_score,
        2,
    )
    label = _label_for_score(final_score)
    feedback = _feedback_for_score(final_score, question.skill)
    evidence = _evidence(
        clean_answer,
        question.keywords,
        response_seconds,
        keyword_score,
        communication_score,
        eq_score,
    )

    return EvaluationResult(
        final_score=final_score,
        skill_score=round(skill_score, 2),
        keyword_score=round(keyword_score, 2),
        communication_score=round(communication_score, 2),
        thinking_score=round(thinking_score, 2),
        eq_score=round(eq_score, 2),
        label=label,
        feedback=feedback,
        evidence=evidence,
    )


@lru_cache(maxsize=1)
def _answer_model() -> Pipeline:
    training_data = [
        ("I used a structured approach, identified the problem, tested alternatives, and measured the result.", 1),
        ("I built a small prototype, compared tradeoffs, communicated blockers, and improved the design.", 1),
        ("My strength is problem solving because I break complex work into testable steps.", 1),
        ("I handled conflict by listening first, clarifying expectations, and agreeing on a plan.", 1),
        ("I would validate the data, explain assumptions, and make a fair recommendation.", 1),
        ("I do not know.", 0),
        ("Nothing special.", 0),
        ("I just need this job.", 0),
        ("No idea, maybe I can try.", 0),
        ("I am good and hardworking only.", 0),
    ]
    texts, labels = zip(*training_data)
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("svm", SVC(kernel="linear", probability=True, random_state=30)),
        ]
    )
    model.fit(texts, labels)
    return model


def _svm_skill_score(answer: str) -> float:
    model = _answer_model()
    probability = model.predict_proba([answer])[0][1]
    return float(probability * 100)


def _keyword_score(answer: str, keywords: tuple[str, ...]) -> float:
    if not keywords:
        return 65.0
    lower_answer = answer.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lower_answer)
    coverage = hits / max(len(keywords), 1)
    return min(100.0, 35.0 + coverage * 80.0)


def _communication_score(answer: str) -> float:
    words = WORD_RE.findall(answer)
    word_count = len(words)
    unique_ratio = len({word.lower() for word in words}) / max(word_count, 1)
    sentence_count = max(answer.count("."), answer.count("?"), answer.count("!"), 1)
    average_sentence_len = word_count / sentence_count

    length_score = _bounded_scale(word_count, low=12, high=80)
    variety_score = min(100.0, unique_ratio * 115.0)
    sentence_score = 100.0 - min(abs(average_sentence_len - 18.0) * 3.0, 55.0)
    example_bonus = 12.0 if _has_example_marker(answer) else 0.0
    return _clamp(0.45 * length_score + 0.30 * variety_score + 0.25 * sentence_score + example_bonus)


def _thinking_score(response_seconds: float) -> float:
    if response_seconds <= 0:
        return 50.0
    if response_seconds <= 4:
        return 72.0
    if response_seconds <= 45:
        return 100.0 - abs(response_seconds - 15.0) * 0.8
    return max(35.0, 76.0 - math.log(response_seconds - 35.0) * 12.0)


def _eq_score(answer: str) -> float:
    lower_answer = answer.lower()
    positive_markers = [
        "listen",
        "empathy",
        "feedback",
        "team",
        "collaborate",
        "support",
        "fair",
        "communicate",
        "learn",
        "reflect",
        "improve",
    ]
    accountability_markers = ["i learned", "i would improve", "mistake", "responsible", "ownership"]
    hits = sum(1 for marker in positive_markers if marker in lower_answer)
    accountability = sum(1 for marker in accountability_markers if marker in lower_answer)
    return _clamp(35.0 + hits * 7.0 + accountability * 10.0)


def _evidence(
    answer: str,
    keywords: tuple[str, ...],
    response_seconds: float,
    keyword_score: float,
    communication_score: float,
    eq_score: float,
) -> list[str]:
    matched_keywords = [keyword for keyword in keywords if keyword.lower() in answer.lower()]
    evidence = [
        f"Response time: {response_seconds:.1f} seconds",
        f"Matched skill terms: {', '.join(matched_keywords) if matched_keywords else 'none'}",
    ]
    if keyword_score < 55:
        evidence.append("Answer should include more role-specific detail.")
    if communication_score >= 70:
        evidence.append("Answer has enough detail for communication scoring.")
    if eq_score >= 65:
        evidence.append("Answer shows collaboration, reflection, or empathy markers.")
    if _has_example_marker(answer):
        evidence.append("Answer includes example-style reasoning.")
    return evidence


def _feedback_for_score(score: float, skill: str) -> str:
    if score >= 80:
        return f"Strong answer with clear evidence for {skill}."
    if score >= 60:
        return f"Good direction. Add a concrete example to strengthen {skill}."
    if score >= 40:
        return f"Partially relevant. Give more detail about your actions and results for {skill}."
    return f"Weak answer. Reframe it with a specific situation, action, and result for {skill}."


def _label_for_score(score: float) -> str:
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Needs detail"
    return "Weak"


def _has_example_marker(answer: str) -> bool:
    lower_answer = answer.lower()
    return any(
        marker in lower_answer
        for marker in ("for example", "in my project", "i built", "i handled", "the result", "because")
    )


def _bounded_scale(value: float, low: float, high: float) -> float:
    if value <= low:
        return value / low * 55.0
    if value >= high:
        return 92.0
    return 55.0 + (value - low) / (high - low) * 45.0


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))
