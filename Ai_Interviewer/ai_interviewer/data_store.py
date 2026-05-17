"""Persistence helpers for interview sessions and dashboard summaries."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .config import INTERVIEW_TEMPLATES_PATH, RESULTS_DIR, ROLE_LABELS, ensure_project_dirs
from .questions import load_question_bank, serialize_question


def new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]


def save_interview(record: dict) -> Path:
    ensure_project_dirs()
    record_id = record.get("session_id") or new_session_id()
    record["session_id"] = record_id
    output_path = RESULTS_DIR / f"{record_id}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    return output_path


def load_interviews() -> list[dict]:
    ensure_project_dirs()
    interviews: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            with path.open("r", encoding="utf-8") as file:
                interviews.append(json.load(file))
        except json.JSONDecodeError:
            continue
    return interviews


def load_interview_templates() -> list[dict]:
    ensure_project_dirs()
    if not INTERVIEW_TEMPLATES_PATH.exists():
        templates = _default_interview_templates()
        save_interview_templates(templates)
        return templates

    try:
        with INTERVIEW_TEMPLATES_PATH.open("r", encoding="utf-8") as file:
            templates = json.load(file)
    except json.JSONDecodeError:
        templates = _default_interview_templates()
        save_interview_templates(templates)
        return templates

    if not isinstance(templates, list) or not templates:
        templates = _default_interview_templates()
        save_interview_templates(templates)
    return templates


def save_interview_templates(templates: list[dict]) -> Path:
    ensure_project_dirs()
    with INTERVIEW_TEMPLATES_PATH.open("w", encoding="utf-8") as file:
        json.dump(templates, file, indent=2)
    return INTERVIEW_TEMPLATES_PATH


def active_interview_templates() -> list[dict]:
    return [template for template in load_interview_templates() if template.get("is_active", True)]


def find_interview_template(template_id: str) -> dict | None:
    for template in load_interview_templates():
        if template.get("id") == template_id:
            return template
    return None


def new_template_id() -> str:
    return "interview-" + uuid4().hex[:8]


def interviews_to_frame(interviews: list[dict]) -> pd.DataFrame:
    rows = []
    for interview in interviews:
        summary = interview.get("summary", {})
        role_key = interview.get("role", "")
        rows.append(
            {
                "candidate": interview.get("candidate_name", "Unknown"),
                "interview": interview.get("interview_title", "Untitled interview"),
                "role": ROLE_LABELS.get(role_key, role_key),
                "completed_at": interview.get("completed_at", ""),
                "final_score": summary.get("final_score", 0),
                "average_stress": summary.get("average_stress", 0),
                "attention": summary.get("average_attention", 0),
                "confidence": summary.get("average_confidence", 0),
                "honesty": summary.get("average_honesty", 0),
                "recommendation": summary.get("recommendation", "Not available"),
                "questions": len(interview.get("answers", [])),
            }
        )
    return pd.DataFrame(rows)


def build_summary(answers: list[dict]) -> dict:
    if not answers:
        return {
            "final_score": 0.0,
            "average_stress": 0.0,
            "average_attention": 0.0,
            "average_confidence": 0.0,
            "average_honesty": 0.0,
            "average_eye_contact": 0.0,
            "average_posture": 0.0,
            "average_face_detection_rate": 0.0,
            "recommendation": "Incomplete",
            "strengths": [],
            "improvements": ["Complete at least one answer"],
            "score_notes": [],
        }

    scores = [float(item["evaluation"]["final_score"]) for item in answers]
    stresses = [float(item["behavior"]["stress_score"]) for item in answers]
    attention_scores = [float(item["behavior"].get("attention_score", item["behavior"].get("eye_contact_score", 0))) for item in answers]
    confidence_scores = [float(item["behavior"].get("confidence_score", 0)) for item in answers]
    honesty_scores = [float(item["behavior"].get("honesty_score", 0)) for item in answers]
    eye_scores = [float(item["behavior"].get("eye_contact_score", 0)) for item in answers]
    posture_scores = [float(item["behavior"].get("posture_score", 0)) for item in answers]
    face_rates = [float(item["behavior"].get("face_detection_rate", 0)) for item in answers]
    skill_totals: dict[str, list[float]] = {}
    for item in answers:
        skill = item["question"]["skill"]
        skill_totals.setdefault(skill, []).append(float(item["evaluation"]["final_score"]))

    skill_averages = {
        skill: round(sum(values) / len(values), 2)
        for skill, values in skill_totals.items()
    }
    final_score = round(sum(scores) / len(scores), 2)
    average_stress = round(sum(stresses) / len(stresses), 2)
    average_attention = round(sum(attention_scores) / len(attention_scores), 2)
    average_confidence = round(sum(confidence_scores) / len(confidence_scores), 2)
    average_honesty = round(sum(honesty_scores) / len(honesty_scores), 2)
    strengths = [
        skill
        for skill, score in sorted(skill_averages.items(), key=lambda item: item[1], reverse=True)
        if score >= 65
    ][:3]
    improvements = [
        skill
        for skill, score in sorted(skill_averages.items(), key=lambda item: item[1])
        if score < 65
    ][:3]

    return {
        "final_score": final_score,
        "average_stress": average_stress,
        "average_attention": average_attention,
        "average_confidence": average_confidence,
        "average_honesty": average_honesty,
        "average_eye_contact": round(sum(eye_scores) / len(eye_scores), 2),
        "average_posture": round(sum(posture_scores) / len(posture_scores), 2),
        "average_face_detection_rate": round(sum(face_rates) / len(face_rates), 2),
        "recommendation": _recommendation(final_score, average_stress, average_confidence),
        "skill_averages": skill_averages,
        "strengths": strengths,
        "improvements": improvements,
        "score_notes": [
            "Attention is estimated from eye-contact proxy, face visibility, posture, and movement stability.",
            "Honesty is a consistency heuristic from visual stability and answer completeness, not lie detection.",
        ],
    }


def _recommendation(final_score: float, average_stress: float, average_confidence: float) -> str:
    if final_score >= 78 and average_stress <= 65 and average_confidence >= 55:
        return "Shortlist"
    if final_score >= 62:
        return "Review"
    return "Needs improvement"


def _default_interview_templates() -> list[dict]:
    templates = []
    for role_key, questions in load_question_bank().items():
        role_label = ROLE_LABELS.get(role_key, role_key)
        templates.append(
            {
                "id": f"default-{role_key}",
                "title": f"{role_label} Screening",
                "description": f"Default AI interview for {role_label} candidates.",
                "role": role_key,
                "max_questions": 5,
                "is_active": True,
                "created_by": "system",
                "questions": [serialize_question(question) for question in questions],
            }
        )
    return templates
