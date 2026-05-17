"""Streamlit user interface for role-based AI interviews."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from .audio import save_audio_file, transcribe_audio
from .auth import authenticate
from .behavior import behavior_from_tracking_summary, summarize_tracking_file
from .config import ACCOUNT_TYPES, ROLE_LABELS, TRACKING_DIR, ensure_project_dirs
from .data_store import (
    active_interview_templates,
    build_summary,
    interviews_to_frame,
    load_interview_templates,
    load_interviews,
    new_session_id,
    new_template_id,
    save_interview,
    save_interview_templates,
)
from .evaluation import evaluate_answer
from .monitoring import CameraMonitor
from .questions import choose_next_question, questions_from_template, serialize_question


def run_app() -> None:
    ensure_project_dirs()
    st.set_page_config(
        page_title="AI Interviewer",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_auth_state()
    st.title("AI Interviewer")

    if st.session_state.auth_user is None:
        _login_view()
        return

    _account_sidebar()
    account_type = st.session_state.auth_user["account_type"]
    if account_type == "hr":
        _dashboard_view()
    elif account_type == "employer":
        _employer_view()
    else:
        _candidate_view()


def _login_view() -> None:
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        account_label = st.selectbox("Account type", list(ACCOUNT_TYPES.values()))
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        account_type = _account_type_from_label(account_label)
        user = authenticate(email, password, account_type)
        if user is None:
            st.error("Invalid login details for the selected account type.")
            return
        st.session_state.auth_user = user
        _reset_candidate_state()
        st.rerun()


def _account_sidebar() -> None:
    user = st.session_state.auth_user
    with st.sidebar:
        st.write(user["display_name"])
        st.caption(user["email"])
        if st.button("Logout", use_container_width=True):
            _stop_question_monitor()
            st.session_state.auth_user = None
            _reset_candidate_state()
            st.rerun()


def _candidate_view() -> None:
    _init_candidate_state()
    templates = active_interview_templates()
    if not templates:
        st.info("No interviews are currently active.")
        return

    if st.session_state.completed_interview is not None and st.session_state.current_question is None:
        _completed_candidate_report()
        return

    if st.session_state.interview_session is None:
        _candidate_interview_picker(templates)
        return

    session = st.session_state.interview_session
    question = st.session_state.current_question
    if question is None:
        _finish_interview()
        st.rerun()

    _ensure_question_monitor(question)
    progress = len(session["answers"]) + 1
    max_questions = session["max_questions"]

    metric_cols = st.columns(5)
    metric_cols[0].metric("Question", f"{min(progress, max_questions)} of {max_questions}")
    metric_cols[1].metric("Interview", session["interview_title"])
    metric_cols[2].metric("Answers", len(session["answers"]))
    metric_cols[3].metric("Monitoring", _monitor_status())
    metric_cols[4].metric("Samples", _monitor_sample_count())

    st.subheader(question.text)
    st.caption(f"Skill: {question.skill} | Level: {question.level}")

    audio_file = st.audio_input("Spoken answer", key=f"audio_{session['session_id']}_{question.id}")
    action_cols = st.columns([1, 1, 4])
    submit_clicked = action_cols[0].button("Submit answer", type="primary", use_container_width=True)
    finish_clicked = action_cols[1].button("Finish", use_container_width=True)

    if submit_clicked:
        if audio_file is None:
            st.warning("Record an answer before submitting.")
        else:
            _submit_voice_answer(audio_file)
            st.rerun()

    if finish_clicked:
        _finish_interview()
        st.rerun()

    _latest_feedback()


def _candidate_interview_picker(templates: list[dict]) -> None:
    labels = [f"{template['title']} | {ROLE_LABELS.get(template['role'], template['role'])}" for template in templates]
    selected_label = st.selectbox("Interview", labels)
    selected_index = labels.index(selected_label)
    selected_template = templates[selected_index]

    detail_cols = st.columns(3)
    detail_cols[0].metric("Role", ROLE_LABELS.get(selected_template["role"], selected_template["role"]))
    detail_cols[1].metric("Questions", min(selected_template.get("max_questions", 5), len(selected_template.get("questions", []))))
    detail_cols[2].metric("Status", "Active" if selected_template.get("is_active", True) else "Paused")
    st.write(selected_template.get("description", ""))

    if st.button("Start interview", type="primary"):
        _start_interview(selected_template)
        st.rerun()


def _completed_candidate_report() -> None:
    completed = st.session_state.completed_interview
    summary = completed.get("summary", {})
    st.success(f"Interview saved. Recommendation: {summary.get('recommendation', 'Not available')}")
    score_cols = st.columns(5)
    score_cols[0].metric("Final score", summary.get("final_score", 0))
    score_cols[1].metric("Attention", summary.get("average_attention", 0))
    score_cols[2].metric("Confidence", summary.get("average_confidence", 0))
    score_cols[3].metric("Honesty", summary.get("average_honesty", 0))
    score_cols[4].metric("Stress", summary.get("average_stress", 0))
    st.write("Strengths:", ", ".join(summary.get("strengths") or ["None yet"]))
    st.write("Improvements:", ", ".join(summary.get("improvements") or ["None"]))
    for note in summary.get("score_notes", []):
        st.caption(note)
    if st.button("Choose another interview"):
        _reset_candidate_state()
        st.rerun()


def _employer_view() -> None:
    templates = load_interview_templates()
    top_cols = st.columns([1, 1, 4])
    if top_cols[0].button("Create interview", type="primary", use_container_width=True):
        templates.append(_blank_template())
        save_interview_templates(templates)
        st.rerun()

    labels = [f"{template['title']} ({template['id']})" for template in templates]
    selected_label = st.selectbox("Interview settings", labels)
    selected_index = labels.index(selected_label)
    template = templates[selected_index]

    with st.form("template_settings"):
        title = st.text_input("Title", value=template.get("title", ""))
        description = st.text_area("Description", value=template.get("description", ""), height=90)
        role_label = st.selectbox(
            "Role",
            list(ROLE_LABELS.values()),
            index=_role_index(template.get("role", "software_engineer")),
        )
        max_questions = st.number_input(
            "Maximum questions",
            min_value=1,
            max_value=20,
            value=int(template.get("max_questions", 5)),
        )
        is_active = st.checkbox("Active", value=bool(template.get("is_active", True)))
        saved = st.form_submit_button("Save settings", type="primary")

    if saved:
        template.update(
            {
                "title": title.strip() or "Untitled Interview",
                "description": description.strip(),
                "role": _role_key_from_label(role_label),
                "max_questions": int(max_questions),
                "is_active": is_active,
            }
        )
        templates[selected_index] = template
        save_interview_templates(templates)
        st.success("Interview settings saved.")

    st.markdown("#### Questions")
    for index, question in enumerate(template.get("questions", [])):
        cols = st.columns([5, 2, 1])
        cols[0].write(question.get("text", ""))
        cols[1].caption(f"{question.get('skill', '')} | {question.get('level', '')}")
        if cols[2].button("Remove", key=f"remove_{template['id']}_{question['id']}", use_container_width=True):
            template["questions"].pop(index)
            templates[selected_index] = template
            save_interview_templates(templates)
            st.rerun()

    with st.form("add_question"):
        st.markdown("#### Add Question")
        new_text = st.text_area("Question", height=90)
        new_skill = st.text_input("Skill", value="communication")
        new_level = st.selectbox("Level", ["core", "supportive", "advanced"])
        new_keywords = st.text_input("Keywords", value="team, result, problem")
        add_clicked = st.form_submit_button("Add question", type="primary")

    if add_clicked:
        if not new_text.strip():
            st.warning("Question text is required.")
        else:
            template.setdefault("questions", []).append(
                {
                    "id": f"{template['id']}-{uuid4().hex[:6]}",
                    "text": new_text.strip(),
                    "skill": new_skill.strip() or "communication",
                    "level": new_level,
                    "keywords": [item.strip() for item in new_keywords.split(",") if item.strip()],
                    "supportive_prompt": new_text.strip(),
                }
            )
            templates[selected_index] = template
            save_interview_templates(templates)
            st.success("Question added.")
            st.rerun()


def _dashboard_view() -> None:
    interviews = load_interviews()
    st.subheader("HR Dashboard")

    if not interviews:
        st.info("No completed interviews yet.")
        return

    frame = interviews_to_frame(interviews)
    metric_cols = st.columns(6)
    metric_cols[0].metric("Candidates", len(frame))
    metric_cols[1].metric("Average score", f"{frame['final_score'].mean():.1f}")
    metric_cols[2].metric("Attention", f"{frame['attention'].mean():.1f}")
    metric_cols[3].metric("Confidence", f"{frame['confidence'].mean():.1f}")
    metric_cols[4].metric("Honesty", f"{frame['honesty'].mean():.1f}")
    metric_cols[5].metric("Shortlisted", int((frame["recommendation"] == "Shortlist").sum()))

    st.dataframe(
        frame.sort_values(["final_score", "confidence", "attention"], ascending=[False, False, False]),
        use_container_width=True,
        hide_index=True,
    )

    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.bar_chart(frame.set_index("candidate")["final_score"])
    with chart_cols[1]:
        st.bar_chart(frame.set_index("candidate")["confidence"])
    with chart_cols[2]:
        st.bar_chart(frame.set_index("candidate")["attention"])

    selected_label = st.selectbox(
        "Candidate detail",
        [f"{item.get('candidate_name', 'Unknown')} | {item.get('session_id', '')}" for item in interviews],
    )
    selected_id = selected_label.split("|")[-1].strip()
    selected = next(item for item in interviews if item.get("session_id") == selected_id)
    _render_interview_detail(selected)

    csv_data = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download dashboard CSV",
        data=csv_data,
        file_name="ai_interviewer_dashboard.csv",
        mime="text/csv",
    )


def _render_interview_detail(interview: dict) -> None:
    summary = interview.get("summary", {})
    st.markdown("#### Candidate Report")
    detail_cols = st.columns(6)
    detail_cols[0].metric("Final", summary.get("final_score", 0))
    detail_cols[1].metric("Stress", summary.get("average_stress", 0))
    detail_cols[2].metric("Attention", summary.get("average_attention", 0))
    detail_cols[3].metric("Confidence", summary.get("average_confidence", 0))
    detail_cols[4].metric("Honesty", summary.get("average_honesty", 0))
    detail_cols[5].metric("Recommendation", summary.get("recommendation", "Not available"))

    skill_averages = summary.get("skill_averages", {})
    if skill_averages:
        st.bar_chart(pd.Series(skill_averages, name="Score"))

    for answer in interview.get("answers", []):
        with st.expander(answer["question"]["text"]):
            st.write("Transcript:", answer.get("answer", ""))
            st.write(
                {
                    "score": answer["evaluation"]["final_score"],
                    "label": answer["evaluation"]["label"],
                    "attention": answer["behavior"].get("attention_score"),
                    "confidence": answer["behavior"].get("confidence_score"),
                    "honesty": answer["behavior"].get("honesty_score"),
                    "stress": answer["behavior"]["stress_score"],
                    "samples": answer["behavior"].get("sample_count", 0),
                    "tracking_file": answer.get("tracking_report", {}).get("path"),
                    "audio_file": answer.get("audio_path"),
                }
            )
            st.write("Feedback:", answer["evaluation"]["feedback"])
            st.write("Evidence:", answer["evaluation"]["evidence"])


def _submit_voice_answer(audio_file) -> None:
    session = st.session_state.interview_session
    question = st.session_state.current_question
    response_seconds = time.time() - st.session_state.question_started_at
    tracking_summary = _stop_question_monitor()

    audio_path = save_audio_file(audio_file, session["session_id"], question.id)
    transcript = transcribe_audio(audio_path)
    answer = transcript.text
    behavior = behavior_from_tracking_summary(tracking_summary, response_seconds, answer)
    evaluation = evaluate_answer(answer, question, response_seconds)

    session["answers"].append(
        {
            "question": serialize_question(question),
            "answer": answer,
            "audio_path": str(audio_path) if audio_path else None,
            "transcript": transcript.to_dict(),
            "response_seconds": round(response_seconds, 2),
            "evaluation": evaluation.to_dict(),
            "behavior": behavior.to_dict(),
            "tracking_report": tracking_summary,
            "answered_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    next_question = choose_next_question(
        session["role"],
        session["answers"],
        max_questions=session["max_questions"],
        question_pool=st.session_state.question_pool,
    )
    st.session_state.latest_feedback = session["answers"][-1]
    st.session_state.current_question = next_question
    st.session_state.question_started_at = time.time()
    st.session_state.current_tracking_path = None
    st.session_state.active_monitor_question_id = None

    if next_question is None:
        _finish_interview()


def _finish_interview() -> None:
    _stop_question_monitor()
    session = st.session_state.interview_session
    if session is None:
        return
    session["completed_at"] = datetime.now().isoformat(timespec="seconds")
    session["summary"] = build_summary(session["answers"])
    save_interview(session)
    st.session_state.completed_interview = session
    st.session_state.interview_session = None
    st.session_state.current_question = None
    st.session_state.latest_feedback = None


def _latest_feedback() -> None:
    feedback = st.session_state.latest_feedback
    if not feedback:
        return

    eval_result = feedback["evaluation"]
    behavior = feedback["behavior"]
    st.markdown("#### Latest Feedback")
    feedback_cols = st.columns(5)
    feedback_cols[0].metric("Answer score", eval_result["final_score"])
    feedback_cols[1].metric("Attention", behavior.get("attention_score", 0))
    feedback_cols[2].metric("Confidence", behavior.get("confidence_score", 0))
    feedback_cols[3].metric("Honesty", behavior.get("honesty_score", 0))
    feedback_cols[4].metric("Stress", behavior["stress_score"])
    st.write("Transcript:", feedback.get("answer", ""))
    if feedback.get("transcript", {}).get("status") != "ok":
        st.warning(feedback.get("transcript", {}).get("error", "Speech was not recognized."))
    st.write(eval_result["feedback"])
    st.write("Scoring evidence:", eval_result["evidence"])
    st.write("Behavior notes:", behavior["notes"])


def _start_interview(template: dict) -> None:
    questions = questions_from_template(template)
    if not questions:
        st.warning("This interview has no questions.")
        return

    session_id = new_session_id()
    st.session_state.question_pool = questions
    st.session_state.interview_session = {
        "session_id": session_id,
        "candidate_name": st.session_state.auth_user["email"],
        "candidate_email": st.session_state.auth_user["email"],
        "interview_template_id": template["id"],
        "interview_title": template["title"],
        "role": template["role"],
        "max_questions": min(int(template.get("max_questions", 5)), len(questions)),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "completed_at": None,
        "answers": [],
        "summary": {},
        "transparency": {
            "uses_demographics": False,
            "scoring_factors": [
                "speech transcript answer quality",
                "role-specific keywords",
                "response timing",
                "continuous eye-contact proxy",
                "facial expression estimate",
                "posture framing",
                "movement stability",
            ],
        },
    }
    st.session_state.current_question = questions[0]
    st.session_state.question_started_at = time.time()
    st.session_state.latest_feedback = None
    st.session_state.completed_interview = None


def _ensure_question_monitor(question) -> None:
    session = st.session_state.interview_session
    monitor_key = f"{session['session_id']}_{question.id}"
    if st.session_state.active_monitor_question_id == monitor_key and st.session_state.camera_monitor is not None:
        return

    _stop_question_monitor()
    tracking_path = TRACKING_DIR / f"{monitor_key}.csv"
    monitor = CameraMonitor(tracking_path)
    monitor.start()
    st.session_state.camera_monitor = monitor
    st.session_state.active_monitor_question_id = monitor_key
    st.session_state.current_tracking_path = str(tracking_path)


def _stop_question_monitor() -> dict:
    monitor = st.session_state.get("camera_monitor")
    if monitor is None:
        path = st.session_state.get("current_tracking_path")
        if path:
            return summarize_tracking_file(Path(path))
        return {"sample_count": 0, "averages": {}, "face_detection_rate": 0.0, "expression_distribution": {}}

    summary = monitor.stop()
    st.session_state.camera_monitor = None
    return summary


def _monitor_status() -> str:
    monitor = st.session_state.get("camera_monitor")
    if monitor is None:
        return "Stopped"
    return monitor.status_message


def _monitor_sample_count() -> int:
    monitor = st.session_state.get("camera_monitor")
    if monitor is None:
        return 0
    return monitor.sample_count


def _blank_template() -> dict:
    return {
        "id": new_template_id(),
        "title": "New Interview",
        "description": "",
        "role": "software_engineer",
        "max_questions": 5,
        "is_active": True,
        "created_by": st.session_state.auth_user["email"],
        "questions": [],
    }


def _init_auth_state() -> None:
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None


def _init_candidate_state() -> None:
    defaults = {
        "interview_session": None,
        "current_question": None,
        "question_started_at": time.time(),
        "latest_feedback": None,
        "completed_interview": None,
        "question_pool": [],
        "camera_monitor": None,
        "active_monitor_question_id": None,
        "current_tracking_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_candidate_state() -> None:
    _stop_question_monitor()
    for key in [
        "interview_session",
        "current_question",
        "latest_feedback",
        "completed_interview",
        "question_pool",
        "camera_monitor",
        "active_monitor_question_id",
        "current_tracking_path",
    ]:
        st.session_state[key] = None if key != "question_pool" else []
    st.session_state.question_started_at = time.time()


def _account_type_from_label(label: str) -> str:
    for account_type, account_label in ACCOUNT_TYPES.items():
        if account_label == label:
            return account_type
    return "candidate"


def _role_key_from_label(label: str) -> str:
    for role_key, role_label in ROLE_LABELS.items():
        if role_label == label:
            return role_key
    return "software_engineer"


def _role_index(role_key: str) -> int:
    keys = list(ROLE_LABELS)
    return keys.index(role_key) if role_key in keys else 0
