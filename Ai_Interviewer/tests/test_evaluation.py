import unittest

from ai_interviewer.auth import authenticate
from ai_interviewer.data_store import build_summary, load_interview_templates
from ai_interviewer.evaluation import evaluate_answer
from ai_interviewer.questions import first_question


class EvaluationTests(unittest.TestCase):
    def test_specific_answer_scores_above_empty_answer(self):
        question = first_question("software_engineer")
        strong = evaluate_answer(
            "In my software project I solved a payment bug by testing the API, coordinating with my team, and measuring the result after release.",
            question,
            response_seconds=18,
        )
        empty = evaluate_answer("", question, response_seconds=18)

        self.assertGreater(strong.final_score, empty.final_score)
        self.assertGreaterEqual(strong.communication_score, 50)

    def test_question_loader_has_default_role(self):
        self.assertEqual(first_question("unknown").id, "se_intro")

    def test_known_accounts_authenticate_by_role(self):
        user = authenticate("hr@itu.edu.pk", "123", "hr")
        self.assertIsNotNone(user)
        self.assertEqual(user["account_type"], "hr")
        self.assertIsNone(authenticate("hr@itu.edu.pk", "123", "candidate"))

    def test_default_interview_templates_exist(self):
        templates = load_interview_templates()
        self.assertGreaterEqual(len(templates), 1)
        self.assertIn("questions", templates[0])

    def test_behavior_scores_are_summarized(self):
        summary = build_summary(
            [
                {
                    "question": {"skill": "communication"},
                    "evaluation": {"final_score": 75},
                    "behavior": {
                        "stress_score": 30,
                        "attention_score": 80,
                        "confidence_score": 70,
                        "honesty_score": 65,
                        "eye_contact_score": 78,
                        "posture_score": 72,
                        "face_detection_rate": 100,
                    },
                }
            ]
        )
        self.assertEqual(summary["average_attention"], 80)
        self.assertEqual(summary["average_confidence"], 70)


if __name__ == "__main__":
    unittest.main()
