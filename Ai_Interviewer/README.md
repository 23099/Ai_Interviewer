# AI Interviewer

An HCI-focused AI interviewer prototype for candidate screening, adaptive questioning, behavior-aware stress estimation, and HR review.

## What it includes

- Candidate interview flow with role selection and five adaptive questions.
- Role-based login for HR, job employer, and job candidate accounts.
- Employer interview settings for creating interviews and adding/removing questions.
- Candidate interview selection with voice-only answer recording.
- Speech-to-text answer evaluation using role keywords, communication quality, response delay, and emotional-intelligence markers.
- Continuous OpenCV webcam monitoring during each answer, saved as per-question CSV tracking logs.
- Average eye-contact proxy, facial expression, posture, attention, confidence, honesty, and stress scoring.
- HR dashboard for comparing candidate scores, stress, recommendations, and question-level feedback.
- Transparent scoring record that avoids demographic fields and stores the evidence used for each score.

## Demo accounts

- HR: `hr@itu.edu.pk` / `123`
- Job Employer: `job_employer@itu.edu.pk` / `123`
- Job Candidate: `candidate@itu.edu.pk` / `123`

## Run

Use the commands in `commands.txt` from the repository root, or run these from this folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel --trusted-host pypi.org --trusted-host files.pythonhosted.org
pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
streamlit run Ai_Interviewer.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.
