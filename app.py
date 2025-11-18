import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not set. Set it locally in a .env file "
        "or as an environment variable / GitHub secret in your deployment platform."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


def _read_text_file(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def _read_pdf_file(path: Path) -> str:
    if not path.is_file():
        return ""
    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages:
        content = page.extract_text() or ""
        if content:
            text_parts.append(content)
    return "\n".join(text_parts)


def load_section(name: str) -> str:
    txt_path = DATA_DIR / f"{name}.txt"
    pdf_path = DATA_DIR / f"{name}.pdf"

    text = _read_text_file(txt_path)
    if not text.strip():
        text = _read_pdf_file(pdf_path)

    # Special case: support a custom Ashish_Jaiswal.pdf file as the resume
    if name == "resume" and not text.strip():
        alt_resume = DATA_DIR / "Ashish_Jaiswal.pdf"
        text = _read_pdf_file(alt_resume)
    return text


def build_profile_context() -> str:
    resume = load_section("resume")
    linkedin = load_section("linkedin")

    parts: list[str] = []
    if resume.strip():
        parts.append(f"## Resume\n{resume}")
    if linkedin.strip():
        parts.append(f"## LinkedIn Profile\n{linkedin}")
    if not parts:
        parts.append(
            "No profile data is available yet. Please add your resume/LinkedIn text or PDFs "
            "to the data/ directory as resume.txt / resume.pdf and linkedin.txt / linkedin.pdf."
        )
    return "\n\n".join(parts)


PERSON_NAME = os.getenv("PERSON_NAME", "Your Name")
PROFILE_CONTEXT = build_profile_context()
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    f"You are acting as {PERSON_NAME}. You are answering questions from recruiters about "
    f"{PERSON_NAME}'s career, background, skills, and experience. Your responsibility is to "
    f"represent {PERSON_NAME} as faithfully and accurately as possible, based only on the "
    f"information provided.\n\n"
    "Use the resume and LinkedIn information below to answer questions. If you don't know "
    "the answer from this information, say that you are not sure instead of inventing details.\n\n"
    f"{PROFILE_CONTEXT}"
)


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", person_name=PERSON_NAME)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not message:
        return jsonify({"error": "message is required"}), 400

    normalized_history: list[dict] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            normalized_history.append({"role": role, "content": content})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + normalized_history + [
        {"role": "user", "content": message}
    ]

    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
        )
        reply = completion.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
