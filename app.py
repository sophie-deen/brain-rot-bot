import json
import os
import re
from datetime import datetime

import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.environ.get("ELEVENLABS_AGENT_ID", "agent_4301kn6yvgkpfn6s6rpe2m3xgbj2")
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")

def _load_system_prompt():
    try:
        path = os.path.join(os.path.dirname(__file__), "dr-daley-system-prompt-v1.4.md")
        with open(path) as f:
            content = f.read()
        match = re.search(r'```\n(.*?)```', content, re.DOTALL)
        return match.group(1).strip() if match else ""
    except Exception:
        return ""

SYSTEM_PROMPT_BASE = _load_system_prompt()


def scrape_linkedin(url: str) -> str:
    """Fetch LinkedIn profile via RapidAPI Real-Time LinkedIn Scraper."""
    resp = requests.get(
        "https://linkedin-api8.p.rapidapi.com/get-profile-data-by-url",
        headers={
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "linkedin-api8.p.rapidapi.com",
        },
        params={"url": url},
        timeout=20,
    )
    resp.raise_for_status()
    p = resp.json()
    lines = []
    name = " ".join(filter(None, [p.get("firstName"), p.get("lastName")]))
    if name:
        lines.append(f"Name: {name}")
    if p.get("headline"):
        lines.append(f"Headline: {p['headline']}")
    for exp in (p.get("experience") or [])[:3]:
        title = exp.get("title", "")
        company = exp.get("company", "")
        if title or company:
            lines.append(f"Role: {title} at {company}")
    if p.get("summary"):
        lines.append(f"Summary: {p['summary'][:400]}")
    return "\n".join(lines)


def get_signed_url_with_context(context: str) -> str:
    """Get ElevenLabs signed URL with LinkedIn context injected into system prompt."""
    prompt = SYSTEM_PROMPT_BASE.replace(
        "No additional user context. Proceed cold.",
        context
    )
    resp = requests.post(
        "https://api.elevenlabs.io/v1/convai/conversation/get_signed_url",
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={
            "agent_id": ELEVENLABS_AGENT_ID,
            "conversation_config_override": {
                "agent": {"prompt": {"prompt": prompt}}
            },
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["signed_url"]


@app.route("/start-call", methods=["POST"])
def start_call():
    """Optionally scrape LinkedIn, then return signed URL or agent ID."""
    data = request.json or {}
    linkedin_url = data.get("linkedin_url", "").strip()

    if linkedin_url and RAPIDAPI_KEY:
        try:
            context = scrape_linkedin(linkedin_url)
            if context:
                signed_url = get_signed_url_with_context(context)
                return jsonify({"signed_url": signed_url})
        except Exception as e:
            print(f"[start-call] LinkedIn/ElevenLabs error: {e}")

    # Fallback: no LinkedIn or scrape failed — plain agent ID
    return jsonify({"agent_id": ELEVENLABS_AGENT_ID})


# In-memory session store: conversation_id → {status, report}
sessions = {}

SCORING_PROMPT = """You are the scoring engine for Brain Rot Bot, a cognitive fitness assessment service.
You have just received the transcript of a conversation between Dr. Axon Daley
and a patient. Your job is to score the patient across six competencies and
generate their report.

Read the full transcript carefully before scoring anything.

---

THE SIX COMPETENCIES

Score each from 1 to 10 based on evidence in the transcript.
Do not infer. Do not be generous. Score what you actually observed.

1. CRITICAL REASONING [CHC: Fluid Intelligence — Gf]
   Did they spot flaws in claims made during the conversation?
   Did they push back with reasoning, or accept without question?
   Did they reason aloud, or just assert positions?
   10 = spotted the flaw, pushed back confidently, reasoned through it
   1 = accepted everything, deferred to authority, no independent evaluation

2. CREATIVE INITIATION [CHC: Fluid Intelligence + Retrieval Fluency — Gf/Gr]
   Did they generate original ideas when asked open questions?
   Or did they react to prompts, ask what you were looking for, or stall?
   Were their answers genuinely theirs, or did they sound AI-generated?
   10 = immediate generative thinking, original, specific, willing to speculate
   1 = stalled, asked for options, gave generic or reactive answers

3. EPISTEMIC CONFIDENCE [CHC: Crystallised Intelligence — Gc]
   Did they know what they actually know vs what they've absorbed?
   Did they distinguish personal knowledge from borrowed information?
   Were they calibrated — accurate about their own uncertainty?
   10 = clear distinction between what they know and don't know, calibrated confidence
   1 = presented absorbed information as personal knowledge, couldn't explain reasoning

4. ATTENTION DEPTH [CHC: Working Memory / Sustained Attention — Gwm]
   Could they sit with a hard thought without deflecting or simplifying?
   Did they tolerate silence and ambiguity, or reach for rescue?
   Did they follow a thought to its end?
   10 = sustained engagement with complexity, tolerated silence, went deeper when asked
   1 = premature closure, deflected to safe ground, unable to follow a thread

5. VERBAL PRECISION [CHC: Language Ability — Gc]
   Could they say exactly what they meant?
   Or did they rely on filler, vague quantifiers, and hedging?
   Did their language become more or less precise under pressure?
   10 = precise, specific, self-correcting, minimal filler
   1 = heavy filler, vague throughout, couldn't define terms when pressed

6. WORKING MEMORY [CHC: Working Memory Capacity — Gwm]
   Could they hold information from earlier in the conversation?
   Did they recall planted details accurately when referenced later?
   Did they lose the thread of their own sentences?
   10 = accurate recall, held context across the full conversation
   1 = frequent repetition needed, lost thread, confabulated to fill gaps

---

ARCHETYPE ASSIGNMENT

Based on the score pattern, assign one of the following archetypes.
Choose the one that best fits the overall profile — not just the total score.

THE PROMPT MONKEY
Clever. Fast. Entirely dependent.
Profile: high verbal precision, low epistemic confidence, low creative initiation.
They sound good. They just aren't thinking.

THE GHOST WRITER
You still think the thoughts. You just don't write them anymore.
Profile: reasonable scores overall but creative initiation notably suppressed.
The atrophy is specific and creeping.

THE PASSENGER
Going somewhere. Not driving.
Profile: low critical reasoning, low attention depth. Accepts. Doesn't interrogate.
Someone else is doing the thinking.

THE CO-PILOT
Actually using AI as a tool. Genuinely rare.
Profile: high scores across the board, especially epistemic confidence and critical reasoning.
Dr. Daley is cautiously optimistic.

THE LUDDITE
Refusing AI on principle. Scores high. Slightly exhausting at parties.
Profile: very high across all competencies, volunteered opinions on AI being a problem.
Fine. But perhaps relax.

THE ATROPHIER
[REDACTED FOR YOUR PROTECTION]
Profile: scores below 4 across three or more competencies.
Dr. Daley has significant concerns. Recommend immediate reassessment.

---

FIVE YEAR PROJECTION

Calculate a projected score in 5 years based on current trajectory.
Assume the following degradation rates (based on AI dependency research):
- If overall score is below 50: decline of 8–12 points over 5 years
- If overall score is 50–69: decline of 4–7 points over 5 years
- If overall score is 70+: stable or slight decline of 1–3 points

Write one dry, specific observation about what this means for the patient.
Format: "AT CURRENT TRAJECTORY, [PATIENT FIRST NAME] WILL [SPECIFIC OUTCOME] BY [YEAR]."
Be specific. Be dry. Be accurate. Do not be cruel. Do not be encouraging.
Example: "AT CURRENT TRAJECTORY, PATIENT WILL REQUIRE AI ASSISTANCE TO FORM AN
OPINION ABOUT A BISCUIT BY 2029."

---

CLINICAL NOTES

Write 2–3 sentences in Dr. Daley's voice.
Register: precise, clinical, high-IQ. Like a BMJ paper written by someone who finds
the patient faintly disappointing. Occasional dry aside. No warmth. No cruelty.
Be specific to what actually happened in the transcript — no generic observations.
Reference specific moments or patterns you observed.

---

INTERVENTIONS

Write exactly three interventions. Each should be:
- Specific and actionable
- Involving no AI whatsoever
- Slightly uncomfortable to read
- Written in the same clinical register as the notes

---

OUTPUT FORMAT

Return valid JSON only. No preamble. No explanation. No markdown.
If you cannot score a competency due to insufficient transcript evidence, score it 5
and note it in clinical_notes.

{
  "patient_name": "[first name from transcript, or PATIENT if unknown]",
  "scores": {
    "critical_reasoning": 0,
    "creative_initiation": 0,
    "epistemic_confidence": 0,
    "attention_depth": 0,
    "verbal_precision": 0,
    "working_memory": 0
  },
  "overall": 0,
  "archetype": "",
  "archetype_tagline": "",
  "clinical_notes": "",
  "five_year_score": 0,
  "five_year_observation": "",
  "interventions": ["", "", ""],
  "ascii_brain_state": "healthy | moderate | terminal"
}

ascii_brain_state rules:
- overall 70–100: "healthy"
- overall 40–69: "moderate"
- overall 0–39: "terminal"

---

TRANSCRIPT:

{{TRANSCRIPT}}"""


def render_bar(score):
    filled = max(0, min(10, int(score)))
    return "█" * filled + "░" * (10 - filled)


def score_transcript(transcript_text: str) -> dict:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = SCORING_PROMPT.replace("{{TRANSCRIPT}}", transcript_text)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(message.content[0].text)


def extract_transcript(data: dict) -> str:
    """Parse ElevenLabs webhook payload into a plain transcript string."""
    # ElevenLabs sends transcript as a list of {role, message} objects
    transcript = data.get("transcript") or data.get("transcription", [])
    if isinstance(transcript, list):
        lines = []
        for t in transcript:
            role = t.get("role", t.get("speaker", "unknown")).upper()
            text = t.get("message", t.get("text", ""))
            lines.append(f"{role}: {text}")
        return "\n".join(lines)
    return str(transcript)


@app.route("/")
def index():
    return render_template("index.html", agent_id=ELEVENLABS_AGENT_ID)


@app.route("/webhook/elevenlabs", methods=["POST"])
def elevenlabs_webhook():
    data = request.json or {}
    conversation_id = (
        data.get("conversation_id")
        or data.get("conversationId")
        or data.get("data", {}).get("conversation_id")
    )

    if not conversation_id:
        return jsonify({"error": "no conversation_id"}), 400

    transcript_text = extract_transcript(data)

    if not transcript_text.strip():
        sessions[conversation_id] = {"status": "error", "error": "Empty transcript"}
        return jsonify({"ok": True})

    try:
        report = score_transcript(transcript_text)
        report["date"] = datetime.now().strftime("%d.%m.%Y")
        report["bars"] = {k: render_bar(v) for k, v in report["scores"].items()}
        sessions[conversation_id] = {"status": "ready", "report": report}
    except Exception as e:
        sessions[conversation_id] = {"status": "error", "error": str(e)}

    return jsonify({"ok": True})


@app.route("/results/<conversation_id>")
def results(conversation_id):
    session = sessions.get(conversation_id)
    if not session:
        return render_template(
            "results.html", status="waiting", conversation_id=conversation_id
        )
    if session["status"] == "error":
        return render_template(
            "results.html",
            status="error",
            error=session.get("error", "Unknown error"),
            conversation_id=conversation_id,
        )
    return render_template(
        "results.html",
        status="ready",
        report=session["report"],
        conversation_id=conversation_id,
    )


def fetch_and_score(conversation_id: str):
    """Fetch transcript from ElevenLabs API and score it. Stores result in sessions."""
    try:
        resp = requests.get(
            f"https://api.elevenlabs.io/v1/convai/conversations/{conversation_id}",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            timeout=10,
        )
        print(f"[ElevenLabs] {conversation_id} -> HTTP {resp.status_code}")
        if resp.status_code != 200:
            print(f"[ElevenLabs] Error body: {resp.text[:300]}")
            return
        data = resp.json()
        el_status = data.get("status")
        transcript = data.get("transcript", [])
        print(f"[ElevenLabs] status={el_status}, transcript_len={len(transcript)}")
        if el_status == "processing":
            return  # not ready yet, keep polling
        if len(transcript) < 2:
            sessions[conversation_id] = {"status": "error", "error": "Call ended too early — nothing to score."}
            return
        if el_status not in ("done", "failed"):
            return
        transcript_text = extract_transcript(data)
        if not transcript_text.strip():
            sessions[conversation_id] = {"status": "error", "error": "Empty transcript"}
            return
        report = score_transcript(transcript_text)
        report["date"] = datetime.now().strftime("%d.%m.%Y")
        report["bars"] = {k: render_bar(v) for k, v in report["scores"].items()}
        sessions[conversation_id] = {"status": "ready", "report": report}
    except Exception as e:
        sessions[conversation_id] = {"status": "error", "error": str(e)}


@app.route("/status/<conversation_id>")
def status(conversation_id):
    session = sessions.get(conversation_id)
    if not session:
        fetch_and_score(conversation_id)
        session = sessions.get(conversation_id)
    if not session:
        return jsonify({"status": "waiting"})
    return jsonify({"status": session["status"]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)
