import os
import base64
import asyncio
import io
import re
import requests
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import edge_tts

app = FastAPI()

# =========================
# CONFIGURATION
# =========================
REMOTE_INFERENCE_URL = os.environ.get("REMOTE_INFERENCE_URL",None)
API_KEY = os.environ.get("API_KEY",None)

SYSTEM_PROMPT = """
You are a Manga OCR and Transcription System. Extract ONLY text that is explicitly visible in the image. Never invent, infer, or paraphrase dialogue.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each line must be exactly:
    label: "text content"

Allowed labels: male | female | narrator
Nothing else. No preamble, descriptions, markdown, or extra lines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 0 — IMAGE AUDIT (Do this FIRST, before any extraction)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ask yourself: Are there ANY speech bubbles, caption boxes, or visible text characters in this image?
  - If NO text exists anywhere → output exactly: narrator: "None"  then STOP.
  - If YES → proceed to Phase 1.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — SPATIAL SCAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read panels in manga order: RIGHT column → LEFT column, TOP → BOTTOM within each panel.
Locate every text region:
  - Speech bubbles (round, spiky, cloud-shaped)
  - Thought bubbles
  - Narration/caption boxes
  - Sound effects (SFX) written in the art
  - Margin notes or small aside text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — VERBATIM EXTRACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Copy text EXACTLY as drawn. Do not fix spelling, grammar, or capitalization.
- Preserve all punctuation exactly: "...", "!?", "——", "?!", etc.
- Partially obscured text: write best attempt + [?]  →  e.g., male: "Get out of here[?]"
- Fully unreadable text: narrator: "[illegible]"
- Single punctuation bubbles are valid lines: male: "..."
- Do NOT skip any bubble, even if it seems like a duplicate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — LABEL ASSIGNMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use ONLY what is visible in the image to assign labels:

  Visible male character speaking       → male
  Visible female character speaking     → female
  Speaker off-panel or not shown        → narrator
  Gender ambiguous or unclear           → narrator
  Narration box / caption               → narrator
  Sound effect / SFX                    → narrator
  Thought bubble, thinker not visible   → narrator

DEFAULT RULE: Any doubt at all → narrator. Never guess gender.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 4 — CONSOLIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Split bubbles: If one continuous sentence spans multiple bubbles for the same speaker in the same beat, merge into one line.
- Separate beats: If the same character has distinct, separate utterances, output each as its own line.
- Repetition: Collapse repeated identical sounds → "HA HA HA HA" becomes "Hahaha!"
- Never merge lines from different speakers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Only output text you can literally see. No invention, no inference.
2. No blank output lines ever.
3. No line outside the schema format.
4. narrator: "None" means the entire image has zero text — not for pages where a speaker is simply off-screen.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
male: "You actually came."
female: "Did you think I'd stay away?"
narrator: "Two years had passed since the incident."
male: "..."
narrator: "CRASH!!"
male: "What was that[?]"
narrator: "[illegible]"
"""

# =========================
# UTILITIES
# =========================
def is_speakable(text):
    return bool(re.search(r'[a-zA-Z0-9]', text))


def clean_ocr_text(text):
    lines = text.strip().split("\n")
    cleaned_lines = []
    seen_content = set()

    for line in lines:
        l = re.sub(r'^[\s\-\*\d\.\#]+', '', line).strip()
        l = l.replace('**', '')

        if re.search(r'^panel\s*\d+', l, re.IGNORECASE):
            continue

        l = re.sub(r'(\b\w+\b)( \1){3,}', r'\1 \1 \1...', l)

        if l in seen_content and len(l) > 5:
            continue
        seen_content.add(l)

        if l:
            cleaned_lines.append(l)

    return "\n".join(cleaned_lines)


def prepare_image(url):
    try:
        response = requests.get(url, timeout=75)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img.thumbnail((2100, 2100))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception:
        return None


async def get_voice_bytes(text, voice):
    if not is_speakable(text):
        return bytearray()

    try:
        communicate = edge_tts.Communicate(text, voice)
        final_data = bytearray()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                final_data.extend(chunk["data"])

        return final_data

    except Exception:
        return bytearray()


# =========================
# MAIN ROUTE
# =========================
@app.post("/process_page")
async def process_page(request: Request):
    try:
        data = await request.json()
        raw_url = data.get("text")

        encoded_image = prepare_image(raw_url)
        if not encoded_image:
            return JSONResponse({"error": "Image fail"}, status_code=400)

        # =========================
        # REMOTE PC INFERENCE
        # =========================
        inference = requests.post(
            REMOTE_INFERENCE_URL,
            headers={"x-api-key": API_KEY},
            json={
                "image": encoded_image,
                "prompt": SYSTEM_PROMPT
            },
            timeout=300
        )

        data = inference.json()

        if data.get("status") != "success":
            print(f"Inference failed: {data}")
            return JSONResponse(
                {"error": f"Inference failed: {data}"},
                status_code=500
            )

        raw_output = data["text"]

        # =========================
        # CLEAN OCR
        # =========================
        cleaned_dialogue = clean_ocr_text(raw_output)
        print(f"Final Cleaned OCR: {cleaned_dialogue}")

        if not cleaned_dialogue or "narrator: none" in cleaned_dialogue.lower():
            return {"response": "No text detected", "audio": "", "status": "empty"}

        # =========================
        # TTS
        # =========================
        tasks = []

        for line in cleaned_dialogue.split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                label, content = parts[0].strip().lower(), parts[1].strip().strip('"').strip()
            else:
                label, content = "narrator", line.strip().strip('"')

            if content.lower() == "none" or not is_speakable(content):
                continue

            voice = (
                "en-US-AriaNeural" if "female" in label
                else "en-GB-RyanNeural" if "narrator" in label
                else "en-US-GuyNeural"
            )

            tasks.append(get_voice_bytes(content, voice))

        if not tasks:
            return {"response": cleaned_dialogue, "audio": "", "status": "empty"}

        audio_segments = await asyncio.gather(*tasks)

        final_audio = bytearray()
        for seg in audio_segments:
            final_audio.extend(seg)

        return {
            "response": cleaned_dialogue,
            "audio": base64.b64encode(final_audio).decode("utf-8"),
            "status": "success",
        }

    except Exception as e:
        print(f'Error : {str(e)}')
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.head("/")
def health_check_head():
    return {"status": "ok"}