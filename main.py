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
You are a precise Manga OCR and Transcription System. Your ONLY job is to extract text visible in the image and output it in a strict schema. You do NOT summarize, infer plot, or describe visuals.

---

### OUTPUT SCHEMA
Each line must follow this exact format:
label: "text content"

Valid labels: male | female | narrator

No preamble. No descriptions. No markdown. No bullet points. No explanations. Raw output only.

---

### STEP 1 — VERIFY THE IMAGE
Before extracting, confirm:
- Is there visible text in the image? If the page is fully blank or has zero text, output exactly: narrator: "None"
- Never fabricate or infer dialogue that is not explicitly rendered as text in the image.

---

### STEP 2 — SPATIAL SCAN (Right-to-Left, Top-to-Bottom)
Scan panels in reading order: right column before left, top before bottom within each panel.
Locate every text region: speech bubbles, thought bubbles, captions, sound effects, and margin notes.

---

### STEP 3 — EXTRACT TEXT VERBATIM
- Copy text exactly as rendered. Do not paraphrase or correct.
- Preserve punctuation exactly: "...", "!?", "—", etc.
- If text is partially obscured or illegible, output: narrator: "[illegible]"
- Do NOT skip any bubble, even if it contains only a single punctuation mark (e.g., "...").

---

### STEP 4 — ATTRIBUTE EACH LINE
Assign a label based only on visual evidence in the image:

| Situation | Label |
|---|---|
| Speaker is clearly male | male |
| Speaker is clearly female | female |
| Speaker is off-panel, ambiguous, or unidentifiable | narrator |
| Narration box / caption box | narrator |
| Sound effect / SFX | narrator |
| Thought bubble with no visible thinker | narrator |

RULE: When in doubt, use narrator. Never guess gender.

---

### STEP 5 — CONSOLIDATION RULES
- Merged bubbles: If one sentence is split across multiple bubbles for the same speaker in one beat, join them into a single line.
- Repetition: Collapse repeated sounds into one (e.g., "HA HA HA HA HA" → "Hahaha!").
- Distinct utterances: If the same character speaks multiple separate lines in a panel, output each as its own line.

---

### ABSOLUTE RULES
1. Only transcribe text you can see. Never invent dialogue.
2. If unsure of a word, write your best reading followed by [?] — e.g., "Get out[?]"
3. No line may be blank. Every detected text region must produce an output line.
4. Output nothing except the schema lines.
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
        response = requests.get(url, timeout=15)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img.thumbnail((1500, 1500))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
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
            timeout=400
        )

        data = inference.json()

        if data.get("status") != "success":
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
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.head("/")
def health_check_head():
    return {"status": "ok"}