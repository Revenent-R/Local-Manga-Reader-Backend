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
You are a Professional Manga OCR and Transcription System. Your task is to extract all dialogue and narrative text from the provided image following a strict schema.

### SCHEMA:
- Format: label: "text content"
- Valid Labels: male, female, narrator
- NO preamble, NO panel descriptions, NO markdown, NO bullet points.

### SCANNING LOGIC:
1. SPATIAL ANALYSIS: Identify all text regions following the Japanese Right-to-Left, Top-to-Bottom flow.
2. TEXT DETECTION: Extract every word, including tiny bubbles and text written outside bubbles (side-notes/SFX).
3. ATTRIBUTION & FALLBACK: Assign 'male' or 'female' based on character appearance and speech style. 
   - CRITICAL: If a character's gender is ambiguous, the speaker is off-screen, or you have any trouble assigning a gender, you MUST default to 'narrator'.

### EXTRACTION RULES:
- CONSOLIDATION: If a single sentence is split into multiple bubbles, merge them into one line.
- REPETITION LIMIT: Consolidate repetitive sounds (e.g., "HA HA HA HA") into a single phrase (e.g., "Hahaha!"). 
- PUNCTUATION: Preserved exactly (e.g., "...", "!?").
- EMPTY PAGE: Only output 'narrator: "None"' if the page is entirely blank.

### STRICT OUTPUT EXAMPLE:
male: "I can't believe it's raining today."
female: "Neither can I! We should head home."
narrator: "Maybe it will stop soon."
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
            timeout=120
        )

        raw_output = inference.json()["text"]

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