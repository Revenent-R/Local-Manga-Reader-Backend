import os
import base64
import asyncio
import io
import re
import httpx
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import edge_tts

app = FastAPI()

# =========================
# CONFIGURATION
# =========================
REMOTE_INFERENCE_URL = os.environ.get("REMOTE_INFERENCE_URL", None)
API_KEY = os.environ.get("API_KEY", None)

# Keeping your exact prompt
SYSTEM_PROMPT = """
You are a Manga OCR and Transcription System. Your sole purpose is to extract EVERY SINGLE piece of text explicitly visible in the image. Never invent, infer, summarize, or paraphrase dialogue.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each line must be exactly formatted as:
  label: "text content"

Allowed labels: male | female | narrator
Nothing else. No preamble, no descriptions, no markdown outside the schema, and no extra blank lines.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 0 — IMAGE AUDIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before extracting, scan the entire image:
  - If NO text exists anywhere in the image → output exactly: narrator: "None" and STOP.
  - If YES → proceed to Phase 1.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — EXHAUSTIVE SPATIAL SCAN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read panels in standard manga order: RIGHT column → LEFT column, TOP → BOTTOM within each panel.
You must locate and prepare to transcribe EVERY text region, including:
  - Speech bubbles (round, spiky, cloud-shaped)
  - Thought bubbles
  - Narration/caption boxes
  - Sound effects (SFX) written in the art
  - Small aside text, whispered text, or margin notes outside of bubbles
  - Text written on clothing, signs, or backgrounds

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — STRICT VERBATIM EXTRACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Copy text EXACTLY as drawn. Do not fix spelling, grammar, or capitalization.
- Preserve all punctuation exactly: "...", "!?", "——", "?!", etc.
- Partially obscured or cut-off text: Transcribe only the exact letters/words you can clearly see. Do NOT use brackets, guess missing words, or use tags like [?] or [illegible].
- Single punctuation bubbles are valid lines (e.g., male: "...").
- NEVER skip a bubble. Even if two bubbles seem duplicate or repetitive, transcribe both.
- NEVER collapse repeated sounds. If the image says "HA HA HA HA", output exactly "HA HA HA HA". Do not shorten it.
- NEVER merge separate text bubbles. Output each distinct text container as its own separate line.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — LABEL ASSIGNMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use ONLY what is visible in the current panel to assign labels:
  Visible male character speaking     → male
  Visible female character speaking   → female
  Speaker off-panel or not shown      → narrator
  Gender ambiguous or unclear         → narrator
  Narration box / caption             → narrator
  Sound effect / SFX                  → narrator
  Thought bubble, thinker not visible → narrator

DEFAULT RULE: If there is any doubt about gender or speaker, use 'narrator'. Never guess.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ZERO OMISSIONS: You must capture every background note, sound effect, and minor text element. 
2. NO INVENTIONS: Only output text you can literally see.
3. NO BLANK LINES.
4. NO FORMAT DEVIATION: Every line must strictly follow label: "text".
5. The FORMAT REFERENCE below is a syntax guide only. Do not reproduce these lines in your output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT REFERENCE (SYNTAX ONLY — DO NOT OUTPUT THESE LINES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  male: "You actually came."
  female: "Did you think I'd stay away?"
  narrator: "Two years had passed since the incident."
  male: "..."
  narrator: "CRASH!!"
  male: "What was that"
  narrator: "BOOM"
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

# 🔥 Made this async using httpx so it doesn't block Render
async def prepare_image(url: str, client: httpx.AsyncClient):
    try:
        response = await client.get(url, timeout=75.0)
        response.raise_for_status()
        
        # CPU-bound PIL operations should technically be in a thread, 
        # but for simple resizing, it's fast enough here.
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        # Explicit high-quality downsampling to save network bandwidth to your local PC
        img.thumbnail((2500, 2500), Image.Resampling.LANCZOS) 
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        print(f"Image prep failed: {e}")
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
    except Exception as e:
        print(f"TTS Error for '{text}': {e}")
        return bytearray()

# =========================
# MAIN ROUTE
# =========================
@app.post("/process_page")
async def process_page(request: Request):
    try:
        data = await request.json()
        raw_url = data.get("text")

        if not raw_url:
            return JSONResponse({"error": "No image URL provided"}, status_code=400)

        # 🔥 Using a single async client for all outbound requests
        async with httpx.AsyncClient() as client:
            
            # 1. Download & Prepare Image
            encoded_image = await prepare_image(raw_url, client)
            if not encoded_image:
                return JSONResponse({"error": "Failed to fetch or process image"}, status_code=400)

            # 2. Remote PC Inference (Async)
            try:
                inference_response = await client.post(
                    REMOTE_INFERENCE_URL,
                    headers={"x-api-key": API_KEY},
                    json={
                        "image": encoded_image,
                        "prompt": SYSTEM_PROMPT
                    },
                    timeout=300.0 # 5 minutes max for complex pages
                )
                inference_response.raise_for_status()
                inference_data = inference_response.json()
            except httpx.HTTPError as e:
                print(f"Local Server Error: {e}")
                return JSONResponse({"error": f"Failed to connect to local GPU: {str(e)}"}, status_code=502)

        if inference_data.get("status") != "success":
            return JSONResponse(
                {"error": f"Inference failed: {inference_data.get('message', 'Unknown error')}"},
                status_code=500
            )

        raw_output = inference_data.get("text", "")

        # 3. Clean OCR
        cleaned_dialogue = clean_ocr_text(raw_output)
        print(f"Final Cleaned OCR:\n{cleaned_dialogue}")

        if not cleaned_dialogue or "narrator: none" in cleaned_dialogue.lower():
            return {"response": "No text detected", "audio": "", "status": "empty"}

        # 4. TTS Generation (Concurrent)
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
        print(f'Critical Process Error: {str(e)}')
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.head("/")
def health_check_head():
    return {"status": "ok"}