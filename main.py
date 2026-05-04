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

REMOTE_INFERENCE_URL = os.environ.get("REMOTE_INFERENCE_URL", None)
API_KEY = os.environ.get("API_KEY", None)

SYSTEM_PROMPT = """
You are a strict Manga OCR System. Your task is verbatim text extraction.

READING ORDER: 
Scan the image Right-to-Left, Top-to-Bottom.

EXTRACTION RULES:
1. Capture EVERYTHING: speech, thought bubbles, drawn sound effects, and margin notes.
2. GROUPING (CRITICAL): Merge all words inside the same speech bubble into a single sentence. Ignore line breaks inside the bubble. Do not split a single sentence into multiple outputs.
3. One distinct text bubble = one new line.
4. Transcribe exactly as drawn. Preserve all punctuation.
5. If there is absolutely zero text in the image, output exactly: narrator: "None"

FORMATTING RULES:
- You MUST use this exact format: label: "text"
- Allowed labels: male, female, narrator.
- If the speaker's gender is off-panel, unclear, or it is a sound effect, use: narrator.

Do not output any markdown, preambles, or additional commentary. Extract the text now.
"""


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


async def prepare_image(url: str, client: httpx.AsyncClient):
    try:
        response = await client.get(url, timeout=75.0)
        response.raise_for_status()

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
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


@app.post("/process_page")
async def process_page(request: Request):
    try:
        data = await request.json()
        raw_url = data.get("text")

        if not raw_url:
            return JSONResponse({"error": "No image URL provided"}, status_code=400)

        async with httpx.AsyncClient() as client:

            encoded_image = await prepare_image(raw_url, client)
            if not encoded_image:
                return JSONResponse({"error": "Failed to fetch or process image"}, status_code=400)

            try:
                inference_response = await client.post(
                    REMOTE_INFERENCE_URL,
                    headers={"x-api-key": API_KEY},
                    json={
                        "image": encoded_image,
                        "prompt": SYSTEM_PROMPT
                    },
                    timeout=300.0
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

        cleaned_dialogue = clean_ocr_text(raw_output)
        print(f"Final Cleaned OCR:\n{cleaned_dialogue}")

        if not cleaned_dialogue or "narrator: none" in cleaned_dialogue.lower():
            return {"response": "No text detected", "audio": "", "status": "empty"}

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