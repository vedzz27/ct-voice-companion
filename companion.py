"""CT voice companion: talks with the patient before a CT scan (English / German).

    python companion.py                 # microphone + speaker, staff dashboard on http://localhost:8002
    python companion.py --text          # no microphone: type the patient's answers in the terminal
    python companion.py --model medium  # bigger Whisper model: more accurate, slower

Whisper runs locally (faster-whisper). Nothing is recorded or sent anywhere; the transcript lives
in memory only and is cleared with "New patient" on the dashboard.
"""
import argparse
import os
import queue
import threading
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from dialog import Session
from speech import Mouth

HERE = os.path.dirname(os.path.abspath(__file__))
session = Session()
app = FastAPI(title="CT voice companion")
runtime = {"mode": "starting", "ear": None, "mouth": Mouth(), "say_q": queue.Queue(), "model": ""}


def speak(lines):
    for line in lines:
        runtime["say_q"].put((line, session.c.get("voice")))


def speaker_loop():
    while True:
        text, voice = runtime["say_q"].get()
        runtime["mouth"].say(text, voice)


def mic_loop(model):
    from speech import Ear
    print(f"Loading Whisper '{model}' (first time: download)...", flush=True)
    ear = Ear(model=model)
    runtime["ear"], runtime["mouth"].ear, runtime["mode"] = ear, ear, "microphone"
    print("Listening. Speak to the companion.", flush=True)
    speak(session.start())
    for audio in ear.utterances():
        t0 = time.time()
        lang = None if session.state == "language" else session.lang
        text, detected = ear.transcribe(audio, lang=lang)
        if not text:
            continue
        print(f"  patient ({detected or lang}, {time.time() - t0:.1f}s): {text}", flush=True)
        speak(session.handle(text, detected_lang=detected))


def text_loop():
    runtime["mode"] = "text"
    for line in session.start():
        print("  companion:", line)
    while True:
        try:
            text = input("patient> ")
        except EOFError:
            return
        for line in session.handle(text):
            print("  companion:", line)


# ---- staff dashboard API ----------------------------------------------------------------------
class Typed(BaseModel):
    text: str


class Reset(BaseModel):
    patient_id: str | None = None


@app.get("/")
def page():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/state")
def state():
    ear, mouth = runtime["ear"], runtime["mouth"]
    return dict(session.view(), mode=runtime["mode"], model=runtime["model"],
                listening=bool(ear and ear.listening), speaking=mouth.speaking,
                level=round(ear.level, 4) if ear else 0)


@app.post("/api/say")
def typed(body: Typed):
    """Staff types what the patient said (or tests without a microphone)."""
    replies = session.handle(body.text)
    if runtime["mode"] == "microphone":
        speak(replies)
    return {"replies": replies}


@app.post("/api/reset")
def reset(body: Reset):
    """New patient: forget everything and greet again."""
    with runtime["say_q"].mutex:
        runtime["say_q"].queue.clear()
    session.reset(body.patient_id)
    greeting = session.start()
    if runtime["mode"] == "microphone":
        speak(greeting)
    return {"ok": True}


@app.get("/api/report", response_class=PlainTextResponse)
def report():
    """Short summary for the radiographer (no recording, no full transcript)."""
    v = session.view()
    lines = [f"CT companion summary  {time.strftime('%Y-%m-%d %H:%M')}",
             f"Patient: {v['patient_id'] or '-'}   Language: {v['lang']}   Checklist: {v['progress']}", ""]
    lines.append("ALERTS:" if v["alerts"] else "No alerts.")
    for a in v["alerts"]:
        lines.append(f"  [{a['severity'].upper()}] {a['note']}" + (f": {a.get('detail')}" if a.get("detail") else "")
                     + (f'   (said: "{a["said"]}")' if a["said"] else ""))
    lines += ["", "ANSWERS:"]
    for a in v["answers"]:
        lines.append(f"  {a['answer'].upper():7} {a['question']}" + (f"  -> {a['detail']}" if a["detail"] else ""))
    lines += ["", "Collected by a voice assistant. Staff must confirm every answer with the patient."]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", action="store_true", help="type instead of speaking")
    p.add_argument("--model", default=os.getenv("WHISPER_MODEL", "small"),
                   help="Whisper size: tiny, base, small (default), medium, large-v3")
    p.add_argument("--port", type=int, default=8002)
    a = p.parse_args()
    runtime["model"] = a.model
    threading.Thread(target=uvicorn.run, args=(app,), kwargs=dict(host="0.0.0.0", port=a.port, log_level="warning"),
                     daemon=True).start()
    print(f"Staff dashboard: http://localhost:{a.port}", flush=True)
    if a.text:
        text_loop()
    else:
        threading.Thread(target=speaker_loop, daemon=True).start()
        mic_loop(a.model)


if __name__ == "__main__":
    main()
