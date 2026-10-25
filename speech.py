"""Microphone -> utterances -> Whisper (local), and text -> speech.

Listening:  16 kHz mono from the default microphone; a simple energy detector cuts the stream into
            utterances (speech followed by ~0.8 s of silence), then faster-whisper transcribes each one
            on this computer. No audio is written to disk or sent anywhere.
Speaking:   macOS `say` with a German or English voice; the microphone is muted while speaking so the
            companion never transcribes itself.
"""
import queue
import shutil
import subprocess
import threading
import time

import numpy as np

RATE = 16000
FRAME = 480                     # 30 ms


class Ear:
    def __init__(self, model="small", device="cpu", compute_type="int8", silence_s=0.8, max_s=15.0,
                 start_factor=3.0, min_speech_s=0.3):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.silence_frames = int(silence_s * RATE / FRAME)
        self.max_frames = int(max_s * RATE / FRAME)
        self.min_frames = int(min_speech_s * RATE / FRAME)
        self.start_factor = start_factor
        self.frames = queue.Queue()
        self.muted = threading.Event()
        self.noise = 0.003
        self.level = 0.0            # for the dashboard's level meter
        self.listening = False

    def _callback(self, indata, n, t, status):
        if not self.muted.is_set():
            self.frames.put(indata[:, 0].copy())

    def utterances(self):
        """Yields float32 audio arrays, one per utterance."""
        import sounddevice as sd
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=FRAME, callback=self._callback):
            buf, silent, speaking = [], 0, False
            while True:
                try:
                    f = self.frames.get(timeout=0.5)
                except queue.Empty:
                    if self.muted.is_set():
                        buf, silent, speaking = [], 0, False
                    continue
                rms = float(np.sqrt(np.mean(f * f)) + 1e-9)
                self.level = rms
                loud = rms > max(self.noise * self.start_factor, 0.006)
                if not speaking:
                    self.noise = 0.95 * self.noise + 0.05 * rms          # adapt to the room
                    buf = (buf + [f])[-10:]                             # keep 0.3 s before speech starts
                    if loud:
                        speaking, silent = True, 0
                    continue
                buf.append(f)
                silent = 0 if loud else silent + 1
                self.listening = True
                if silent >= self.silence_frames or len(buf) >= self.max_frames:
                    speaking, self.listening = False, False
                    if len(buf) - silent >= self.min_frames:
                        yield np.concatenate(buf)
                    buf, silent = [], 0

    # Whisper sometimes "hears" these in noise; never treat them as patient speech
    HALLUCINATIONS = {"thank you.", "thanks for watching!", "thank you for watching.", "you", ".", "bye.",
                      "untertitel im auftrag des zdf, 2020", "vielen dank.", "untertitelung des zdf, 2020",
                      "untertitel der amara.org-community", "copyright wdr 2021"}

    def transcribe(self, audio, lang=None):
        """-> (text, detected_language). lang=None lets Whisper detect English/German."""
        segs, info = self.model.transcribe(
            audio, language=lang, beam_size=1, vad_filter=True, condition_on_previous_text=False,
            initial_prompt="CT, Kontrastmittel, contrast, Metformin, Dialyse, Schilddrüse, ja, nein, yes, no.")
        good = [s.text.strip() for s in segs if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0]
        text = " ".join(t for t in good if t).strip()
        if text.lower() in self.HALLUCINATIONS:
            text = ""
        detected = info.language if info.language in ("en", "de") else None
        return text, detected


class Mouth:
    def __init__(self, ear=None):
        self.ear = ear
        self.has_say = shutil.which("say") is not None
        self.speaking = False

    def say(self, text, voice=None):
        if not text:
            return
        self.speaking = True
        if self.ear:
            self.ear.muted.set()
        try:
            if self.has_say:
                cmd = ["say"] + (["-v", voice] if voice else []) + [text]
                if subprocess.run(cmd, capture_output=True).returncode != 0 and voice:
                    subprocess.run(["say", text], capture_output=True)       # voice not installed: default
            else:
                print(f"  [companion] {text}", flush=True)
                time.sleep(min(4.0, 0.06 * len(text)))
        finally:
            time.sleep(0.3)                                                   # let the room echo die down
            if self.ear:
                with self.ear.frames.mutex:
                    self.ear.frames.queue.clear()
                self.ear.muted.clear()
            self.speaking = False
