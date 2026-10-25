"""Microphone segmentation without a microphone or model: fake audio stream.  Run: python test_speech.py"""
import sys
import threading
import types

import numpy as np

import speech


def fake_sounddevice(signal):
    class Stream:
        def __init__(self, callback, **kw):
            self.cb = callback
        def __enter__(self):
            def feed():
                for i in range(0, len(signal), speech.FRAME):
                    self.cb(signal[i:i + speech.FRAME].reshape(-1, 1), speech.FRAME, None, None)
            threading.Thread(target=feed, daemon=True).start()
            return self
        def __exit__(self, *a):
            pass
    sys.modules["sounddevice"] = types.SimpleNamespace(InputStream=Stream)


def test_cuts_two_utterances():
    rng = np.random.default_rng(0)
    quiet = lambda s: (0.002 * rng.standard_normal(int(s * speech.RATE))).astype(np.float32)
    talk = lambda s: (0.1 * np.sin(np.arange(int(s * speech.RATE)) * 2 * np.pi * 220 / speech.RATE)).astype(np.float32)
    signal = np.concatenate([quiet(1.0), talk(1.2), quiet(1.2), talk(0.8), quiet(1.2), talk(0.1), quiet(1.5)])
    fake_sounddevice(signal)
    ear = speech.Ear.__new__(speech.Ear)                     # skip loading Whisper
    ear.silence_frames, ear.max_frames = int(0.8 * speech.RATE / speech.FRAME), int(15 * speech.RATE / speech.FRAME)
    ear.min_frames, ear.start_factor = int(0.3 * speech.RATE / speech.FRAME), 3.0
    import queue
    ear.frames, ear.muted, ear.noise, ear.level, ear.listening = queue.Queue(), threading.Event(), 0.003, 0.0, False
    got = []
    gen = ear.utterances()
    for _ in range(2):
        got.append(len(next(gen)) / speech.RATE)
    assert 1.1 < got[0] < 2.4 and 0.7 < got[1] < 2.0, got     # speech + trailing silence, 0.1 s click ignored
    print(f"ok  stream cut into utterances of {got[0]:.1f} s and {got[1]:.1f} s; short click ignored")


def test_muted_while_speaking():
    m = speech.Mouth()
    m.has_say = False
    ear = types.SimpleNamespace(muted=threading.Event(), frames=__import__("queue").Queue())
    m.ear = ear
    seen = []
    orig = speech.time.sleep
    speech.time.sleep = lambda s: seen.append(ear.muted.is_set())
    try:
        m.say("Hello")
    finally:
        speech.time.sleep = orig
    assert seen[0] is True and not ear.muted.is_set()
    print("ok  microphone muted while the companion speaks")


test_cuts_two_utterances()
test_muted_while_speaking()
print("all speech tests passed")
