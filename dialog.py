"""Conversation logic: language choice -> safety checklist -> approved Q&A.
Pure Python, no audio: feed it transcripts, get back what to say. Fully testable."""
import json
import os
import threading
import time

import nlu

CONTENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content")
SEVERITY_ORDER = {"urgent": 0, "high": 1, "medium": 2, "info": 3}


def load_content(lang):
    with open(os.path.join(CONTENT_DIR, f"{lang}.json"), encoding="utf-8") as f:
        return json.load(f)


class Session:
    LANGS = ("en", "de")

    def __init__(self, patient_id=None, default_lang="en"):
        self.content = {l: load_content(l) for l in self.LANGS}
        self.lock = threading.Lock()
        self.default_lang = default_lang
        self.reset(patient_id)

    # ---- state -------------------------------------------------------------------------
    def reset(self, patient_id=None):
        with getattr(self, "lock", threading.Lock()):
            self.patient_id = patient_id
            self.lang = self.default_lang
            self.state = "language"          # language -> checklist -> detail -> done
            self.index = 0
            self.retries = 0
            self.answers = {}                # id -> {question, answer, detail, said}
            self.alerts = []
            self.transcript = []
            self.last_prompt = []
            self.started = time.time()

    @property
    def c(self):
        return self.content[self.lang]

    def _item(self):
        return self.c["checklist"][self.index]

    # ---- output helpers -------------------------------------------------------------------
    def _say(self, *texts):
        out = [t for t in texts if t]
        for t in out:
            self.transcript.append({"t": time.time(), "who": "companion", "text": t, "lang": self.lang})
        return out

    def _alert(self, severity, note, text="", item=None):
        self.alerts.append({"t": time.time(), "severity": severity, "note": note, "said": text, "item": item})

    def _ask_current(self, *before):
        self.last_prompt = [self._item()["question"]]
        return self._say(*before, self._item()["question"])

    def _next(self, *before):
        self.index += 1
        self.retries = 0
        if self.index >= len(self.c["checklist"]):
            self.state = "done"
            self.last_prompt = list(self.c["after_checklist"])
            return self._say(*before, *self.c["after_checklist"])
        self.state = "checklist"
        return self._ask_current(*before)

    # ---- entry points ----------------------------------------------------------------------
    def start(self):
        with self.lock:
            self.last_prompt = [self.content["en"]["greeting"]]
            return self._say(self.content["en"]["greeting"])

    def handle(self, text, detected_lang=None):
        """One patient utterance (Whisper transcript) -> list of sentences to speak."""
        text = (text or "").strip()
        if not text:
            return []
        with self.lock:
            self.transcript.append({"t": time.time(), "who": "patient", "text": text, "lang": detected_lang or self.lang})
            return self._handle(text, detected_lang)

    def _handle(self, text, detected_lang):
        # 1) safety first: urgent phrases in either language, in any state
        if any(nlu.is_urgent(text, self.content[l]["urgent"]) for l in self.LANGS):
            self._alert("urgent", "Patient may need help NOW", text)
            return self._say(self.c["urgent_reply"])

        # 2) language: chosen at the start, or switched any time with a short "Deutsch" / "English"
        choice = nlu.language_choice(text)
        if self.state == "language":
            self.lang = choice or (detected_lang if detected_lang in self.LANGS else self.default_lang)
            self.state, self.index = "checklist", 0
            return self._ask_current(self.c["start"])
        if choice and len(nlu.norm(text).split()) <= 3 and choice != self.lang:
            self.lang = choice
            return self._repeat()

        # 3) "pardon?" -> say the last prompt again
        if nlu.wants_repeat(text, self.c["repeat_words"]):
            return self._repeat()

        if self.state == "detail":
            return self._got_detail(text)
        if self.state == "checklist":
            return self._got_answer(text)
        return self._free_question(text)                    # done: Q&A only

    def _repeat(self):
        if self.state in ("checklist", "detail"):
            return self._ask_current() if self.state == "checklist" else self._say(self.c["detail_prompt"])
        return self._say(*self.last_prompt)

    def _got_answer(self, text):
        item = self._item()
        ans = nlu.yes_no(text, self.lang)
        if ans is None:
            if nlu.looks_like_question(text, self.lang):     # a question in the middle of the checklist
                reply = self._answer_question(text)
                return self._say(reply, self.c["back_to_checklist"]) + self._ask_current()
            self.retries += 1
            if self.retries < 2:
                return self._ask_current(self.c["not_understood"])
            ans = "unsure"                                   # never guess: unclear twice -> staff checks
        self.answers[item["id"]] = {"question": item["question"], "answer": ans, "detail": "", "said": text,
                                    "flagged": ans in item["flag_on"] or ans == "unsure"}
        if ans in item["flag_on"] or ans == "unsure":
            sev = item["severity"] if ans in item["flag_on"] else "info"
            self._alert(sev, item["staff_note"] + (" (unsure)" if ans == "unsure" else ""), text, item["id"])
        if ans in ("yes", "unsure") and item.get("ask_detail"):
            self.state = "detail"
            self.last_prompt = [self.c["detail_prompt"]]
            pre = self.c["unsure_ack"] if ans == "unsure" else None
            return self._say(pre, self.c["detail_prompt"])
        if ans == "unsure":
            return self._next(self.c["unsure_ack"])
        return self._next(item.get("ack_yes") if ans == "yes" else None)

    def _got_detail(self, text):
        item = self._item()
        self.answers[item["id"]]["detail"] = text
        for a in reversed(self.alerts):
            if a["item"] == item["id"]:
                a["detail"] = text
                break
        self.state = "checklist"
        return self._next(self.c["detail_ack"], item.get("ack_yes"))

    def _answer_question(self, text):
        a = nlu.match_faq(text, self.c["faq"])
        if a:
            return a
        self._alert("info", "Patient question for staff", text)
        return self.c["unknown_question"]

    def _free_question(self, text):
        self.last_prompt = [self._answer_question(text)]
        return self._say(*self.last_prompt)

    # ---- staff view ------------------------------------------------------------------------
    def view(self):
        with self.lock:
            total = len(self.c["checklist"])
            return {
                "patient_id": self.patient_id, "lang": self.lang, "state": self.state,
                "progress": f"{min(self.index, total)}/{total}",
                "alerts": sorted(self.alerts, key=lambda a: (SEVERITY_ORDER[a["severity"]], a["t"])),
                "answers": [dict(id=i["id"], **self.answers[i["id"]]) for i in self.c["checklist"] if i["id"] in self.answers],
                "transcript": self.transcript[-60:],
            }
