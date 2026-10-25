"""Small, transparent text understanding for EN/DE. Deliberately rule-based: every decision can be
read and tested, nothing medical is generated. Input is the Whisper transcript of one utterance."""
import re
import unicodedata

from rapidfuzz import fuzz

_WORDS = re.compile(r"[a-zäöüß']+")


def norm(text):
    t = unicodedata.normalize("NFC", text or "").lower().replace("’", "'")
    return " ".join(_WORDS.findall(t))


# "not sure" before "no": "I don't know" contains "don't", "nicht sicher" contains "nicht"
UNSURE = {
    "en": ["not sure", "don't know", "dont know", "do not know", "maybe", "possibly", "might be", "could be",
           "i think so", "perhaps", "no idea", "can't remember", "cannot remember", "not certain", "unsure"],
    "de": ["nicht sicher", "weiß nicht", "weiss nicht", "vielleicht", "eventuell", "möglicherweise", "kann sein",
           "könnte sein", "keine ahnung", "unsicher", "ich glaube schon", "weiß ich nicht", "weiss ich nicht"],
}
NO = {
    "en": ["no", "nope", "not", "never", "don't", "dont", "haven't", "havent", "hasn't", "isn't", "i'm not",
           "none", "nah", "negative"],
    "de": ["nein", "nö", "nee", "nicht", "kein", "keine", "keinen", "nie", "niemals", "noch nie"],
}
YES = {
    "en": ["yes", "yeah", "yep", "yup", "i do", "i am", "i have", "i did", "correct", "right", "sure", "absolutely",
           "affirmative", "of course", "that's right", "indeed"],
    "de": ["ja", "jawohl", "genau", "doch", "stimmt", "richtig", "natürlich", "klar", "habe ich", "bin ich", "tue ich"],
}


LEAD_YES = {"en": {"yes", "yeah", "yep", "yup", "sure", "correct"}, "de": {"ja", "jawohl", "genau", "doch", "stimmt"}}
LEAD_NO = {"en": {"no", "nope", "nah", "never"}, "de": {"nein", "nö", "nee", "nie"}}


def _has(t, phrases):
    padded = f" {t} "
    return any(f" {p} " in padded for p in phrases)


def yes_no(text, lang):
    """'yes' | 'no' | 'unsure' | None (not an answer)."""
    t = norm(text)
    if not t:
        return None
    if _has(t, UNSURE[lang]):
        return "unsure"
    first = t.split()[0]            # "Yes, but it's not bad" / "Nein, nie" -> the first word decides
    if first in LEAD_YES[lang]:
        return "yes"
    if first in LEAD_NO[lang]:
        return "no"
    if _has(t, NO[lang]):
        return "no"
    if _has(t, YES[lang]):
        return "yes"
    return None


def is_urgent(text, phrases):
    t = norm(text)
    return any(norm(p) and f" {norm(p)} " in f" {t} " for p in phrases)


LANG_WORDS = {
    "de": ["deutsch", "german", "auf deutsch", "germany"],
    "en": ["english", "englisch", "in english", "auf englisch"],
}


def language_choice(text):
    t = norm(text)
    for lang, words in LANG_WORDS.items():
        if _has(t, words):
            return lang
    return None


def wants_repeat(text, words):
    t = norm(text)
    return len(t.split()) <= 4 and _has(t, [norm(w) for w in words])


QUESTION_START = {
    "en": ["how", "what", "why", "when", "where", "who", "can", "could", "will", "is", "are", "do", "does", "should", "am", "may", "i am", "i'm"],
    "de": ["wie", "was", "warum", "wann", "wo", "wer", "kann", "darf", "muss", "ist", "sind", "tut", "wird", "soll", "brauche", "ich habe"],
}


def looks_like_question(text, lang):
    t = norm(text)
    return text.strip().endswith("?") or any(t.startswith(w + " ") for w in QUESTION_START[lang])


def match_faq(text, faq, threshold=78):
    """Best approved answer for a free question, or None. Only answers from the approved list."""
    t = norm(text)
    if len(t.split()) < 2:
        return None
    best, best_score = None, 0
    for item in faq:
        for q in item["q"]:
            s = fuzz.token_set_ratio(t, norm(q))
            if s > best_score:
                best, best_score = item, s
    return best["a"] if best is not None and best_score >= threshold else None
