"""Conversation + understanding tests, no microphone needed.  Run: python test_dialog.py"""
import nlu
from dialog import Session


def test_yes_no_en():
    cases = {"Yes": "yes", "yeah I do": "yes", "No": "no", "no, never": "no", "I'm not": "no",
             "I don't know": "unsure", "maybe": "unsure", "not sure": "unsure", "No idea": "unsure",
             "Yes, but it's not bad": "yes", "I have no allergies": "no", "I think so": "unsure",
             "the weather is nice": None}
    for text, want in cases.items():
        assert nlu.yes_no(text, "en") == want, (text, nlu.yes_no(text, "en"), want)
    print("ok  yes/no/unsure (English)")


def test_yes_no_de():
    cases = {"Ja": "yes", "ja, habe ich": "yes", "Nein": "no", "nein, noch nie": "no", "Ich weiß nicht": "unsure",
             "vielleicht": "unsure", "Keine Ahnung": "unsure", "Ja, aber nicht schlimm": "yes",
             "Ich habe keine Allergien": "no", "Das Wetter ist schön": None}
    for text, want in cases.items():
        assert nlu.yes_no(text, "de") == want, (text, nlu.yes_no(text, "de"), want)
    print("ok  ja/nein/unsicher (German)")


def run(s, *utterances, lang=None):
    out = []
    for u in utterances:
        out.append(s.handle(u, detected_lang=lang))
    return out


def test_full_checklist_en():
    s = Session()
    s.start()
    replies = run(s, "English please", "No", "No", "Yes", "Rash after a CT last year", "No", "No", "No",
                  "Yes", "Metformin twice a day", "No", "No", "No", "No")
    assert s.state == "done", s.state
    v = s.view()
    notes = [a["note"] for a in v["alerts"]]
    assert notes[0] == "Previous contrast reaction" and v["alerts"][0]["severity"] == "high"
    assert v["alerts"][0]["detail"] == "Rash after a CT last year"
    assert any(n.startswith("Diabetes") for n in notes)
    assert "remove metal objects" in " ".join(replies[-1]).lower()
    assert v["progress"] == "11/11"
    print("ok  full English checklist, flags + details for staff, metal instructions at the end")


def test_full_checklist_de_and_unsure():
    s = Session()
    s.start()
    run(s, "Deutsch")
    assert s.lang == "de" and "schwanger" in s.transcript[-1]["text"]
    run(s, "Ich bin mir nicht sicher")                    # pregnancy unsure -> high alert
    a = s.view()["alerts"][0]
    assert a["severity"] == "high" and "unsicher" not in a["note"] and "(unsure)" in a["note"]
    print("ok  German; 'not sure' about pregnancy goes to staff as HIGH")


def test_language_from_whisper_detection():
    s = Session()
    s.start()
    run(s, "Guten Tag", lang="de")
    assert s.lang == "de"
    print("ok  language taken from Whisper's detection if not said explicitly")


def test_unclear_twice_becomes_unsure():
    s = Session()
    s.start()
    run(s, "English")
    r = run(s, "the weather is nice")
    assert "didn't catch" in r[0][0]
    run(s, "blue")
    assert s.answers["pregnancy"]["answer"] == "unsure" and s.view()["alerts"]
    print("ok  never guesses: unclear twice -> marked unsure for staff")


def test_urgent_any_time_both_languages():
    for text in ("I have chest pain", "Ich habe Atemnot"):
        s = Session()
        s.start()
        run(s, "English")
        r = run(s, text)
        assert s.view()["alerts"][0]["severity"] == "urgent"
        assert "calling a member of staff" in r[0][0]
        assert s.index == 0                                   # the checklist waits
    print("ok  urgent phrases (EN + DE) alert staff immediately")


def test_question_during_checklist_and_after():
    s = Session()
    s.start()
    run(s, "English")
    r = run(s, "How long does the scan take?")
    assert "few minutes" in r[0][0] and "pregnant" in r[0][-1]   # answered, then back to the question
    for _ in range(11):
        run(s, "No")
    r = run(s, "Will it hurt?")
    assert "doesn't hurt" in r[0][0]
    r = run(s, "Can I bring my dog to the scan room tomorrow?")
    assert "pass it on to the staff" in r[0][0]
    assert any(a["note"] == "Patient question for staff" for a in s.view()["alerts"])
    print("ok  approved answers only; unknown questions go to staff")


def test_repeat_and_switch_language():
    s = Session()
    s.start()
    run(s, "English")
    r = run(s, "Pardon?")
    assert "pregnant" in r[0][0]
    r = run(s, "Deutsch")
    assert s.lang == "de" and "schwanger" in r[0][0]
    print("ok  'pardon?' repeats; 'Deutsch' switches language mid-way")


def test_content_complete():
    from dialog import load_content
    en, de = load_content("en"), load_content("de")
    assert [i["id"] for i in en["checklist"]] == [i["id"] for i in de["checklist"]]
    for c in (en, de):
        for i in c["checklist"]:
            assert i["severity"] in ("high", "medium", "info") and i["flag_on"]
    print("ok  English and German checklists match item by item")


for t in (test_yes_no_en, test_yes_no_de, test_full_checklist_en, test_full_checklist_de_and_unsure,
          test_language_from_whisper_detection, test_unclear_twice_becomes_unsure, test_urgent_any_time_both_languages,
          test_question_during_checklist_and_after, test_repeat_and_switch_language, test_content_complete):
    t()
print("all dialog tests passed")
