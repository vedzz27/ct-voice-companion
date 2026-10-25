# CT Voice Companion

A bilingual (English / German) voice assistant that talks with patients before a CT scan.
It listens through the microphone, transcribes speech **locally with Whisper**, asks the standard
pre-CT safety questions, answers common questions from an **approved** list, explains how to
prepare, and **alerts staff** to anything that needs attention.

> Research prototype, not a medical device. The companion never makes medical decisions: it asks,
> explains approved information and passes everything relevant to the radiology staff.
> All patient-facing text must be reviewed and approved by the radiology team before use.

## What it does

1. **Greets** the patient and lets them choose English or German (or detects the language from
   their speech).
2. **Safety checklist**, one question at a time, answered with yes / no / not sure:
   possible pregnancy, breastfeeding, previous reaction to contrast dye, other allergies, asthma,
   kidney disease or dialysis, diabetes / metformin, thyroid condition, heart disease, regular
   medication, pain / anxiety / difficulty lying still. For some answers it asks for details.
3. **Preparation instructions** at the end: removing metal objects, what the scan is like,
   what contrast dye may feel like, and when to tell staff.
4. **Questions at any time** ("How long does it take?", "Does it hurt?", "Darf ich essen?").
   Answers come **only** from the approved list in `content/`; anything else is passed to staff.
5. **Staff dashboard** at <http://localhost:8002>: alerts by severity, all answers, the
   conversation, a printable summary, and "New patient" to start over.

### Safety rules built in

- **Never guesses**: an answer it can't understand twice is recorded as *unsure* and flagged.
- **"Not sure" counts as a flag**: e.g. unsure about pregnancy → HIGH alert.
- **Urgent phrases** in either language ("chest pain", "can't breathe", "Atemnot", "Hilfe")
  raise an URGENT alert immediately, at any point in the conversation.
- **No generated medical content**: no LLM writes answers; replies are approved text only.
- **Privacy**: audio is processed in memory on this computer and never stored or uploaded; the
  transcript is cleared with "New patient".

## Run it (macOS)

```bash
python3.12 -m venv ~/venvs/ctvoice          # keep environments outside iCloud folders
source ~/venvs/ctvoice/bin/activate
pip install -r requirements.txt

python companion.py                          # microphone + spoken replies
```

- First start downloads the Whisper model (`small`, ~500 MB). Allow microphone access for
  Terminal when macOS asks.
- Open <http://localhost:8002> for the staff dashboard.
- Spoken replies use the macOS voices *Samantha* (English) and *Anna* (German). If Anna is
  missing: System Settings → Accessibility → Spoken Content → System voice → Manage Voices → German.

Options:

```bash
python companion.py --text            # no microphone: type the patient's answers
python companion.py --model medium    # more accurate Whisper, slower (tiny/base/small/medium/large-v3)
python companion.py --port 8010
```

## Change the content

Everything the companion says is in `content/en.json` and `content/de.json`: questions, which
answers raise which alert (`flag_on`, `severity`), follow-up questions, the instructions at the end,
approved Q&A, and urgent phrases. Edit these files (with the radiology team), not the code.
`python test_dialog.py` checks that both languages still match item by item.

## Tests

```bash
python test_dialog.py     # yes/no/unsure in EN + DE, full checklists, alerts, urgent phrases, Q&A
python test_speech.py     # microphone segmentation (simulated audio), mic muted while speaking
```

## How it works

```
microphone ─► energy-based voice detection ─► faster-whisper (local, EN/DE)
                                                   │ transcript + detected language
                                                   ▼
                       dialog.py  (language → checklist → Q&A, rules from content/*.json)
                          │                         │
                          ▼                         ▼
                 macOS `say` (mic muted)     staff dashboard + alerts (FastAPI)
```

| File | Purpose |
|---|---|
| `companion.py` | starts everything: microphone loop, speaker, dashboard API |
| `speech.py` | microphone segmentation, Whisper transcription, text-to-speech |
| `dialog.py` | conversation state machine, answers, alerts |
| `nlu.py` | yes / no / unsure, urgent phrases, approved-answer matching (EN + DE) |
| `content/*.json` | all patient-facing text and alert rules |
| `static/index.html` | staff dashboard |

## Sources for the checklist

Built from common pre-CT screening topics, e.g. RadiologyInfo.org,
["How to prepare for your CT exam"](https://www.radiologyinfo.org/en/info/how-to-prepare-your-ct-exam),
and contrast-media guidance (ACR Manual on Contrast Media, ESUR guidelines). Local hospital
protocols take precedence and must be used to review the content.

## Connect to CT Scan Companion (later)

The staff API (`/api/state`, `/api/reset` with `patient_id`, `/api/report`) is designed so the
camera-based [CT Scan Companion](https://github.com/vedzz27/ct-scan-companion) can start a voice
session for the patient the robot is with, and show the alerts on its dashboard.

## Author

Vedika Darokar · [GitHub](https://github.com/vedzz27) · [LinkedIn](https://www.linkedin.com/in/vedikadarokar)
