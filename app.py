"""
CrimeVision AI - Intelligent Multimodal Forensic Analysis Platform
Skill Wallet Capstone Project 2

Modules (Kanban epics):
  Epic 1  Configuration, security, chain-of-custody
  Epic 2  Crime-scene visual evidence tagger (RT-DETR style detection)
  Epic 3  Witness audio transcription (Whisper) + timeline extraction
  Epic 4  CCTV video tracking + indicative penal classification
  Epic 5  Multimodal fusion ledger + grounded evidence chat (RAG)
  Epic 6  Dashboard + ReportLab PDF export

Run:  streamlit run app.py
"""
import base64
import hashlib
import io
import json
import os
import random
import re
import tempfile
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

try:
    import cv2
    CV2_OK = True
except Exception:  # pragma: no cover
    CV2_OK = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle, PageBreak)
    RL_OK = True
except Exception:  # pragma: no cover
    RL_OK = False

# --------------------------------------------------------------------------- #
# Constants (change model names here if a provider retires one)
# --------------------------------------------------------------------------- #
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_WHISPER_MODEL = "whisper-large-v3"
GEMINI_MODEL = "gemini-2.0-flash"

LABELS = ["Person", "Firearm", "Edged weapon", "Blood spatter", "Forced entry point"]
LABEL_COLORS = {"Person": (66, 165, 245), "Firearm": (239, 83, 80), "Edged weapon": (255, 112, 67),
                "Blood spatter": (171, 71, 188), "Forced entry point": (255, 202, 40)}
THREAT_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#ef6c00", "Critical": "#c62828"}
CITE_RX = r"\b(?:IMG|AUD|VID)-[0-9A-F]{8}-E\d{2}\b"

CONCEPTS = {
    "weapon": r"weapon|firearm|gun|pistol|knife|edged|crowbar|blade",
    "flee": r"flee|fled|fleeing|running|escape|flight",
    "entry": r"forced entry|break|entry|window",
    "gunshot": r"gunshot|shot|firearm|gun",
    "vehicle": r"vehicle|car|sedan|truck|van|bike",
    "blood": r"blood|injur|wound|bleed",
}

DEMO_SEGS = [
    {"start": 0.0, "end": 6.5, "speaker": "Dispatcher", "text": "112 emergency, what is the address of your emergency?"},
    {"start": 6.5, "end": 14.0, "speaker": "Caller", "text": "A man is breaking the back window of the house next door. I can see a crowbar in his hand."},
    {"start": 14.0, "end": 22.0, "speaker": "Dispatcher", "text": "Stay on the line. Can you describe him?"},
    {"start": 22.0, "end": 33.0, "speaker": "Caller", "text": "Tall, wearing a black hoodie. He just went inside. I heard a loud bang, maybe a gunshot."},
    {"start": 33.0, "end": 41.0, "speaker": "Caller", "text": "He is running out of the front door now, towards a white sedan."},
    {"start": 41.0, "end": 48.0, "speaker": "Dispatcher", "text": "Units are on the way. Do not approach him."},
]

AUDIO_RULES = [
    (r"break|smash|crowbar|forced", "Forced entry reported", "Forced entry"),
    (r"gun|shot|shoot|firearm|pistol|bang", "Possible gunshot / firearm reported", "Weapon / gunfire"),
    (r"knife|blade|stab", "Edged weapon reported", "Weapon"),
    (r"blood|bleeding|injur|wound", "Injury or blood reported", "Injury"),
    (r"running|ran away|fled|flee|escap", "Suspect fleeing the scene", "Suspect movement"),
    (r"car|sedan|vehicle|bike|truck|van\b", "Suspect vehicle reported", "Vehicle"),
    (r"hoodie|tall|wearing|height|description", "Suspect description provided", "Description"),
]

LEGAL = {
    "enter": "Observation - no offence indicated",
    "exit": "Observation - subject leaves / lost from view",
    "loiter_p": "Suspicious activity - possible reconnaissance (no offence per se)",
    "loiter_v": "Observation - stationary vehicle (record make/plate for tracing)",
    "flee_p": "Possible flight from scene - corroborate with other evidence",
    "flee_v": "Possible getaway / rash driving - BNS s.281 if on public way",
    "prox": "Possible physical altercation - BNS s.115 (hurt) if injury established",
}

# --------------------------------------------------------------------------- #
# Page setup + styling
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="CrimeVision AI", page_icon="🕵️", layout="wide")
st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 15% 0%, #0f1b2d 0%, #0a0f18 55%);}
h1,h2,h3 {letter-spacing:.3px;}
.cv-hero {border:1px solid #1f3a5a; border-left:5px solid #29b6f6; border-radius:10px;
  padding:14px 20px; background:linear-gradient(90deg,#0d2238,#0a1626); margin-bottom:10px;}
.cv-hero h1 {margin:0; font-size:1.6rem; color:#e3f2fd;}
.cv-hero p {margin:2px 0 0 0; color:#90a4ae; font-size:.9rem;}
.cv-badge {display:inline-block; padding:3px 12px; border-radius:20px; font-weight:700; font-size:.8rem; color:#fff;}
.cv-card {border:1px solid #1f3a5a; border-radius:10px; padding:12px 16px; background:#0c1a2b; margin-bottom:8px;}
.cv-mono {font-family:monospace; color:#80cbc4; font-size:.85rem;}
.cv-sim {color:#ffb74d; font-size:.8rem; font-weight:600;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hms_to_sec(s: str) -> float:
    try:
        sec = 0.0
        for p in str(s).strip().split(":"):
            sec = sec * 60 + float(p)
        return sec
    except Exception:
        return 0.0


def sec_to_hms(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def to_jpeg(data: bytes, max_side: int = 1024) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=88)
    return out.getvalue()


def parse_json(txt: str) -> dict:
    txt = re.sub(r"```(?:json)?", "", txt or "").strip()
    s, e = txt.find("{"), txt.rfind("}")
    return json.loads(txt[s:e + 1])


def get_secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.environ.get(name, "")
    except Exception:
        return os.environ.get(name, "")


# --------------------------------------------------------------------------- #
# Session state, chain of custody, ledger
# --------------------------------------------------------------------------- #
def init_state():
    defaults = {"ledger": {}, "registry": {}, "custody": [], "results": {"IMG": {}, "AUD": {}, "VID": {}},
                "last": {}, "chat": [], "signed_at": None, "pdf": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def log_custody(action: str, evid: str = "-"):
    st.session_state.custody.append({
        "Timestamp": now_utc(), "Evidence ID": evid, "Action": action,
        "Handler": st.session_state.get("investigator") or "Unassigned"})


def register(prefix: str, name: str, data: bytes, kind: str) -> str:
    digest = sha256(data)
    evid = f"{prefix}-{digest[:8].upper()}"
    if evid not in st.session_state.registry:
        st.session_state.registry[evid] = {
            "Evidence ID": evid, "File": name, "Type": kind, "Size (KB)": round(len(data) / 1024, 1),
            "SHA-256": digest, "Logged": now_utc()}
        log_custody(f"REGISTERED {kind} '{name}' (SHA-256 {digest[:12]}...)", evid)
    return evid


def set_ledger(evid, source, base_clock, items, simulated):
    entries = []
    for n, it in enumerate(items, 1):
        clock = base_clock + float(it.get("offset", 0))
        entries.append({
            "entry_id": f"{evid}-E{n:02d}", "evidence_id": evid, "source": source, "clock": clock,
            "time": sec_to_hms(clock), "event": it["event"], "category": it.get("category", "Observation"),
            "confidence": round(float(it.get("confidence", 0.0)), 2), "detail": it.get("detail", ""),
            "simulated": simulated})
    st.session_state.ledger[evid] = entries


def flat_ledger():
    rows = [e for v in st.session_state.ledger.values() for e in v]
    return sorted(rows, key=lambda e: (e["clock"], e["entry_id"]))


# --------------------------------------------------------------------------- #
# LLM / Whisper provider wrapper (Groq or Gemini) with demo fallback
# --------------------------------------------------------------------------- #
class LLM:
    def __init__(self, provider: str, key: str):
        self.provider, self.key = provider, (key or "").strip()
        self.live = bool(self.key)

    def _gem(self, system=None):
        import google.generativeai as genai
        genai.configure(api_key=self.key)
        return genai.GenerativeModel(GEMINI_MODEL, system_instruction=system)

    def text(self, system: str, user: str, json_mode=False) -> str:
        if self.provider == "Groq":
            from groq import Groq
            kw = {"response_format": {"type": "json_object"}} if json_mode else {}
            r = Groq(api_key=self.key).chat.completions.create(
                model=GROQ_TEXT_MODEL, temperature=0, **kw,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return r.choices[0].message.content
        cfg = {"temperature": 0}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        return self._gem(system).generate_content(user, generation_config=cfg).text

    def vision(self, system: str, user: str, images: list) -> str:
        if self.provider == "Groq":
            from groq import Groq
            content = [{"type": "text", "text": user}]
            for b in images:
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}})
            r = Groq(api_key=self.key).chat.completions.create(
                model=GROQ_VISION_MODEL, temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": content}])
            return r.choices[0].message.content
        pil = [Image.open(io.BytesIO(b)) for b in images]
        return self._gem(system).generate_content([user] + pil, generation_config={"temperature": 0}).text

    def transcribe(self, data: bytes, filename: str) -> list:
        """Return [{'start','end','text'}]."""
        if self.provider == "Groq":
            from groq import Groq
            r = Groq(api_key=self.key).audio.transcriptions.create(
                file=(filename, data), model=GROQ_WHISPER_MODEL, response_format="verbose_json")
            segs = getattr(r, "segments", None) or (r.get("segments") if isinstance(r, dict) else None) or []
            out = []
            for s in segs:
                g = (lambda k: s[k]) if isinstance(s, dict) else (lambda k: getattr(s, k))
                out.append({"start": float(g("start")), "end": float(g("end")), "text": str(g("text")).strip()})
            if not out:
                out = [{"start": 0.0, "end": 0.0, "text": str(getattr(r, "text", "")).strip()}]
            return out
        ext = filename.rsplit(".", 1)[-1].lower()
        mime = {"mp3": "audio/mp3", "wav": "audio/wav", "m4a": "audio/mp4", "ogg": "audio/ogg",
                "flac": "audio/flac"}.get(ext, "audio/mp3")
        prompt = ('Transcribe this audio verbatim. Return ONLY JSON: {"segments":[{"start":sec,"end":sec,"text":str}]}'
                  " with realistic timestamps in seconds. Do not add content that is not spoken.")
        raw = self._gem().generate_content([{"mime_type": mime, "data": data}, prompt],
                                           generation_config={"temperature": 0}).text
        return parse_json(raw)["segments"]


# --------------------------------------------------------------------------- #
# Epic 2 - Visual evidence tagger
# --------------------------------------------------------------------------- #
DETECT_SYSTEM = ("You are a forensic computer-vision detector emulating RT-DETR. Report ONLY objects that are "
                 "genuinely visible. Never invent objects. Respond with JSON only.")
DETECT_USER = ('Analyse this crime-scene image. Return JSON: {"scene_caption": str, "detections": [{"label": one of '
               f'{LABELS}, "confidence": 0-1, "bbox": [x1,y1,x2,y2] normalised 0-1}}], "leads": [up to 5 prioritised '
               'investigative leads]}. Use an empty detections list if nothing relevant is visible.')

LEAD_TEMPLATES = {
    "Person": "Identify the person(s) in frame; cross-match with CCTV tracks and witness descriptions.",
    "Firearm": "Secure firearm for ballistics, fingerprint and serial-number tracing; check for spent casings.",
    "Edged weapon": "Bag and tag blade for DNA/fingerprint lifting; match against wound patterns if any.",
    "Blood spatter": "Request bloodstain pattern analysis and DNA sampling before the scene is disturbed.",
    "Forced entry point": "Photograph tool marks; compare with recovered tools (e.g. crowbar); check neighbours' CCTV.",
}


def demo_scene_image() -> bytes:
    img = Image.new("RGB", (960, 600), (38, 42, 50))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 380, 960, 600], fill=(62, 56, 50))
    d.rectangle([600, 80, 880, 300], outline=(150, 160, 170), width=6)
    d.line([600, 80, 880, 300], fill=(200, 210, 220), width=3)
    d.ellipse([200, 430, 300, 470], fill=(120, 20, 25))
    d.rectangle([380, 250, 440, 420], fill=(90, 100, 120))
    d.ellipse([385, 215, 435, 265], fill=(160, 130, 110))
    d.polygon([(445, 400), (500, 390), (448, 408)], fill=(210, 215, 225))
    d.text((20, 20), "SIMULATED DEMO SCENE", fill=(230, 230, 230))
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def demo_detect(data: bytes, preset: bool) -> list:
    if preset:
        return [{"label": "Person", "confidence": 0.94, "bbox": [0.38, 0.35, 0.47, 0.70]},
                {"label": "Edged weapon", "confidence": 0.81, "bbox": [0.46, 0.64, 0.53, 0.69]},
                {"label": "Blood spatter", "confidence": 0.89, "bbox": [0.19, 0.70, 0.33, 0.80]},
                {"label": "Forced entry point", "confidence": 0.92, "bbox": [0.62, 0.13, 0.93, 0.50]}]
    rng = random.Random(int(sha256(data)[:12], 16))
    labels = ["Person"] + rng.sample(LABELS, rng.randint(2, 4))
    dets = []
    for lb in labels:
        w, h = rng.uniform(.1, .3), rng.uniform(.12, .35)
        x, y = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
        dets.append({"label": lb, "confidence": round(rng.uniform(.6, .98), 2),
                     "bbox": [round(x, 3), round(y, 3), round(x + w, 3), round(y + h, 3)]})
    return dets


def sanitize_dets(raw) -> list:
    out = []
    for d in raw or []:
        try:
            if d["label"] not in LABELS:
                continue
            x1, y1, x2, y2 = [min(1.0, max(0.0, float(v))) for v in d["bbox"]]
            out.append({"label": d["label"], "confidence": min(1.0, max(0.0, float(d["confidence"]))),
                        "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]})
        except Exception:
            continue
    return out


def assess(dets: list) -> dict:
    s = {d["label"] for d in dets}
    weapon = bool({"Firearm", "Edged weapon"} & s)
    if "Blood spatter" in s and weapon:
        return {"threat": "Critical", "scene": "Homicide",
                "why": "Blood spatter co-occurs with a weapon: indicates violent crime with possible fatality."}
    if weapon and "Person" in s:
        return {"threat": "High", "scene": "Assault", "why": "Armed individual present without confirmed fatality."}
    if "Blood spatter" in s:
        return {"threat": "High", "scene": "Assault", "why": "Blood present; violent incident likely."}
    if "Forced entry point" in s:
        return {"threat": "High" if weapon else "Medium", "scene": "Burglary",
                "why": "Forced entry indicators" + (" with weapon (armed burglary)." if weapon else ".")}
    return {"threat": "Low", "scene": "Vandalism",
            "why": "Default class: insufficient violent or entry indicators. Manual review recommended."}


def region(b):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return f"{['top', 'middle', 'bottom'][min(2, int(cy * 3))]}-{['left', 'centre', 'right'][min(2, int(cx * 3))]}"


def annotate(img: Image.Image, dets: list) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    for det in dets:
        c = LABEL_COLORS[det["label"]]
        x1, y1, x2, y2 = det["bbox"][0] * W, det["bbox"][1] * H, det["bbox"][2] * W, det["bbox"][3] * H
        d.rectangle([x1, y1, x2, y2], outline=c, width=max(2, W // 300))
        tag = f"{det['label']} {det['confidence']:.2f}"
        d.rectangle([x1, max(0, y1 - 16), x1 + 7 * len(tag) + 6, y1], fill=c)
        d.text((x1 + 3, max(0, y1 - 14)), tag, fill=(0, 0, 0))
    return img


def process_image(name, data, clock, llm, preset=False):
    evid = register("IMG", name, data, "Image")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    W, H = img.size
    dets, caption, leads, simulated = None, "", None, True
    mode = "SIMULATED - RT-DETR emulation (no API key / fallback)"
    if llm.live:
        try:
            j = parse_json(llm.vision(DETECT_SYSTEM, DETECT_USER, [to_jpeg(data)]))
            dets, caption, leads = sanitize_dets(j.get("detections")), j.get("scene_caption", ""), j.get("leads")
            simulated, mode = False, f"LIVE - {llm.provider} vision model"
        except Exception as ex:
            st.warning(f"Live vision failed ({ex}); using simulated detector.")
    if dets is None:
        dets = demo_detect(data, preset)
        caption = "Simulated scene caption: detections are generated by the demo detector, not by analysing pixels."
    a = assess(dets)
    if not leads:
        leads = [LEAD_TEMPLATES[l] for l in dict.fromkeys(d["label"] for d in sorted(
            dets, key=lambda x: -LABELS.index(x["label"])))]
    ann = annotate(img, dets)
    buf = io.BytesIO()
    ann.save(buf, "PNG")
    items = []
    ev_text = {"Person": "Person present in scene", "Firearm": "Weapon visible: firearm",
               "Edged weapon": "Weapon visible: edged weapon", "Blood spatter": "Blood spatter observed",
               "Forced entry point": "Forced entry point observed"}
    for d in dets:
        px = [int(d["bbox"][0] * W), int(d["bbox"][1] * H), int(d["bbox"][2] * W), int(d["bbox"][3] * H)]
        items.append({"offset": 0, "event": f"{ev_text[d['label']]} ({region(d['bbox'])})", "category": d["label"],
                      "confidence": d["confidence"], "detail": f"bbox px {px}"})
    items.append({"offset": 0, "event": f"Scene classified as {a['scene']}, threat {a['threat']}",
                  "category": "Scene classification", "confidence": 0.8, "detail": a["why"]})
    set_ledger(evid, "Image", clock, items, simulated)
    st.session_state.results["IMG"][evid] = {
        "evid": evid, "name": name, "mode": mode, "caption": caption, "dets": dets, "assessment": a,
        "leads": leads, "png": buf.getvalue(), "size": (W, H), "clock": clock}
    st.session_state.last["IMG"] = evid
    log_custody("ANALYSED image evidence", evid)
    return evid


# --------------------------------------------------------------------------- #
# Epic 3 - Audio transcription + timeline
# --------------------------------------------------------------------------- #
def keyword_events(segs):
    events = []
    for i, s in enumerate(segs):
        low = s["text"].lower()
        for rx, ev, cat in AUDIO_RULES:
            if re.search(rx, low):
                events.append({"offset": s["start"], "event": ev, "category": cat, "confidence": 0.6,
                               "detail": f"Seg {i} [{s['speaker']}]: {s['text'][:140]}"})
    return events


def enrich_audio(llm, segs):
    lines = "\n".join(f"{i} [{s['start']:.1f}-{s['end']:.1f}s] {s['text']}" for i, s in enumerate(segs))
    system = ("You are a forensic transcript analyst. Use ONLY the transcript. Do not invent facts. JSON only.")
    user = ('For each numbered segment assign a speaker tag (e.g. Dispatcher, Caller, Witness, Suspect, Speaker A) '
            'and extract key events. Return {"speakers":[one tag per segment, in order], "events":[{"segment":int,'
            '"event":str,"category":str,"confidence":0-1}]}.\n\n' + lines)
    j = parse_json(llm.text(system, user, json_mode=True))
    sp = j.get("speakers", [])
    for i, s in enumerate(segs):
        s["speaker"] = sp[i] if i < len(sp) else "Speaker"
    events = []
    for e in j.get("events", []):
        i = int(e["segment"])
        if 0 <= i < len(segs):
            events.append({"offset": segs[i]["start"], "event": e["event"], "category": e.get("category", "Event"),
                           "confidence": float(e.get("confidence", 0.7)),
                           "detail": f"Seg {i} [{segs[i]['speaker']}]: {segs[i]['text'][:140]}"})
    return events


def process_audio(name, data, clock, llm, use_sample=False):
    evid = register("AUD", name, data, "Audio")
    segs, events, simulated = None, None, True
    mode = "SIMULATED - sample 112 call transcript (NOT derived from the file)"
    if llm.live and not use_sample:
        try:
            segs = [dict(s, speaker="Speaker") for s in llm.transcribe(data, name)]
            simulated, mode = False, f"LIVE - {llm.provider} " + ("Whisper large-v3" if llm.provider == "Groq" else "Gemini audio")
            try:
                events = enrich_audio(llm, segs)
            except Exception as ex:
                st.warning(f"Speaker/event enrichment failed ({ex}); using keyword extraction.")
        except Exception as ex:
            st.warning(f"Live transcription failed ({ex}); using sample transcript.")
            segs = None
    if segs is None:
        segs, simulated = [dict(s) for s in DEMO_SEGS], True
    if not events:
        events = keyword_events(segs)
    set_ledger(evid, "Audio", clock, events or [{"offset": 0, "event": "No key events extracted",
                                                  "category": "Observation", "confidence": 0.3}], simulated)
    st.session_state.results["AUD"][evid] = {"evid": evid, "name": name, "mode": mode, "segments": segs,
                                             "events": events, "clock": clock}
    st.session_state.last["AUD"] = evid
    log_custody("TRANSCRIBED audio evidence", evid)
    return evid


# --------------------------------------------------------------------------- #
# Epic 4 - Video tracking + classification
# --------------------------------------------------------------------------- #
TRACK_COLORS = [(66, 165, 245), (239, 83, 80), (102, 187, 106), (255, 202, 40), (171, 71, 188), (255, 112, 67)]


def analyze_video(path, sample_fps=3.0, min_area_pct=0.3, warmup=3):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Video could not be opened (unsupported codec?).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))
    bg = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=40, detectShadows=False)
    kernel = np.ones((3, 3), np.uint8)
    tracks, keyframes, ref, W, H = [], [], None, 480, 270
    idx = sidx = 0
    last_t = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / fps
            last_t = t
            scale = 480 / frame.shape[1] if frame.shape[1] > 480 else 1.0
            small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
            H, W = small.shape[:2]
            mask = bg.apply(small)
            if sidx >= warmup:
                mask = cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), kernel, iterations=2)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                blobs = []
                for c in cnts:
                    a = cv2.contourArea(c)
                    if a < min_area_pct / 100 * W * H:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    kind = "Vehicle" if (w / max(h, 1) > 1.3 and a > 0.02 * W * H) else "Person"
                    blobs.append((x + w / 2, y + h / 2, kind, a))
                taken = set()
                for tr in tracks:
                    tr["miss"] += 1
                for cx, cy, kind, a in blobs:
                    best, bd = None, 0.2 * W
                    for tr in tracks:
                        if tr["miss"] > 4 or tr["id"] in taken or tr["kind"] != kind:
                            continue
                        d = ((tr["pts"][-1][1] - cx) ** 2 + (tr["pts"][-1][2] - cy) ** 2) ** 0.5
                        if d < bd:
                            best, bd = tr, d
                    if best is None:
                        best = {"id": len(tracks) + 1, "kind": kind, "pts": [], "miss": 0}
                        tracks.append(best)
                    best["pts"].append((t, cx, cy))
                    best["miss"] = 0
                    taken.add(best["id"])
                motion = sum(b[3] for b in blobs) / (W * H)
                if ref is None:
                    ref = small.copy()
                if motion > 0:
                    keyframes.append((motion, t, cv2.imencode(".jpg", small)[1].tobytes()))
                    keyframes = sorted(keyframes, key=lambda k: -k[0])[:12]
            sidx += 1
        idx += 1
    cap.release()
    if sidx == 0:
        raise ValueError("No frames could be read from this video.")
    return {"tracks": [t for t in tracks if len(t["pts"]) >= 3], "W": W, "H": H, "ref": ref,
            "keyframes": sorted(keyframes, key=lambda k: -k[0])[:4], "duration": last_t, "fps": fps}


def demo_video():
    W, H = 480, 270

    def seg(t0, t1, p0, p1, dt=0.5):
        n = max(2, int((t1 - t0) / dt) + 1)
        return [(t0 + (t1 - t0) * i / (n - 1), p0[0] + (p1[0] - p0[0]) * i / (n - 1),
                 p0[1] + (p1[1] - p0[1]) * i / (n - 1)) for i in range(n)]
    t1 = seg(2, 8, (20, 200), (240, 180)) + seg(8.5, 13, (240, 180), (242, 181)) + seg(13.3, 14.1, (262, 185), (420, 205), 0.2)
    t2 = seg(1, 3, (470, 215), (440, 210)) + seg(3.5, 17, (440, 210), (441, 210), 1.0) + seg(17.4, 18.6, (430, 210), (60, 210), 0.2)
    t3 = seg(9, 12, (272, 180), (272, 181), 0.5)
    tracks = [{"id": 1, "kind": "Person", "pts": t1}, {"id": 2, "kind": "Vehicle", "pts": t2},
              {"id": 3, "kind": "Person", "pts": t3}]
    return {"tracks": tracks, "W": W, "H": H, "ref": None, "keyframes": [], "duration": 19.0, "fps": 25.0}


def build_video_events(tracks, W):
    ev = []
    for tr in tracks:
        pts, veh = tr["pts"], tr["kind"] == "Vehicle"
        lab = f"{tr['kind']} T{tr['id']}"
        ev.append(dict(offset=pts[0][0], subject=lab, event=f"{lab} enters frame", legal=LEGAL["enter"], confidence=0.7))
        sp = []
        for a, b in zip(pts, pts[1:]):
            dt = max(b[0] - a[0], 1e-3)
            sp.append((((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5 / dt / W, b[0]))
        vmax, tmax = max(sp)
        if vmax > (0.5 if veh else 0.3):
            ev.append(dict(offset=tmax, subject=lab, confidence=min(0.95, 0.55 + vmax / 2),
                           event=f"{lab} rapid movement ({vmax:.2f} frame-widths/s), possible flight",
                           legal=LEGAL["flee_v" if veh else "flee_p"]))
        for i, p in enumerate(pts):
            j = next((j for j in range(i, len(pts)) if pts[j][0] - p[0] >= 3), None)
            if j is None:
                break
            if max(((q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2) ** 0.5 for q in pts[i:j + 1]) < 0.06 * W:
                ev.append(dict(offset=p[0], subject=lab, confidence=0.75,
                               event=f"{lab} {'stationary' if veh else 'loitering'} for >=3 s",
                               legal=LEGAL["loiter_v" if veh else "loiter_p"]))
                break
        ev.append(dict(offset=pts[-1][0], subject=lab, event=f"{lab} leaves frame / lost", legal=LEGAL["exit"],
                       confidence=0.6))
    persons = [t for t in tracks if t["kind"] == "Person"]
    seen = set()
    for i, a in enumerate(persons):
        for b in persons[i + 1:]:
            bt = {round(p[0], 2): p for p in b["pts"]}
            for p in a["pts"]:
                q = bt.get(round(p[0], 2))
                if q and ((p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2) ** 0.5 < 0.08 * W and (a["id"], b["id"]) not in seen:
                    seen.add((a["id"], b["id"]))
                    ev.append(dict(offset=p[0], subject=f"T{a['id']}+T{b['id']}", confidence=0.55, legal=LEGAL["prox"],
                                   event=f"Close-proximity interaction between Person T{a['id']} and Person T{b['id']}"))
    return sorted(ev, key=lambda e: e["offset"])


def llm_keyframe_events(llm, keyframes):
    desc = "; ".join(f"image {i + 1} at t={k[1]:.1f}s" for i, k in enumerate(keyframes))
    user = ('These CCTV keyframes show peak motion (' + desc + '). Describe ONLY observable events. Return JSON: '
            '{"events":[{"t": seconds, "event": str, "legal": indicative offence category under Indian BNS or "Observation", '
            '"confidence": 0-1}]}. Do not speculate beyond what is visible.')
    j = parse_json(llm.vision("You are a CCTV forensic analyst. JSON only.", user, [k[2] for k in keyframes]))
    return [dict(offset=float(e["t"]), subject="LLM vision", event=str(e["event"]), legal=str(e.get("legal", "Observation")),
                 confidence=float(e.get("confidence", .5))) for e in j.get("events", [])]


def draw_tracks(res):
    ref = res.get("ref")
    img = Image.fromarray(ref[:, :, ::-1]) if ref is not None else Image.new("RGB", (res["W"], res["H"]), (18, 24, 32))
    d = ImageDraw.Draw(img)
    for tr in res["tracks"]:
        c = TRACK_COLORS[(tr["id"] - 1) % len(TRACK_COLORS)]
        pts = [(p[1], p[2]) for p in tr["pts"]]
        d.line(pts, fill=c, width=3)
        d.ellipse([pts[0][0] - 5, pts[0][1] - 5, pts[0][0] + 5, pts[0][1] + 5], outline=c, width=2)
        d.ellipse([pts[-1][0] - 5, pts[-1][1] - 5, pts[-1][0] + 5, pts[-1][1] + 5], fill=c)
        d.text((pts[0][0] + 6, pts[0][1] - 14), f"{tr['kind']} T{tr['id']}", fill=c)
    return img


def process_video(name, data, clock, llm, sample_fps, min_area, use_llm, demo=False):
    evid = register("VID", name, data, "Video")
    simulated = demo
    if demo:
        res = demo_video()
        mode = "SIMULATED - synthetic CCTV trajectories (NOT derived from a file)"
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1] or ".mp4") as f:
            f.write(data)
            path = f.name
        try:
            res = analyze_video(path, sample_fps, min_area)
        finally:
            os.unlink(path)
        mode = "REAL - OpenCV MOG2 motion tracking (heuristic, verify manually)"
    events = build_video_events(res["tracks"], res["W"])
    if use_llm and llm.live and res["keyframes"]:
        try:
            events += llm_keyframe_events(llm, res["keyframes"])
            events.sort(key=lambda e: e["offset"])
            mode += f" + {llm.provider} keyframe vision"
        except Exception as ex:
            st.warning(f"Keyframe vision failed ({ex}).")
    items = [{"offset": e["offset"], "event": e["event"], "category": e["legal"], "confidence": e["confidence"],
              "detail": f"subject {e['subject']}"} for e in events] or [
        {"offset": 0, "event": "No significant motion detected", "category": "Observation", "confidence": 0.4}]
    set_ledger(evid, "Video", clock, items, simulated)
    ov = draw_tracks(res)
    buf = io.BytesIO()
    ov.save(buf, "PNG")
    st.session_state.results["VID"][evid] = {
        "evid": evid, "name": name, "mode": mode, "events": events, "clock": clock, "png": buf.getvalue(),
        "tracks": track_table(res["tracks"], res["W"]), "duration": res["duration"]}
    st.session_state.last["VID"] = evid
    log_custody("ANALYSED video evidence", evid)
    return evid


def track_table(tracks, W):
    rows = []
    for tr in tracks:
        p = tr["pts"]
        v = max(((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5 / max(b[0] - a[0], 1e-3) / W for a, b in zip(p, p[1:]))
        rows.append({"Track": f"T{tr['id']}", "Type": tr["kind"], "First seen (s)": round(p[0][0], 1),
                     "Last seen (s)": round(p[-1][0], 1), "Samples": len(p), "Peak speed (frame-widths/s)": round(v, 2)})
    return rows


# --------------------------------------------------------------------------- #
# Epic 5 - Grounded evidence chat
# --------------------------------------------------------------------------- #
CHAT_SYSTEM = (
    "You are the evidence-interrogation assistant of a forensic platform. Answer ONLY from the EVIDENCE LEDGER. "
    "Rules: (1) every factual statement must end with citation(s) in square brackets using the exact entry IDs, "
    "e.g. [IMG-1A2B3C4D-E01]; (2) never use outside knowledge or speculate; if the ledger cannot answer, reply exactly: "
    "'Insufficient evidence in the ledger to answer this.'; (3) use the time column for before/after reasoning and note "
    "that clocks come from investigator-entered source start times; (4) flag SIMULATED or low-confidence (<0.6) items; "
    "(5) be concise.")
INSUFFICIENT = "Insufficient evidence in the ledger to answer this."


def ledger_text(rows):
    return "\n".join(f"[{e['entry_id']}] time={e['time']} | {e['source']} | {e['category']} | {e['event']} | "
                     f"conf={e['confidence']} | {e['detail']}{' | SIMULATED' if e['simulated'] else ''}" for e in rows[:120])


def entry_blob(e):
    return f"{e['event']} {e['category']} {e['detail']}".lower()


def demo_answer(q, rows):
    ql = q.lower()
    found = sorted([(m.start(), n) for n, rx in CONCEPTS.items() for m in [re.search(rx, ql)] if m])
    names = list(dict.fromkeys(n for _, n in found))
    word = "before" if "before" in ql else ("after" if "after" in ql else None)
    if word and len(names) >= 2:
        a_name, b_name = names[0], names[1]
        A = [e for e in rows if re.search(CONCEPTS[a_name], entry_blob(e)) and e["category"] != "Scene classification"]
        B = [e for e in rows if re.search(CONCEPTS[b_name], entry_blob(e)) and e["category"] != "Scene classification"]
        if not B or not A:
            return INSUFFICIENT
        tb = min(e["clock"] for e in B)
        hit = [e for e in A if (e["clock"] < tb if word == "before" else e["clock"] > tb)]
        anchor = min(B, key=lambda e: e["clock"])
        cites = " ".join(f"[{e['entry_id']}]" for e in hit[:4])
        if hit:
            return (f"Yes. {a_name.capitalize()}-related evidence is timestamped {word} the earliest "
                    f"{b_name}-related event ({anchor['time']}, [{anchor['entry_id']}]): "
                    + "; ".join(f"{e['event']} at {e['time']} [{e['entry_id']}]" for e in hit[:4]) +
                    ". Note: ordering depends on the source start clocks you entered.")
        return (f"No {a_name}-related entry is timestamped {word} the earliest {b_name}-related event "
                f"({anchor['time']}, [{anchor['entry_id']}]). Check the source clock offsets.")
    stop = {"the", "was", "were", "and", "did", "does", "what", "when", "who", "any", "there", "visible", "that", "with", "from"}
    toks = {t for t in re.findall(r"[a-z]{3,}", ql) if t not in stop}
    scored = []
    for e in rows:
        b = entry_blob(e)
        s = sum(1 for t in toks if t in b) + sum(2 for n in names if re.search(CONCEPTS[n], b))
        if s:
            scored.append((s, e))
    scored.sort(key=lambda x: (-x[0], x[1]["clock"]))
    if not scored:
        return INSUFFICIENT
    return "Relevant ledger entries:\n" + "\n".join(
        f"- {e['time']} | {e['source']} | {e['event']} (conf {e['confidence']}) [{e['entry_id']}]"
        for _, e in scored[:6])


def answer_question(q, llm):
    rows = flat_ledger()
    if not rows:
        return INSUFFICIENT + " The ledger is empty.", "n/a", []
    mode, ans = "DEMO (extractive)", None
    if llm.live:
        try:
            ans = llm.text(CHAT_SYSTEM, f"EVIDENCE LEDGER:\n{ledger_text(rows)}\n\nQUESTION: {q}")
            valid = {e["entry_id"] for e in rows}
            cited = set(re.findall(CITE_RX, ans))
            if (cited - valid) or (not cited and INSUFFICIENT not in ans):
                st.warning("Guardrail: LLM answer had missing/invalid citations; using extractive answer instead.")
                ans = None
            else:
                mode = f"LIVE ({llm.provider}) + citation guardrail"
        except Exception as ex:
            st.warning(f"Live chat failed ({ex}); using extractive answer.")
    if ans is None:
        ans = demo_answer(q, rows)
    return ans, mode, sorted(set(re.findall(CITE_RX, ans)))


# --------------------------------------------------------------------------- #
# Epic 6 - PDF export
# --------------------------------------------------------------------------- #
def build_pdf(case, rows, results, registry, custody, chat, signed_at):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=18 * mm,
                            bottomMargin=16 * mm, title=f"Forensic Report {case['id']}", author="CrimeVision AI")
    ss = getSampleStyleSheet()
    navy = colors.HexColor("#0b3d5c")
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=navy, fontSize=20, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=navy, fontSize=13, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5)
    warn = ParagraphStyle("w", parent=body, textColor=colors.HexColor("#b71c1c"), fontName="Helvetica-Bold")

    def P(t, st_=body):
        return Paragraph(escape(str(t)), st_)

    def tbl(data, widths):
        rows_ = [[P(c, small) for c in r] for r in data]
        t = Table(rows_, colWidths=[w * mm for w in widths], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d6e4ee")),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fa")])]))
        return t

    def footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.grey)
        c.drawString(16 * mm, 9 * mm, f"CrimeVision AI | Case {case['id']} | CONFIDENTIAL - FORENSIC WORK PRODUCT")
        c.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {d.page}")
        c.restoreState()

    S = []
    S += [P("FORENSIC EVIDENCE SUMMARY REPORT", h1), P("CrimeVision AI - Intelligent Multimodal Forensic Analysis Platform", body)]
    if not signed_at:
        S.append(P("UNSIGNED DRAFT: investigator sign-off has not been recorded.", warn))
    S += [Spacer(1, 6), tbl([["Field", "Value"], ["Case ID", case["id"]], ["Investigator", case["inv"] or "-"],
                             ["Badge / ID", case["badge"] or "-"], ["Report generated", now_utc()],
                             ["Sign-off", signed_at or "NOT SIGNED"],
                             ["Ledger integrity (SHA-256)", sha256(json.dumps(rows, sort_keys=True, default=str).encode())]],
                            [45, 130])]
    S.append(P("1. Evidence Register & Integrity", h2))
    S.append(tbl([["Evidence ID", "File", "Type", "KB", "SHA-256"]] +
                 [[r["Evidence ID"], r["File"], r["Type"], r["Size (KB)"], r["SHA-256"]] for r in registry.values()],
                 [24, 36, 14, 12, 90]) if registry else P("No evidence registered."))
    S.append(P("2. Visual Evidence Findings", h2))
    for r in results["IMG"].values():
        a = r["assessment"]
        S += [P(f"{r['evid']} - {r['name']} | Threat: {a['threat']} | Scene: {a['scene']} | Method: {r['mode']}"),
              P(f"Rationale: {a['why']}"), P(f"Caption: {r['caption']}", small)]
        W, H = r["size"]
        wpt = 110 * mm
        S.append(RLImage(io.BytesIO(r["png"]), width=wpt, height=wpt * H / W))
        S.append(tbl([["Label", "Confidence", "Region", "BBox (norm.)"]] +
                     [[d["label"], f"{d['confidence']:.2f}", region(d["bbox"]), str([round(x, 2) for x in d["bbox"]])]
                      for d in r["dets"]], [45, 25, 40, 65]))
        S.append(P("Investigative leads: " + " | ".join(r["leads"]), small))
    if not results["IMG"]:
        S.append(P("No image evidence analysed."))
    S.append(P("3. Witness Audio Findings", h2))
    for r in results["AUD"].values():
        S += [P(f"{r['evid']} - {r['name']} | Method: {r['mode']}"),
              tbl([["Time", "Speaker", "Transcript"]] +
                  [[f"{sec_to_hms(s['start'])}", s["speaker"], s["text"]] for s in r["segments"]], [20, 28, 127])]
    if not results["AUD"]:
        S.append(P("No audio evidence analysed."))
    S.append(P("4. CCTV Video Findings", h2))
    for r in results["VID"].values():
        S += [P(f"{r['evid']} - {r['name']} | Method: {r['mode']}")]
        S.append(RLImage(io.BytesIO(r["png"]), width=110 * mm, height=110 * mm * 270 / 480))
        S.append(tbl([["Offset (s)", "Subject", "Event", "Indicative classification", "Conf."]] +
                     [[f"{e['offset']:.1f}", e["subject"], e["event"], e["legal"], f"{e['confidence']:.2f}"]
                      for e in r["events"]], [18, 22, 60, 60, 15]))
    if not results["VID"]:
        S.append(P("No video evidence analysed."))
    S.append(PageBreak())
    S.append(P("5. Unified Forensic Event Ledger", h2))
    S.append(tbl([["Entry ID", "Time", "Source", "Event", "Category", "Conf.", "Sim."]] +
                 [[e["entry_id"], e["time"], e["source"], e["event"], e["category"], e["confidence"],
                   "Y" if e["simulated"] else "N"] for e in rows], [34, 15, 14, 56, 36, 11, 9]) if rows
             else P("Ledger is empty."))
    S.append(P("6. Evidence Interrogation Log", h2))
    for c in chat:
        S += [P(f"Q: {c['q']}", body), P(f"A ({c['mode']}): {c['a']}", small), Spacer(1, 3)]
    if not chat:
        S.append(P("No evidence queries recorded."))
    S.append(P("7. Chain of Custody", h2))
    S.append(tbl([["Timestamp", "Evidence ID", "Action", "Handler"]] +
                 [[c["Timestamp"], c["Evidence ID"], c["Action"], c["Handler"]] for c in custody], [38, 28, 80, 29]))
    S.append(P("8. Limitations & Certification", h2))
    S.append(P("AI-assisted outputs (object detection, transcription, tracking, legal classification) are investigative "
               "aids, not conclusions. Items flagged SIMULATED were produced by demo fallbacks and MUST NOT be relied on. "
               "Penal categories are indicative (Indian BNS references) and require review by qualified counsel. "
               "Admissibility of electronic records depends on jurisdiction; attach the applicable certificate "
               "(e.g. under s.63 BSA 2023) separately.", small))
    S += [Spacer(1, 18), tbl([["Investigator signature", "Date / Time", "Supervising officer"],
                              [f"{case['inv'] or '________'} {'(signed ' + signed_at + ')' if signed_at else ''}",
                               signed_at or "________", "________"]], [75, 45, 55])]
    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
init_state()

with st.sidebar:
    st.markdown("### 🔐 Configuration")
    provider = st.radio("LLM provider", ["Groq", "Gemini"], horizontal=True)
    secret_name = "GROQ_API_KEY" if provider == "Groq" else "GEMINI_API_KEY"
    key = st.text_input(f"{provider} API key", type="password", value=get_secret(secret_name),
                        help="Stored only in this session. Leave empty for automatic DEMO mode.")
    llm = LLM(provider, key)
    st.markdown(f"**Mode:** <span class='cv-badge' style='background:{'#2e7d32' if llm.live else '#ef6c00'}'>"
                f"{'LIVE' if llm.live else 'DEMO FALLBACK'}</span>", unsafe_allow_html=True)
    st.markdown("### 🗂 Chain of Custody")
    case_id = st.text_input("Case ID", value="CV-2026-0001", key="case_id")
    investigator = st.text_input("Investigator name", key="investigator")
    badge = st.text_input("Badge / ID", key="badge")
    signed = st.checkbox("I certify custody of this evidence and sign off on this case file.")
    if signed and not st.session_state.signed_at and investigator:
        st.session_state.signed_at = now_utc()
        log_custody("INVESTIGATOR SIGN-OFF")
    if not signed:
        st.session_state.signed_at = None
    st.caption(f"Sign-off: {st.session_state.signed_at or 'pending (name + tick required)'}")
    if signed and not investigator:
        st.warning("Enter investigator name to record sign-off.")
    st.markdown("### ⚡ Quick actions")
    if st.button("Load full demo case", width="stretch"):
        process_image("demo_scene.png", demo_scene_image(), hms_to_sec("22:14:30"), LLM(provider, ""), preset=True)
        process_audio("demo_112_call.wav", b"demo-112-call", hms_to_sec("22:14:00"), LLM(provider, ""), use_sample=True)
        process_video("demo_cctv.mp4", b"demo-cctv", hms_to_sec("22:14:20"), LLM(provider, ""), 3, .3, False, demo=True)
        st.success("Demo case loaded - see the Fusion tab.")
    if st.button("Reset case", width="stretch"):
        for k in ["ledger", "registry", "custody", "results", "last", "chat", "signed_at", "pdf"]:
            st.session_state.pop(k, None)
        init_state()
        st.rerun()
    with st.expander("Custody log"):
        st.dataframe(pd.DataFrame(st.session_state.custody), width="stretch", hide_index=True)

st.markdown("""<div class='cv-hero'><h1>🕵️ CrimeVision AI</h1>
<p>Intelligent Multimodal Forensic Analysis Platform · Skill Wallet Capstone Project 2 · image + audio + video evidence fusion</p></div>""",
            unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Case", case_id)
c2.metric("Evidence items", len(st.session_state.registry))
c3.metric("Ledger events", len(flat_ledger()))
c4.metric("Sign-off", "Signed" if st.session_state.signed_at else "Pending")

t1, t2, t3, t4, t5 = st.tabs(["🖼 Scene Tagger", "🎙 Witness Audio", "📹 CCTV Analysis", "🧩 Fusion & Evidence Chat", "📄 PDF Report"])

# ---- Tab 1 ----------------------------------------------------------------
with t1:
    st.subheader("Crime Scene Visual Evidence Tagger")
    up = st.file_uploader("Upload crime scene photograph", type=["jpg", "jpeg", "png", "webp"], key="img_up")
    clk = st.text_input("Capture clock (HH:MM:SS)", "00:00:00", key="img_clk",
                        help="Real-world time the photo was taken; used to order evidence in the ledger.")
    b1, b2 = st.columns(2)
    if b1.button("🔍 Run detection", disabled=up is None, key="img_run"):
        try:
            process_image(up.name, up.getvalue(), hms_to_sec(clk), llm)
        except Exception as ex:
            st.error(f"Image analysis failed: {ex}")
    if b2.button("Use simulated demo scene", key="img_demo"):
        process_image("demo_scene.png", demo_scene_image(), hms_to_sec(clk), LLM(provider, ""), preset=True)
    r = st.session_state.results["IMG"].get(st.session_state.last.get("IMG"))
    if r:
        a = r["assessment"]
        col = THREAT_COLORS[a["threat"]]
        st.markdown(f"<span class='cv-mono'>{r['evid']}</span> · <span class='cv-sim'>{r['mode']}</span>", unsafe_allow_html=True)
        left, right = st.columns([3, 2])
        left.image(r["png"], width="stretch")
        right.markdown(f"**Threat level** <span class='cv-badge' style='background:{col}'>{a['threat']}</span><br>"
                       f"**Scene classification:** {a['scene']}", unsafe_allow_html=True)
        right.caption(a["why"])
        right.markdown(f"**Caption:** {r['caption']}")
        right.markdown("**Prioritised investigative leads**")
        for i, l in enumerate(r["leads"], 1):
            right.markdown(f"{i}. {l}")
        W, H = r["size"]
        st.dataframe(pd.DataFrame([{"Item": d["label"], "Confidence": d["confidence"], "Region": region(d["bbox"]),
                                    "BBox px (x1,y1,x2,y2)": str([int(d["bbox"][0] * W), int(d["bbox"][1] * H),
                                                                  int(d["bbox"][2] * W), int(d["bbox"][3] * H)])}
                                   for d in r["dets"]]), width="stretch", hide_index=True)

# ---- Tab 2 ----------------------------------------------------------------
with t2:
    st.subheader("Witness Audio Transcription & Timeline")
    upa = st.file_uploader("Upload witness statement / 911-112 call", type=["mp3", "wav", "m4a", "ogg", "flac"], key="aud_up")
    clk_a = st.text_input("Recording start clock (HH:MM:SS)", "00:00:00", key="aud_clk")
    a1, a2 = st.columns(2)
    if a1.button("🎧 Transcribe & extract timeline", disabled=upa is None, key="aud_run"):
        with st.spinner("Transcribing..."):
            try:
                process_audio(upa.name, upa.getvalue(), hms_to_sec(clk_a), llm)
            except Exception as ex:
                st.error(f"Audio analysis failed: {ex}")
    if a2.button("Use sample 112 call", key="aud_demo"):
        process_audio("demo_112_call.wav", b"demo-112-call", hms_to_sec(clk_a), LLM(provider, ""), use_sample=True)
    r = st.session_state.results["AUD"].get(st.session_state.last.get("AUD"))
    if r:
        st.markdown(f"<span class='cv-mono'>{r['evid']}</span> · <span class='cv-sim'>{r['mode']}</span>", unsafe_allow_html=True)
        st.markdown("**Speaker-tagged transcript**")
        st.dataframe(pd.DataFrame([{"Time": sec_to_hms(r["clock"] + s["start"]), "Offset (s)": round(s["start"], 1),
                                    "Speaker": s["speaker"], "Transcript": s["text"]} for s in r["segments"]]),
                     width="stretch", hide_index=True)
        st.markdown("**Chronological key events**")
        st.dataframe(pd.DataFrame([{"Time": sec_to_hms(r["clock"] + e["offset"]), "Event": e["event"],
                                    "Category": e["category"], "Confidence": round(e["confidence"], 2)}
                                   for e in r["events"]]), width="stretch", hide_index=True)

# ---- Tab 3 ----------------------------------------------------------------
with t3:
    st.subheader("CCTV Video Analysis & Event Progression")
    if not CV2_OK:
        st.error("OpenCV not installed: upload analysis disabled (simulated demo still works).")
    upv = st.file_uploader("Upload CCTV footage", type=["mp4", "avi", "mov", "mkv"], key="vid_up")
    v1, v2, v3 = st.columns(3)
    fps_s = v1.slider("Sampling rate (frames/s)", 1, 6, 3)
    min_a = v2.slider("Min motion area (% of frame)", 0.05, 2.0, 0.3)
    use_kf = v3.checkbox("Vision-LLM on keyframes", value=False, disabled=not llm.live)
    clk_v = st.text_input("Video start clock (HH:MM:SS)", "00:00:00", key="vid_clk")
    d1, d2 = st.columns(2)
    if d1.button("🎬 Analyse video", disabled=upv is None or not CV2_OK, key="vid_run"):
        with st.spinner("Sampling frames and tracking subjects..."):
            try:
                process_video(upv.name, upv.getvalue(), hms_to_sec(clk_v), llm, fps_s, min_a, use_kf)
            except Exception as ex:
                st.error(f"Video analysis failed: {ex}")
    if d2.button("Run simulated CCTV demo", key="vid_demo"):
        process_video("demo_cctv.mp4", b"demo-cctv", hms_to_sec(clk_v), LLM(provider, ""), fps_s, min_a, False, demo=True)
    r = st.session_state.results["VID"].get(st.session_state.last.get("VID"))
    if r:
        st.markdown(f"<span class='cv-mono'>{r['evid']}</span> · <span class='cv-sim'>{r['mode']}</span>", unsafe_allow_html=True)
        l, rr = st.columns([3, 2])
        l.image(r["png"], caption="Suspect / vehicle trajectories (ring = first seen, dot = last seen)", width="stretch")
        rr.markdown("**Tracked subjects**")
        rr.dataframe(pd.DataFrame(r["tracks"]), width="stretch", hide_index=True)
        st.markdown("**Chronological event log with indicative penal classification**")
        st.dataframe(pd.DataFrame([{"Time": sec_to_hms(r["clock"] + e["offset"]), "Offset (s)": round(e["offset"], 1),
                                    "Subject": e["subject"], "Event": e["event"],
                                    "Indicative classification": e["legal"], "Confidence": round(e["confidence"], 2)}
                                   for e in r["events"]]), width="stretch", hide_index=True)
        st.caption("Motion-based tracking is heuristic (background subtraction + centroid matching). Classifications are "
                   "indicative only and need legal review.")

# ---- Tab 4 ----------------------------------------------------------------
with t4:
    st.subheader("Multimodal Fusion Ledger & Grounded Evidence Chat")
    rows = flat_ledger()
    if not rows:
        st.info("Ledger is empty. Analyse evidence in tabs 1-3 or click **Load full demo case** in the sidebar.")
    else:
        st.dataframe(pd.DataFrame([{"Entry ID": e["entry_id"], "Time": e["time"], "Source": e["source"],
                                    "Event": e["event"], "Category": e["category"], "Conf.": e["confidence"],
                                    "Simulated": "yes" if e["simulated"] else "no"} for e in rows]),
                     width="stretch", hide_index=True)
        st.caption("Times combine the investigator-entered start clock of each source plus in-clip offsets.")
    st.markdown("#### Interrogate the evidence")
    st.caption("Answers use ONLY ledger entries and must cite entry IDs; uncited or unknown citations are rejected.")
    for c in st.session_state.chat:
        with st.chat_message("user"):
            st.write(c["q"])
        with st.chat_message("assistant"):
            st.write(c["a"])
            st.caption(f"{c['mode']} · citations: {', '.join(c['cites']) or 'none'}")
    q = st.chat_input("e.g. Was a weapon visible before the suspect fled?")
    if q:
        ans, mode, cites = answer_question(q, llm)
        st.session_state.chat.append({"q": q, "a": ans, "mode": mode, "cites": cites})
        log_custody(f"EVIDENCE QUERY: {q[:80]}")
        st.rerun()

# ---- Tab 5 ----------------------------------------------------------------
with t5:
    st.subheader("Automated Forensic PDF Export")
    if not RL_OK:
        st.error("ReportLab is not installed (pip install reportlab).")
    else:
        st.write("Generates a summary with register + SHA-256 hashes, findings, unified ledger, query log, custody trail "
                 "and signature block.")
        if not st.session_state.signed_at:
            st.warning("Sign-off pending: the report will be stamped UNSIGNED DRAFT.")
        if st.button("📄 Generate forensic report", type="primary"):
            try:
                log_custody("PDF REPORT GENERATED")
                case = {"id": case_id, "inv": investigator, "badge": badge}
                st.session_state.pdf = build_pdf(case, flat_ledger(), st.session_state.results, st.session_state.registry,
                                                 st.session_state.custody, st.session_state.chat, st.session_state.signed_at)
            except Exception as ex:
                st.error(f"PDF generation failed: {ex}")
        if st.session_state.pdf:
            st.download_button("⬇ Download PDF", st.session_state.pdf, file_name=f"CrimeVision_{case_id}.pdf",
                               mime="application/pdf")
            st.success(f"Report ready ({len(st.session_state.pdf) / 1024:.1f} KB).")
