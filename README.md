# CrimeVision AI - Intelligent Multimodal Forensic Analysis Platform
Skill Wallet Capstone Project 2

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Optional: set `GROQ_API_KEY` or `GEMINI_API_KEY` (env var or `.streamlit/secrets.toml`), or paste the key in the sidebar.
No key = automatic DEMO mode (clearly labelled SIMULATED everywhere, including the PDF).

## Deploy (Streamlit Community Cloud)
Push `app.py`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml` to GitHub, create the app, add the key under Secrets.

## Epic map
| Epic | Where |
|---|---|
| 1 Init, security, custody | sidebar, `register()`, `log_custody()` |
| 2 Object detection + caption | Tab 1, `process_image()` |
| 3 Whisper + timeline | Tab 2, `process_audio()` (Groq whisper-large-v3 / Gemini) |
| 4 Video tracking + classification | Tab 3, `analyze_video()` (OpenCV MOG2 + centroid tracker) |
| 5 Fusion + grounded chat | Tab 4, `flat_ledger()`, `answer_question()` |
| 6 Dashboard + PDF | header metrics, Tab 5, `build_pdf()` |

## Testing
Click "Load full demo case" in the sidebar, then ask: "Was a weapon visible before the suspect fled?"
