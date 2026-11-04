# 📖 Qirat Tajweed Checker

Upload or record your Quran recitation. AI (Google Gemini Flash) listens, gives a
**Tajweed score**, lists mistakes with timestamps, and explains how to correct your pronunciation.

> ⚠️ AI feedback is a practice aid and can be wrong. It does **not** replace a qualified Tajweed teacher.

## Features
- Upload audio (mp3, wav, ogg, flac, aac, aiff, m4a) or record in the browser
- Overall score + 7 category scores (Makharij, Sifat, Madd, Ghunnah/Noon/Meem, Qalqalah, Waqf, Fluency)
- Mistakes with timestamp, Tajweed rule, and how to fix
- Optional verse text for more accurate checking
- Choose riwayah and strictness; download the report as JSON

## Run locally
Requires Python 3.10+.

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"      # Windows PowerShell: $env:GEMINI_API_KEY="your_key_here"
streamlit run app.py
```
Get a free API key at https://aistudio.google.com/apikey (or paste it in the app sidebar).

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (never commit your API key).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
5. Click **Deploy**.

## Notes
- Max upload: 15 MB. Use clear audio with little background noise.
- Change the model with the `GEMINI_MODEL` environment variable or the sidebar (default: `gemini-3.8-flash`; the app also auto-switches to another available Flash model if the default is ever retired).

## Troubleshooting
- **503 UNAVAILABLE / high demand:** Google's servers are temporarily busy. The app retries automatically and switches to another Flash model; if it still fails, wait a minute and press Analyze again.
- **429 / quota:** you hit the free-tier rate limit; wait a bit or use a key with more quota.
- **404 model not found:** type a current Gemini Flash model name in the sidebar.

## Files
- `app.py` – Streamlit app and Gemini logic
- `requirements.txt` – dependencies
- `README.md` – this file
