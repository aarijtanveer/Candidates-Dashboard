# Candidate Profile Dashboard (Enterprise UI)

A professional, high-contrast dashboard with an **enterprise look** (no iOS styling), working **Appearance** controls, and a clean two-column layout to reduce scrolling while preserving readability.

## Highlights
- **Appearance controls** in the sidebar (light/dark mode, primary & accent colors, font family, base font size, density) that **apply instantly**.
- **Header band** keeps candidate name & subtitle *inside* the header (no overflow), with high contrast and pills for quick facts.
- **KPI row** with crisp labels and large values.
- **Two-column layout**: left = Personal + Experience; right = Education + Publications + Achievements + Training + Application details.
- **In‑depth analysis** block with a composite research score and **PDF export** (falls back to CSV if PyMuPDF is unavailable, with *no noisy warning*).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
