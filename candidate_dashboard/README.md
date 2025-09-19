# Candidate Profile Dashboard (Streamlit)

An iOS-inspired, gradient-rich, single-page dashboard to browse candidates from a CSV. Pick a name from the dropdown to see a beautifully grouped profile (Personal, Education, Experience, Publications, Achievements). Includes file upload to load your own data.

## ✨ Features
- Dropdown to select a candidate
- Upload your own CSV (same header format)
- iPadOS-like gradients, neumorphic cards, and pill chips
- Experience timeline, publication breakdowns, and key metrics
- Fuzzy column matching to handle slight header differences

## 🚀 Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Data
A sample CSV is expected at `data/Dashboard Format.csv`. Replace with your data or upload via the sidebar.

## 🔧 Notes
- Date parsing is tolerant of multiple date formats.
- If a field is missing in the CSV, the app hides that widget gracefully.
- Tested with Streamlit 1.36+ and Python 3.10+.
