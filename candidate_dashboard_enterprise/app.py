import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from pathlib import Path
import re

st.set_page_config(page_title="Candidate Profiles – Enterprise", page_icon="📇", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / 'assets'
DATA_DIR = BASE_DIR / 'data'

# ---------------- Utilities ----------------

def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).strip().lower())


def find_col(df, *candidates):
    cols_norm = {normalize(c): c for c in df.columns}
    for cand in candidates:
        key = normalize(cand)
        if key in cols_norm:
            return cols_norm[key]
        for k, orig in cols_norm.items():
            if key and key in k:
                return orig
    return None


def parse_date(val):
    if val is None:
        return None
    s = str(val).strip()
    if s == '' or s.lower() in {'na','n/a','nil','none','-','—','\\-'}:
        return None
    s = re.sub(r'([A-Za-z]{3})-(\d{2})$', r'\1 20\2', s)
    return pd.to_datetime(s, dayfirst=True, errors='coerce')


def get_num(x):
    if pd.isna(x):
        return None
    s = str(x).replace(',', '').strip()
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except:
        return None


def inject_css():
    css_path = ASSETS_DIR / 'styles.css'
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8') if css_path.exists() else ''}</style>", unsafe_allow_html=True)

# --- THEME STATE ---
DEFAULT_THEME = {
    'mode': 'Light',
    'primary': '#2563EB',  # blue-600
    'accent': '#0EA5E9',   # sky-500
    'text': '#0F172A',     # slate-900
    'muted': '#475569',    # slate-600
    'bg': '#F8FAFC',       # slate-50
    'card': '#FFFFFF',
    'border': '#E5E7EB',
    'font': 'Inter',
    'base_scale': 1.0,
    'density': 'Comfortable'
}

for k,v in DEFAULT_THEME.items():
    st.session_state.setdefault(k, v)


def apply_theme():
    t = st.session_state
    # Dark mode overrides
    if t['mode'] == 'Dark':
        bg = '#0B1220'; text = '#E6E8F2'; muted = '#A3ADC2'; card = '#0F172A'; border = '#1F2937'
    else:
        bg = t['bg']; text = t['text']; muted = t['muted']; card = t['card']; border = t['border']

    css = f"""
<style>
:root {{
  --primary: {{t['primary']}};
  --accent: {{t['accent']}};
  --text: {{text}};
  --muted: {{muted}};
  --bg: {{bg}};
  --card: {{card}};
  --border: {{border}};
  --font: {{t['font']}}, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, sans-serif;
  --base-scale: {{t['base_scale']}};
  --radius: 12px;
  --shadow: 0 8px 24px rgba(0,0,0,.08);
}}
html, body, [class^="css"] {{
  font-family: var(--font);
  color: var(--text);
  font-size: calc(16px * var(--base-scale));
}}
.stat {{
  text-align: center;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: .6rem .8rem;
}}
.stat-label {{
  color: var(--muted);
  font-weight: 700;
  font-size: .85rem;
}}
.stat-value {{
  color: var(--primary);
  font-size: 1.6rem;
  font-weight: 900;
}}
</style>
"""

    st.markdown(css, unsafe_allow_html=True)


def load_data(uploaded_file):
    opts = dict(encoding='utf-8-sig', engine='python')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, **opts, on_bad_lines='skip')
    else:
        sample = DATA_DIR / 'Dashboard Format.csv'
        if sample.exists():
            df = pd.read_csv(sample, **opts, on_bad_lines='skip')
        else:
            st.warning("No data found. Upload a CSV from the sidebar.")
            return pd.DataFrame()
    df.dropna(how='all', inplace=True)
    df.columns = [c.strip() for c in df.columns]
    return df

# ---------------- Sidebar ----------------
inject_css()
apply_theme()  # apply defaults immediately
with st.sidebar:
    st.subheader("Data")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    df = load_data(uploaded)
    if df.empty:
        st.stop()

    name_col = find_col(df, 'Name') or df.columns[1]
    names = sorted([str(x).strip() for x in df[name_col].dropna().unique() if str(x).strip()!=''])
    search = st.text_input("Search name")
    options = [n for n in names if search.lower() in n.lower()] if search else names
    sel = st.selectbox("Select candidate", options)

    st.markdown("---")
    st.subheader("Appearance")
    st.session_state['mode'] = st.radio("Mode", ['Light','Dark'], index=0 if st.session_state['mode']=='Light' else 1, horizontal=True)
    st.session_state['primary'] = st.color_picker("Primary", value=st.session_state['primary'])
    st.session_state['accent'] = st.color_picker("Accent", value=st.session_state['accent'])
    st.session_state['font'] = st.selectbox("Font", ['Inter','Segoe UI','Roboto','Nunito','Lato','Montserrat','Poppins','System UI'], index=['Inter','Segoe UI','Roboto','Nunito','Lato','Montserrat','Poppins','System UI'].index(st.session_state['font']))
    st.session_state['base_scale'] = st.slider("Base font size", 0.9, 1.3, st.session_state['base_scale'], 0.05)
    st.session_state['density'] = st.selectbox("Density", ['Comfortable','Compact'], index=0 if st.session_state['density']=='Comfortable' else 1)

apply_theme()  # re-apply after user changes

# ---------------- Helpers ----------------

def F(df, row, *keys, default='—'):
    col = find_col(df, *keys)
    if not col:
        return default
    val = row.get(col, default)
    if pd.isna(val) or str(val).strip()=='' or str(val).strip().lower() in {'na','n/a','nil','none'}:
        return default
    return val

row = df[df[name_col]==sel].iloc[0]

# ---------------- Header band ----------------
applied_pos = F(df, row, 'Applied Position')
st.markdown(f"""
<div class='header-band'>
  <div class='title'>📇 {sel}</div>
  <div class='subtitle'>{applied_pos}</div>
  <div class='pills'>
    <span class='pill'>🎂 {F(df,row,'Date of Birth')}</span>
    <span class='pill'>🧮 Age: {F(df,row,'Your age')}</span>
    <span class='pill'>💍 {F(df,row,'Marital Status')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------- KPI row ----------------
colK = st.columns(5)
metrics = [
    (F(df,row,'Total No. of Publications'), 'Total Publications'),
    (F(df,row,'Total No. of Publications With Impact Factor'), 'With Impact Factor'),
    (F(df,row,'No. of W Category Publications'), 'W Category'),
    (F(df,row,'No. of X Category Publications'), 'X Category'),
    (F(df,row,'No. of Y Category Publications'), 'Y Category'),
]
for c,(v,label) in zip(colK, metrics):
    with c:
        st.markdown(f"<div class='kpi'><div class='value'>{v}</div><div class='label'>{label}</div></div>", unsafe_allow_html=True)

# ---------------- Layout (two-column professional) ----------------
left, right = st.columns([1.1, 0.9])

with left:
    # Personal Details
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Personal Details</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        for label, key in [
            ("Date of Birth", 'Date of Birth'),
            ("Age", 'Your age'),
            ("Marital Status", 'Marital Status'),
            ("Previously at Riphah?", 'Previously employed at Riphah?'),
        ]:
            val = F(df,row,key)
            if val not in ('—','',None):
                st.markdown(f"<div class='qa'><div class='qa-label'>{label}</div><div class='qa-sep'></div><div class='qa-value'>{val}</div></div>", unsafe_allow_html=True)
    with c2:
        cities = ', '.join(
            [str(F(df,row,'Experience City 1','')),
             str(F(df,row,'Experience City 2','')),
             str(F(df,row,'Experience City 3','')),
             str(F(df,row,'Experience City 4',''))]
        ).replace(', ,', ', ').strip(', ').strip()
        for label, val in [
            ("Current Salary", F(df,row,'What is Your Current Salary')),
            ("Expected Salary", F(df,row,'What is Your Expected Salary')),
            ("Experience Cities", cities if cities else '—'),
            ("Health (if any)", F(df,row,'Any Prolong illness / disease')),
        ]:
            if val not in ('—','',None):
                st.markdown(f"<div class='qa'><div class='qa-label'>{label}</div><div class='qa-sep'></div><div class='qa-value'>{val}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Experience
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Experience</div>", unsafe_allow_html=True)
    roles = []
    def add_role(title_key, org_key, start_key, end_key):
        title = F(df,row,title_key)
        org = F(df,row,org_key)
        s = parse_date(F(df,row,start_key))
        e = parse_date(F(df,row,end_key))
        if str(title)!='—' or str(org)!='—':
            end_eff = e if e else datetime.today()
            months = None
            if s is not None:
                months = (end_eff.year - s.year)*12 + (end_eff.month - s.month)
            roles.append({'Role': title, 'Organization': org, 'Start': s, 'End': end_eff, 'Months': months, 'EndRaw': F(df,row,end_key)})
    add_role('Most Recent Designation/Position (Employment)','Organization/Institute/Company/Employer','Current Employment - Start Date','Most recent Employment - End Date')
    add_role('Second Most Recent Designation/Position','Second most recent Organization/Institute/Company/Employer','Second most recent Employment - Start Date','Second most recent Employment - End Date')
    add_role('Third Most Recent Designation/Position','Third most recent Organization/Institute/Company/Employer','Third most recent Employment - Start Date','Third most recent Employment - End Date')
    add_role('Fourth Most Recent Designation/Position','Fourth most recent Organization/Institute/Company/Employer','Fourth most recent Employment - Start Date','Fourth most recent Employment - End Date')

    for r in roles:
        s_txt = r['Start'].date().isoformat() if isinstance(r['Start'], pd.Timestamp) else '—'
        e_txt = r['End'].date().isoformat() if r['EndRaw']!='—' and r['End'] else 'Present'
        tenure = f"{r['Months']//12}y {r['Months']%12}m" if r['Months'] is not None else '—'
        st.markdown(f"""
        <div class='role'>
          <div class='role-top'>
            <div class='role-left'>
              <div class='role-title'>{r['Role']}</div>
              <div class='role-meta'>
                <span class='chip'>📅 {s_txt} → {e_txt}</span>
                <span class='chip'>⏱️ {tenure}</span>
              </div>
            </div>
            <div class='role-org' title='{r['Organization']}'>🏢 {r['Organization']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    # Summary
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='qa'><div class='qa-label'>Total Experience</div><div class='qa-sep'></div><div class='qa-value'>{F(df,row,'Total Experience (In Years & months)')}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='qa'><div class='qa-label'>Student Feedback (last year %)</div><div class='qa-sep'></div><div class='qa-value'>{F(df,row,'Annual Student Feedback - Last year (in %)')}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    # Education
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Education</div>", unsafe_allow_html=True)
    def degree(level, title, result, start, end, uni, wr, ar):
        if (level in ('—','',None)) and (title in ('—','',None)):
            return
        bits = [
            ("Result", result), ("Start", start), ("Graduation", end), ("University", uni)
        ]
        if wr not in ('—','',None):
            bits.append(("World Rank", wr))
        if ar not in ('—','',None):
            bits.append(("Asia Rank", ar))
        st.markdown(f"**{level}** — {title}")
        for lab,val in bits:
            if val not in ('—','',None):
                st.markdown(f"<div class='qa'><div class='qa-label'>{lab}</div><div class='qa-sep'></div><div class='qa-value'>{val}</div></div>", unsafe_allow_html=True)
        st.markdown("<hr style='border:0;border-top:1px solid var(--border);margin:.5rem 0' />", unsafe_allow_html=True)
    degree(
        F(df,row,'Latest Qualification/Degree (Completed)'),
        F(df,row,'Major/Specialization With Complete Degree Title'),
        f"{F(df,row,'Obtained / Awarded GPA/Grade/Percentage/Marks')} / {F(df,row,'Total GPA/Grade/Percentage/Marks')}",
        F(df,row,'Last Degree - Start Date'), F(df,row,'Most Recent Degree - Graduation Date'),
        F(df,row,'Degree Awarding University'),
        F(df,row,"Latest Degree Awarding University's World Ranking"),
        F(df,row,"Latest Degree Awarding University's Asia Ranking (In case of Asian Universities)")
    )
    degree(
        F(df,row,'Second Last Qualification/Degree (Completed)'),
        F(df,row,'Major/Specialization With Complete Degree Title2'),
        f"{F(df,row,'Obtained / Awarded GPA/Grade/Percentage/Marks3')} / {F(df,row,'Total GPA/Grade/Percentage/Marks2')}",
        F(df,row,'Second Last Degree - Start Date'), F(df,row,'Second Last Degree - Graduation Date'),
        F(df,row,'Second Last Degree Awarding University'), None, None
    )
    inproc = F(df,row,'Qualification/Degree In Process (If Any)')
    if str(inproc) not in {'—','', 'NA','N/A','Nil','nil'}:
        degree(
            inproc + ' (In Process)', F(df,row,'Major/Specialization of Degree (In Process)'),
            F(df,row,'GPA/Grade/Percentage'), F(df,row,'In Process Degree Start Date'),
            F(df,row,'End Date (Expected)'), '—', None, None
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Publications & Citations
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Publications & Citations</div>", unsafe_allow_html=True)
    pub_total = get_num(F(df,row,'Total No. of Publications')) or 0
    pub_if = get_num(F(df,row,'Total No. of Publications With Impact Factor')) or 0
    pub_nonif = max(pub_total - pub_if, 0)
    w = get_num(F(df,row,'No. of W Category Publications')) or 0
    x = get_num(F(df,row,'No. of X Category Publications')) or 0
    y = get_num(F(df,row,'No. of Y Category Publications')) or 0

    cat_df = pd.DataFrame({'Type':['Impact Factor','Non-Impact','W','X','Y'], 'Count':[pub_if, pub_nonif, w, x, y]})
    fig = px.bar(cat_df, x='Type', y='Count', text='Count', color='Type', color_discrete_sequence=['#2563EB','#94A3B8','#F59E0B','#EF4444','#10B981'])
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.update_layout(margin=dict(l=4,r=4,t=2,b=2), height=250, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='stat'><div class='stat-label'>Publications (Total)</div><div class='stat-value'>{int(pub_total)}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat'><div class='stat-label'>With Impact Factor</div><div class='stat-value'>{int(pub_if)}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='stat'><div class='stat-label'>Conference Papers</div><div class='stat-value'>{F(df,row,'Total No. of Conference Papers')}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Achievements
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Achievements</div>", unsafe_allow_html=True)
    ach = str(F(df,row,'ACADEMIC/PROFESSIONAL ACHIEVEMENTS/INITIATIVES (If Any)'))
    items = [i.strip(' \t\n-•') for i in re.split(r'[\n\r;]+', ach) if i and i.strip() and i.strip().lower() not in {'—','na','n/a','nil'}]
    if items:
        st.markdown("\n".join([f"- {i}" for i in items]))
    else:
        st.caption("No achievements listed.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Trainings & Certifications
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Trainings & Certifications</div>", unsafe_allow_html=True)
    tr = str(F(df,row,'Training/Workshop/Diploma/Certificate (Related to Post Applied If Any)'))
    titems = [i.strip(' \t\n-•') for i in re.split(r'[\n\r;]+', tr) if i and i.strip() and i.strip().lower() not in {'—','na','n/a','nil'}]
    if titems:
        st.markdown("\n".join([f"- {i}" for i in titems]))
    else:
        st.caption("No trainings/certifications listed.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Application / Vacancy Details
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Application / Vacancy Details</div>", unsafe_allow_html=True)
    fac_raw = F(df,row,'Strategic Unit the position belongs to')
    fac_map = {'FSSH':'Faculty of Social Sciences & Humanities','RIMS':'Riphah Institute of Media Sciences','RIPP':'Riphah Institute of Public Policy','DSS':'Department of Social Sciences','DSS Islamic Studies':'Department of Islamic Studies'}
    faculty = fac_map.get(str(fac_raw).strip(), fac_raw)
    for label, val in [
        ("Faculty", faculty),
        ("Strategic Unit", fac_raw),
        ("Department Applied For", F(df,row,'Department (Position Department)')),
    ]:
        if val not in ('—','',None):
            st.markdown(f"<div class='qa'><div class='qa-label'>{label}</div><div class='qa-sep'></div><div class='qa-value'>{val}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Analysis & Export ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Candidate Insight & Analysis</div>", unsafe_allow_html=True)

total_pubs = get_num(F(df,row,'Total No. of Publications')) or 0
if_pubs = get_num(F(df,row,'Total No. of Publications With Impact Factor')) or 0
conf = get_num(F(df,row,'Total No. of Conference Papers')) or 0
w = get_num(F(df,row,'No. of W Category Publications')) or 0
x = get_num(F(df,row,'No. of X Category Publications')) or 0
y = get_num(F(df,row,'No. of Y Category Publications')) or 0
non_if = max(total_pubs - if_pubs, 0)
if_share = (if_pubs/(if_pubs+non_if)) if (if_pubs+non_if)>0 else 0
exp_str = str(F(df,row,'Total Experience (In Years & months)'))

score = 0
score += min(total_pubs/20, 2.0)
score += min(if_share*2.0, 1.5)
score += min((w+x)/10, 1.0)
score += min(conf/20, 0.5)
score = round(score, 2)

st.markdown("**Highlights**")
st.markdown(f"- Publications: **{int(total_pubs)}** total; IF: **{int(if_pubs)}** ({if_share:.0%})")
st.markdown(f"- W/X/Y: **{int(w)} / {int(x)} / {int(y)}**")
st.markdown(f"- Experience: **{exp_str}**")
st.markdown(f"- Composite research score: **{score}/5** (heuristic)")

strengths = []
if if_share >= 0.25: strengths.append("Healthy proportion of IF publications")
if (w+x) >= 5: strengths.append("Strong record in W/X category journals")
if conf >= 8: strengths.append("Active conference presence (≥ 8)")
if strengths:
    st.markdown("**Strengths**\n"+"\n".join([f"- ✅ {s}" for s in strengths]))

gaps = []
if total_pubs < 10: gaps.append("Low overall publication count — target ≥ 10")
if if_pubs == 0 and total_pubs > 0: gaps.append("Convert some outputs to Impact Factor journals")
if (w+x+y) == 0 and total_pubs > 0: gaps.append("No W/X/Y categories — aim for recognized venues")
if gaps:
    st.markdown("**Development Areas**\n"+"\n".join([f"- ⚠️ {g}" for g in gaps]))

# Export options
try:
    import fitz  # PyMuPDF
    doc = fitz.open(); page = doc.new_page(); y = 40
    def write(text, size=14, bold=False):
        global y_ref
        global y
        font = "helvB" if bold else "helv"
        page.insert_text((40, y), text, fontsize=size, fontname=font)
        y += size + 8
    write(f"Candidate: {sel}", 20, True)
    write(f"Position: {applied_pos}")
    write("")
    write("Key Numbers", 16, True)
    write(f"Total Publications: {int(total_pubs)}")
    write(f"IF Publications: {int(if_pubs)} (share {if_share:.0%})")
    write(f"Conference Papers: {int(conf)}")
    write(f"W/X/Y: {int(w)}/{int(x)}/{int(y)}")
    write("")
    write("Experience", 16, True); write(f"{exp_str}")
    write("")
    write("Heuristic Research Score", 16, True); write(f"{score}/5")
    if strengths:
        write(""); write("Strengths", 16, True)
        for s in strengths: write(f"• {s}")
    if gaps:
        write(""); write("Development Areas", 16, True)
        for g in gaps: write(f"• {g}")
    pdf_bytes = doc.tobytes(); doc.close()
    st.download_button("📄 Export profile PDF", data=pdf_bytes, file_name=f"{sel.replace(' ','_')}_profile.pdf", mime='application/pdf')
except Exception:
    # Quietly show a CSV export of the profile as a fallback; no noisy message
    prof = {
        'Name': sel, 'Applied Position': applied_pos,
        'Total Publications': total_pubs, 'IF Publications': if_pubs,
        'Conference Papers': conf, 'W': w, 'X': x, 'Y': y,
        'Experience': exp_str
    }
    csv_data = pd.DataFrame([prof]).to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Export summary CSV", data=csv_data, file_name=f"{sel.replace(' ','_')}_summary.csv", mime='text/csv')

st.markdown("</div>", unsafe_allow_html=True)

st.caption("Enterprise UI – professional, high-contrast, robust layout. Settings in sidebar apply instantly.")
