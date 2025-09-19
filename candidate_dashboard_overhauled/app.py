import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from pathlib import Path
import re

st.set_page_config(page_title="Candidate Profiles – Overhauled", page_icon="✨", layout="wide")

# ---------------- Path setup (robust) ----------------
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
    if pd.isna(val) or str(val).strip()=='' or str(val).strip().lower() in {'na','n/a','nil','none','-','—','\\-'}:
        return None
    try:
        return pd.to_datetime(val, dayfirst=True, errors='coerce')
    except Exception:
        return None


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


def set_theme(gradient: str, density: str, header_scale: float, accent: str):
    """Inject CSS variables for runtime theming"""
    gradients = {
        'Sunset': ('#FF7E5F', '#FEB47B'),
        'Peach':  ('#f6d365', '#fda085'),
        'Mango':  ('#f7971e', '#ffd200'),
        'Rose':   ('#feada6', '#f5efef'),
        'Coral':  ('#ff9966', '#ff5e62')
    }
    start, end = gradients.get(gradient, gradients['Sunset'])
    density_map = {'Cozy':'18px','Compact':'12px','Roomy':'22px'}
    radius = density_map.get(density, '18px')
    header_font = max(1.0, min(header_scale, 2.5))

    st.markdown(f"""
    <style>
      :root{
        --header-grad-start: {start};
        --header-grad-end: {end};
        --accent: {accent};
        --radius: {radius};
        --header-scale: {header_font};
      }
    </style>
    """, unsafe_allow_html=True)


def qa(label: str, value: str):
    label = str(label).strip()
    val = '—' if value is None or str(value).strip()=='' else str(value)
    st.markdown(
        f"""
        <div class='qa'>
          <div class='qa-label'>{label}</div>
          <div class='qa-sep'></div>
          <div class='qa-value'>{val}</div>
        </div>
        """, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
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


# ---------------- Sidebar Controls ----------------
inject_css()
with st.sidebar:
    st.markdown("## 📦 Data")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    df = load_data(uploaded)
    if df.empty:
        st.stop()

    name_col = find_col(df, 'Name') or df.columns[1]
    names = sorted([str(x).strip() for x in df[name_col].dropna().unique() if str(x).strip()!=''])
    st.markdown("---")
    st.markdown("## 🔎 Find a Candidate")
    search = st.text_input("Search name")
    options = [n for n in names if search.lower() in n.lower()] if search else names
    sel = st.selectbox("Select candidate", options)

    st.markdown("---")
    st.markdown("## 🎨 UI Settings")
    grad = st.selectbox("Header gradient", ['Sunset','Peach','Mango','Rose','Coral'], index=0)
    density = st.radio("Card density", ['Cozy','Compact','Roomy'], index=0, horizontal=True)
    header_scale = st.slider("Header size", 1.2, 2.4, 2.0, 0.1)
    accent = st.color_picker("Accent color", value="#FF8A65")

    # Visualization toggles
    st.markdown("## 📊 Visuals")
    show_timeline = st.toggle("Show experience timeline", value=True)
    show_if_pie = st.toggle("Show Impact vs Non-Impact pie", value=True)

set_theme(grad, density, header_scale, accent)

# ---------------- Selected row & field helper ----------------
row = df[df[name_col]==sel].iloc[0]

def F(*keys, default='—'):
    col = find_col(df, *keys)
    if not col:
        return default
    val = row.get(col, default)
    if pd.isna(val) or str(val).strip()=='' or str(val).strip().lower() in {'na','n/a','nil','none'}:
        return default
    return val

# --------------- Header ---------------
applied_pos = F('Applied Position')
st.markdown(f"""
<div class='header warm'>
  <div class='title' style='font-size: calc(1.4rem * var(--header-scale));'>👤 {sel}</div>
  <div class='subtitle'>{applied_pos}</div>
  <div class='header-pills'>
    <span class='pill'>🎂 {F('Date of Birth')}</span>
    <span class='pill'>🧮 Age: {F('Your age')}</span>
    <span class='pill'>💍 {F('Marital Status')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# --------------- KPI Row (updated) ---------------
# Replace with: Total No. of Publications, Total No. of Publications With Impact Factor,
# No. of W Category Publications, No. of X Category Publications, No. of Y Category Publications
k1 = F('Total No. of Publications')
k2 = F('Total No. of Publications With Impact Factor')
k3 = F('No. of W Category Publications')
k4 = F('No. of X Category Publications')
k5 = F('No. of Y Category Publications')

c1, c2, c3, c4, c5 = st.columns(5)
for c, v, label in [
    (c1, k1, 'Total No. of Publications'),
    (c2, k2, 'Publications With Impact Factor'),
    (c3, k3, 'W Category Publications'),
    (c4, k4, 'X Category Publications'),
    (c5, k5, 'Y Category Publications'),
]:
    with c:
        st.markdown(f"<div class='kpi'><div class='value'>{v}</div><div class='label'>{label}</div></div>", unsafe_allow_html=True)

st.write("")

# --------------- Personal Details (overhauled) ---------------
with st.container():
    st.markdown("<div class='card section-personal'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>👤 Personal Details</div>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        qa("Date of Birth", F('Date of Birth'))
        qa("Age", F('Your age'))
        qa("Marital Status", F('Marital Status'))
        qa("Previously employed at Riphah?", F('Previously employed at Riphah?'))
    with colB:
        qa("Current Salary", F('What is Your Current Salary'))
        qa("Expected Salary", F('What is Your Expected Salary'))
        # Cities
        cities = ', '.join([
            str(F('Experience City 1','')),
            str(F('Experience City 2','')),
            str(F('Experience City 3','')),
            str(F('Experience City 4',''))
        ]).replace(', ,', ', ').strip(', ').strip()
        qa("Experience Cities", cities if cities else '—')
        qa("Relatives at Riphah", F('Any Relative(s) working in Riphah?'))
        qa("Health (if any)", F('Any Prolong illness / disease'))
    st.markdown("</div>", unsafe_allow_html=True)

# --------------- Education (overhauled) ---------------
with st.container():
    st.markdown("<div class='card section-education'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎓 Education</div>", unsafe_allow_html=True)

    def degree_card(level, title, result, start, end, uni, wrank, arank):
        st.markdown(
            f"""
            <div class='degree'>
                <div class='degree-head'>
                    <div class='deg-level'>{level}</div>
                    <div class='deg-major'>{title}</div>
                </div>
                <div class='degree-grid'>
                    <div><span class='deg-label'>Result</span><span class='deg-val'>{result}</span></div>
                    <div><span class='deg-label'>Start</span><span class='deg-val'>{start}</span></div>
                    <div><span class='deg-label'>Graduation</span><span class='deg-val'>{end}</span></div>
                    <div class='wide'><span class='deg-label'>University</span><span class='deg-val'>{uni}</span></div>
                    <div><span class='deg-label'>World Rank</span><span class='deg-val'>{wrank}</span></div>
                    <div><span class='deg-label'>Asia Rank</span><span class='deg-val'>{arank}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    degree_card(
        F('Latest Qualification/Degree (Completed)'),
        F('Major/Specialization With Complete Degree Title'),
        f"{F('Obtained / Awarded GPA/Grade/Percentage/Marks')} / {F('Total GPA/Grade/Percentage/Marks')}",
        F('Last Degree - Start Date'),
        F('Most Recent Degree - Graduation Date'),
        F('Degree Awarding University'),
        F("Latest Degree Awarding University's World Ranking"),
        F("Latest Degree Awarding University's Asia Ranking (In case of Asian Universities)")
    )

    degree_card(
        F('Second Last Qualification/Degree (Completed)'),
        F('Major/Specialization With Complete Degree Title2'),
        f"{F('Obtained / Awarded GPA/Grade/Percentage/Marks3')} / {F('Total GPA/Grade/Percentage/Marks2')}",
        F('Second Last Degree - Start Date'),
        F('Second Last Degree - Graduation Date'),
        F('Second Last Degree Awarding University'),
        '—','—'
    )

    # In-process
    inproc = F('Qualification/Degree In Process (If Any)')
    if str(inproc) not in {'—','', 'NA','N/A','Nil','nil'}:
        degree_card(
            inproc + ' (In Process)',
            F('Major/Specialization of Degree (In Process)'),
            F('GPA/Grade/Percentage'),
            F('In Process Degree Start Date'),
            F('End Date (Expected)'),
            '—','—','—'
        )

    st.markdown(
        f"<div class='edu-durations'><span class='pill'>🧪 PhD: {F('Duration of Course PhD')}</span> "
        f"<span class='pill'>🧪 M.Phil: {F('Duration of M.Phil Course')}</span> "
        f"<span class='pill'>🧪 Masters: {F('Duration of Masters Course')}</span></div>",
        unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------- Experience (overhauled) ---------------
with st.container():
    st.markdown("<div class='card section-experience warm'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>💼 Experience</div>", unsafe_allow_html=True)

    roles = []
    def add_role(title_key, org_key, start_key, end_key):
        title = F(title_key)
        org = F(org_key)
        s = parse_date(F(start_key))
        e_raw = F(end_key)
        e = parse_date(e_raw)
        if str(title)!='—' or str(org)!='—':
            end_eff = e if e else datetime.today()
            months = None
            if s is not None:
                months = (end_eff.year - s.year)*12 + (end_eff.month - s.month)
            roles.append({
                'Role': title, 'Organization': org,
                'Start': s, 'End': end_eff, 'RawEnd': e_raw, 'Months': months
            })

    add_role('Most Recent Designation/Position (Employment)','Organization/Institute/Company/Employer','Current Employment - Start Date','Most recent Employment - End Date')
    add_role('Second Most Recent Designation/Position','Second most recent Organization/Institute/Company/Employer','Second most recent Employment - Start Date','Second most recent Employment - End Date')
    add_role('Third Most Recent Designation/Position','Third most recent Organization/Institute/Company/Employer','Third most recent Employment - Start Date','Third most recent Employment - End Date')
    add_role('Fourth Most Recent Designation/Position','Fourth most recent Organization/Institute/Company/Employer','Fourth most recent Employment - Start Date','Fourth most recent Employment - End Date')

    if roles:
        # Timeline
        if show_timeline:
            rdf = pd.DataFrame(roles)
            try:
                fig = px.timeline(rdf, x_start='Start', x_end='End', y='Organization', color='Role', hover_data=['Role','Organization'])
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Timeline not available due to inconsistent dates; showing list below.")

        # Cards list
        for r in roles:
            start_txt = r['Start'].date().isoformat() if isinstance(r['Start'], pd.Timestamp) else '—'
            end_txt = r['End'].date().isoformat() if r['RawEnd']!='—' and r['End'] else 'Present'
            months = r['Months']
            tenure = f"{months//12}y {months%12}m" if months is not None else '—'
            st.markdown(
                f"""
                <div class='role'>
                  <div class='role-top'>
                    <div class='role-title'>{r['Role']}</div>
                    <div class='role-org'>{r['Organization']}</div>
                  </div>
                  <div class='role-meta'>
                    <span class='chip'>📅 {start_txt} → {end_txt}</span>
                    <span class='chip'>⏱️ {tenure}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='qa mini'><div class='qa-label'>Total Experience</div><div class='qa-value'>{F('Total Experience (In Years & months)')}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='qa mini'><div class='qa-label'>Student Feedback (last year %)</div><div class='qa-value'>{F('Annual Student Feedback - Last year (in %)')}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='qa mini'><div class='qa-label'>Department</div><div class='qa-value'>{F('Department (Position Department)')}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --------------- Publications & Citations (enhanced) ---------------
with st.container():
    st.markdown("<div class='card section-publications'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📚 Publications & Citations</div>", unsafe_allow_html=True)

    pub_total = get_num(F('Total No. of Publications')) or 0
    pub_if = get_num(F('Total No. of Publications With Impact Factor')) or 0
    pub_nonif = get_num(F('Total No. of Publications - Non-Impact Factor')) or max(pub_total - pub_if, 0)
    w = get_num(F('No. of W Category Publications')) or 0
    x = get_num(F('No. of X Category Publications')) or 0
    y = get_num(F('No. of Y Category Publications')) or 0

    left, right = st.columns([2,1])
    with left:
        cat_df = pd.DataFrame({
            'Type':['Impact Factor','Non-Impact Factor','W Category','X Category','Y Category'],
            'Count':[pub_if, pub_nonif, w, x, y]
        })
        fig = px.bar(cat_df, x='Type', y='Count', color='Type', text='Count', height=320,
                     color_discrete_sequence=['#FF7043','#BDBDBD','#FFB74D','#FF8A65','#FFCC80'])
        fig.update_traces(textposition='outside')
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if show_if_pie:
            pie_df = pd.DataFrame({'Type':['Impact','Non-Impact'], 'Count':[pub_if, pub_nonif]})
            pie = px.pie(pie_df, values='Count', names='Type', color='Type',
                         color_discrete_map={'Impact':'#FF7043','Non-Impact':'#BDBDBD'})
            pie.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(pie, use_container_width=True)
        st.write("**Impact Factor (Σ)**", F('Total Impact Factor (number)'))
        st.write("**Conference Papers**", F('Total No. of Conference Papers'))
        st.write("**MS/MPhil Supervision**", F('MS/MPhil Supervision Load'))
        st.write("**PhD Supervision**", F('PhD Supervision Load'))
        st.write("**Citation Index**", F('Citation Index'))
        st.write("**Scopus Index Articles**", F('Scopus Index Article'))

    st.markdown("</div>", unsafe_allow_html=True)

# --------------- Achievements / Training / Projects (split) ---------------
# Achievements
with st.container():
    st.markdown("<div class='card section-achievements'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🏅 Achievements</div>", unsafe_allow_html=True)
    ach = F('ACADEMIC/PROFESSIONAL ACHIEVEMENTS/INITIATIVES (If Any)')
    items = [i.strip(' \t\n-•') for i in re.split(r'[\n\r;]+', str(ach)) if i and i.strip() and i.strip().lower() not in {'—','na','n/a','nil'}]
    if items:
        st.markdown("\n".join([f"- {i}" for i in items]))
    else:
        st.info("No achievements listed.")
    st.markdown("</div>", unsafe_allow_html=True)

# Training & Certifications
with st.container():
    st.markdown("<div class='card section-training'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎓 Trainings & Certifications</div>", unsafe_allow_html=True)
    tr = F('Training/Workshop/Diploma/Certificate (Related to Post Applied If Any)')
    titems = [i.strip(' \t\n-•') for i in re.split(r'[\n\r;]+', str(tr)) if i and i.strip() and i.strip().lower() not in {'—','na','n/a','nil'}]
    if titems:
        st.markdown("\n".join([f"- {i}" for i in titems]))
    else:
        st.info("No trainings/certifications listed.")
    st.markdown("</div>", unsafe_allow_html=True)

# Projects / Research Grants
with st.container():
    st.markdown("<div class='card section-projects'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>💡 Projects / Research Grants</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        qa("Involved in funded project?", F('Involvement in any funded project (If Any) '))
        qa("Details", F('If Involved in any Funded Project (Details)'))
    with col2:
        qa("Grant Amount", F('Grant Amount (Funded Project)'))
        qa("Self / Organization", F('Self/Organization (Funded Project)'))
    with col3:
        qa("Expected Completion", F('Expected Completion Date (Funded Project)'))
    st.markdown("</div>", unsafe_allow_html=True)

# --------------- Faculty / Strategic Unit / Department (end section) ---------------
with st.container():
    st.markdown("<div class='card section-meta'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🏛️ Affiliation</div>", unsafe_allow_html=True)

    fac_raw = F('Strategic Unit the position belongs to')
    fac_map = {
        'FSSH':'Faculty of Social Sciences & Humanities',
        'RIMS':'Riphah Institute of Media Sciences',
        'RIPP':'Riphah Institute of Public Policy',
        'DSS':'Department of Social Sciences',
        'DSS Islamic Studies':'Department of Islamic Studies'
    }
    faculty = fac_map.get(str(fac_raw).strip(), fac_raw)

    colm1, colm2, colm3 = st.columns(3)
    with colm1:
        qa("Faculty", faculty)
    with colm2:
        qa("Strategic Unit", fac_raw)
    with colm3:
        qa("Department Applied For", F('Department (Position Department)'))
    st.markdown("</div>", unsafe_allow_html=True)

# --------------- Resume ---------------
with st.expander("📎 Resume / CV"):
    st.write(F("Candidates' Resume/CV"))

st.caption("Made with ❤️ – iOS-style warm gradients, enhanced readability, and flexible UI controls")
