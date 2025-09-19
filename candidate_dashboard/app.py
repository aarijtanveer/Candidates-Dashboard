import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import re

st.set_page_config(page_title="Candidate Profiles", page_icon="🧑‍🎓", layout="wide")

# ------------------------ Utilities ------------------------

def inject_css():
    with open('assets/styles.css', 'r', encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).strip().lower())


def find_col(df, *candidates):
    cols_norm = {normalize(c): c for c in df.columns}
    for cand in candidates:
        key = normalize(cand)
        # exact
        if key in cols_norm:
            return cols_norm[key]
        # contains
        for k, orig in cols_norm.items():
            if key and key in k:
                return orig
    return None


def parse_date(val):
    if pd.isna(val) or str(val).strip()=='' or str(val).strip().lower() in {'na','n/a','nil','none','\n','-','—','\\-'}:
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


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    opts = dict(encoding='utf-8-sig', engine='python')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, **opts, on_bad_lines='skip')
    else:
        try:
            df = pd.read_csv('data/Dashboard Format.csv', **opts, on_bad_lines='skip')
        except Exception:
            st.warning("Couldn't find sample CSV in data/. Please upload a file.")
            return pd.DataFrame()

    # Drop fully empty rows
    df.dropna(how='all', inplace=True)

    # Strip whitespace from headers
    df.columns = [c.strip() for c in df.columns]

    return df


def get_row_map(df, row):
    """Return a dict-like resolver to fetch fields by pattern"""
    def field(*patterns, default='—'):
        col = find_col(df, *patterns)
        if not col:
            return default
        val = row.get(col, default)
        if pd.isna(val) or str(val).strip()=='':
            return default
        return val
    return field


def kpi(value, label):
    col = st.container()
    with col:
        st.markdown(f"<div class='kpi'><div class='value'>{value}</div><div class='label'>{label}</div></div>", unsafe_allow_html=True)


# ------------------------ Sidebar ------------------------
inject_css()
if 'light' not in st.session_state:
    st.session_state.light = False

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    uploaded = st.file_uploader("Upload CSV (same format as sample)", type=['csv'])
    df = load_data(uploaded)

    if df.empty:
        st.stop()

    # Candidate selector
    name_col = find_col(df, 'Name') or df.columns[1]
    names = sorted([str(x).strip() for x in df[name_col].dropna().unique() if str(x).strip()!=''])
    search = st.text_input("Search name")
    options = [n for n in names if search.lower() in n.lower()] if search else names
    sel = st.selectbox("Select candidate", options)

    st.write("\n")
    # (Light/Dark runtime toggle removed for Streamlit safety)


# ------------------------ Selected row ------------------------
row = df[df[name_col]==sel].iloc[0]
F = get_row_map(df, row)

# Header
applied_pos = F('Applied Position')
st.markdown(f"""
<div class='header'>
  <div class='title'>📇 {sel}</div>
  <div class='subtitle'>{applied_pos}</div>
  <div style='margin-top:.5rem;'>
    <span class='pill'>🎂 {F('Date of Birth')}</span>
    <span class='pill'>🧮 Age: {F('Your age')}</span>
    <span class='pill'>💍 {F('Marital Status')}</span>
    <span class='pill'>🏢 Dept: {F('Department (Position Department)','Strategic Unit the position belongs to')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi(F('Total No. of Publications'), 'Publications (Total)')
with col2:
    kpi(F('Total No. of Publications With Impact Factor'), 'With Impact Factor')
with col3:
    kpi(F('Total Impact Factor (number)'), 'Impact Factor (Σ)')
with col4:
    kpi(F('Total No. of Conference Papers'), 'Conference Papers')

st.write("")

# ------------------------ Sections ------------------------
# Personal Details
with st.container():
    st.markdown("<div class='card section-personal'>", unsafe_allow_html=True)
    st.markdown("<div class='gradient-title'>👤 Personal Details</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write("**Name**", sel)
        st.write("**Date of Birth**", F('Date of Birth'))
        st.write("**Age**", F('Your age'))
    with c2:
        st.write("**Marital Status**", F('Marital Status'))
        st.write("**Previously at Riphah**", F('Previously employed at Riphah?'))
        st.write("**Strategic Unit**", F('Strategic Unit the position belongs to'))
    with c3:
        st.write("**Department**", F('Department (Position Department)'))
        st.write("**Expected Salary**", F('What is Your Expected Salary'))
        st.write("**Current Salary**", F('What is Your Current Salary'))
    with c4:
        st.write("**Experience Cities**", ', '.join([
            str(F('Experience City 1','—')),
            str(F('Experience City 2','—')),
            str(F('Experience City 3','—')),
            str(F('Experience City 4','—'))
        ]).replace('—','').replace(', ,', ', ').strip(', '))
        st.write("**Relatives at Riphah**", F('Any Relative(s) working in Riphah?'))
        st.write("**Health**", F('Any Prolong illness / disease'))
    st.markdown("</div>", unsafe_allow_html=True)

# Education
with st.container():
    st.markdown("<div class='card section-education'>", unsafe_allow_html=True)
    st.markdown("<div class='gradient-title'>🎓 Education</div>", unsafe_allow_html=True)

    edu_rows = []
    # Latest degree
    edu_rows.append({
        'Level': F('Latest Qualification/Degree (Completed)'),
        'Major / Title': F('Major/Specialization With Complete Degree Title'),
        'Result': f"{F('Obtained / Awarded GPA/Grade/Percentage/Marks')} / {F('Total GPA/Grade/Percentage/Marks')}",
        'Start': F('Last Degree - Start Date'),
        'End / Graduation': F('Most Recent Degree - Graduation Date'),
        'University': F('Degree Awarding University'),
        'World Rank': F("Latest Degree Awarding University's World Ranking"),
        'Asia Rank': F("Latest Degree Awarding University's Asia Ranking (In case of Asian Universities)")
    })
    # Second last
    edu_rows.append({
        'Level': F('Second Last Qualification/Degree (Completed)'),
        'Major / Title': F('Major/Specialization With Complete Degree Title2'),
        'Result': f"{F('Obtained / Awarded GPA/Grade/Percentage/Marks3')} / {F('Total GPA/Grade/Percentage/Marks2')}",
        'Start': F('Second Last Degree - Start Date'),
        'End / Graduation': F('Second Last Degree - Graduation Date'),
        'University': F('Second Last Degree Awarding University'),
        'World Rank': '—', 'Asia Rank': '—'
    })
    # In process
    in_process = F('Qualification/Degree In Process (If Any)')
    if str(in_process) not in {'—', 'NA', 'N/A', 'Nil', 'nil', ''}:
        edu_rows.append({
            'Level': in_process + ' (In Process)',
            'Major / Title': F('Major/Specialization of Degree (In Process)'),
            'Result': F('GPA/Grade/Percentage'),
            'Start': F('In Process Degree Start Date'),
            'End / Graduation': F('End Date (Expected)'),
            'University': '—', 'World Rank':'—', 'Asia Rank':'—'
        })

    edf = pd.DataFrame(edu_rows)
    st.dataframe(edf, use_container_width=True, hide_index=True)

    # Durations
    st.markdown("\n")
    st.markdown(
        f"<span class='pill'>🧪 PhD Duration: {F('Duration of Course PhD')}</span> "
        f"<span class='pill'>🧪 M.Phil: {F('Duration of M.Phil Course')}</span> "
        f"<span class='pill'>🧪 Masters: {F('Duration of Masters Course')}</span>",
        unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Experience
with st.container():
    st.markdown("<div class='card section-experience'>", unsafe_allow_html=True)
    st.markdown("<div class='gradient-title'>💼 Experience</div>", unsafe_allow_html=True)

    roles = []
    def add_role(title_key, org_key, start_key, end_key):
        title = F(title_key)
        org = F(org_key)
        start = parse_date(F(start_key))
        end_raw = F(end_key)
        end = parse_date(end_raw)
        if str(title)!='—' or str(org)!='—':
            roles.append({
                'Role': title, 'Organization': org,
                'Start': start, 'End': end if end else datetime.today()
            })
    add_role('Most Recent Designation/Position (Employment)','Organization/Institute/Company/Employer','Current Employment - Start Date','Most recent Employment - End Date')
    add_role('Second Most Recent Designation/Position','Second most recent Organization/Institute/Company/Employer','Second most recent Employment - Start Date','Second most recent Employment - End Date')
    add_role('Third Most Recent Designation/Position','Third most recent Organization/Institute/Company/Employer','Third most recent Employment - Start Date','Third most recent Employment - End Date')
    add_role('Fourth Most Recent Designation/Position','Fourth most recent Organization/Institute/Company/Employer','Fourth most recent Employment - Start Date','Fourth most recent Employment - End Date')

    if roles:
        rdf = pd.DataFrame(roles)
        # Timeline chart
        try:
            fig = px.timeline(rdf, x_start='Start', x_end='End', y='Organization', color='Role', hover_data=['Role','Organization'])
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.dataframe(rdf.drop(columns=['End']), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Total Experience**", F('Total Experience (In Years & months)'))
    with c2:
        st.write("**Student Feedback (last year %)**", F('Annual Student Feedback - Last year (in %)'))
    with c3:
        st.write("**Department**", F('Department (Position Department)'))

    st.markdown("</div>", unsafe_allow_html=True)

# Publications & Supervision
with st.container():
    st.markdown("<div class='card section-publications'>", unsafe_allow_html=True)
    st.markdown("<div class='gradient-title'>📚 Publications & Supervision</div>", unsafe_allow_html=True)

    pub_total = get_num(F('Total No. of Publications')) or 0
    pub_if = get_num(F('Total No. of Publications With Impact Factor')) or 0
    pub_nonif = get_num(F('Total No. of Publications - Non-Impact Factor')) or (pub_total - pub_if)
    w = get_num(F('No. of W Category Publications')) or 0
    x = get_num(F('No. of X Category Publications')) or 0
    y = get_num(F('No. of Y Category Publications')) or 0

    left, right = st.columns([2,1])
    with left:
        p_df = pd.DataFrame({
            'Type':['Impact Factor','Non-Impact Factor','W Category','X Category','Y Category'],
            'Count':[pub_if, pub_nonif, w, x, y]
        })
        fig = px.bar(p_df, x='Type', y='Count', color='Type', text='Count', height=320)
        fig.update_traces(textposition='outside')
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.write("**Impact Factor (Σ)**", F('Total Impact Factor (number)'))
        st.write("**Conference Papers**", F('Total No. of Conference Papers'))
        st.write("**Supervision (MS/MPhil)**", F('MS/MPhil Supervision Load'))
        st.write("**Supervision (PhD)**", F('PhD Supervision Load'))
        st.write("**Citation Index**", F('Citation Index'))
        st.write("**Scopus Index Articles**", F('Scopus Index Article'))

    st.markdown("</div>", unsafe_allow_html=True)

# Achievements & Training
with st.container():
    st.markdown("<div class='card section-achievements'>", unsafe_allow_html=True)
    st.markdown("<div class='gradient-title'>🏅 Achievements, Training & Projects</div>", unsafe_allow_html=True)
    st.write("**Achievements / Initiatives**")
    st.write(F('ACADEMIC/PROFESSIONAL ACHIEVEMENTS/INITIATIVES (If Any)'))

    st.write("**Training / Workshops (Related)**")
    st.write(F('Training/Workshop/Diploma/Certificate (Related to Post Applied If Any)'))

    st.write("**Funded Projects**")
    colA, colB, colC = st.columns(3)
    with colA:
        st.write("Involved?", F('Involvement in any funded project (If Any) '))
        st.write("Details", F('If Involved in any Funded Project (Details)'))
    with colB:
        st.write("Grant Amount", F('Grant Amount (Funded Project)'))
        st.write("Self / Organization", F('Self/Organization (Funded Project)'))
    with colC:
        st.write("Expected Completion", F('Expected Completion Date (Funded Project)'))

    st.markdown("</div>", unsafe_allow_html=True)

# Resume
with st.expander("📎 Resume / CV"):
    st.write(F("Candidates' Resume/CV"))

# Footer
st.caption("Made with ❤️ in Streamlit – iOS-style gradients & cards")
