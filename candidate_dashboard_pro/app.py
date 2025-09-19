import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from pathlib import Path
import re, io

st.set_page_config(page_title="Candidate Profiles – Pro UI", page_icon="🌈", layout="wide")

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


def hex_to_rgb_tuple(hex_color: str):
    h = hex_color.lstrip('#')
    if len(h)==3:
        h = ''.join([c*2 for c in h])
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))


def set_theme(
    header_grad: str,
    density: str,
    header_scale: float,
    accent: str,
    base_bg: str,
    card_hex: str,
    card_alpha: float,
    font_family: str,
    font_color: str,
    base_font_scale: float,
    header_text: str,
    section_colors: dict
):
    gradients = {
        'Sunset': ('#FF7E5F', '#FEB47B'),
        'Peach':  ('#f6d365', '#fda085'),
        'Mango':  ('#f7971e', '#ffd200'),
        'Rose':   ('#feada6', '#f5efef'),
        'Coral':  ('#ff9966', '#ff5e62')
    }
    start, end = gradients.get(header_grad, gradients['Sunset'])
    density_map = {'Cozy':'18px','Compact':'12px','Roomy':'22px'}
    radius = density_map.get(density, '18px')
    header_font = max(1.0, min(header_scale, 2.8))
    base_scale = max(0.8, min(base_font_scale, 1.4))

    r,g,b = hex_to_rgb_tuple(card_hex)

    # Build section accent CSS vars
    section_css = "".join([f"--accent-{k}:{v};" for k,v in section_colors.items()])

    css = f"""
    <style>
      :root{{
        --header-grad-start: {start};
        --header-grad-end: {end};
        --accent: {accent};
        --radius: {radius};
        --header-scale: {header_font};
        --bg: {base_bg};
        --card-rgb: {r}, {g}, {b};
        --card-alpha: {card_alpha};
        --font-family: {font_family}, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, sans-serif;
        --font-color: {font_color};
        --header-text: {header_text};
        --base-scale: {base_scale};
        {section_css}
      }}
      html, body, [class^="css"]  {{
          font-family: var(--font-family);
          color: var(--font-color);
          font-size: calc(16px * var(--base-scale));
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


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

# --------------- Sidebar (collapsible UI settings) ---------------
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

    # Collapsible UI settings via button + expander
    if 'open_settings' not in st.session_state:
        st.session_state.open_settings = False
    if st.button("⚙️ UI Settings"):
        st.session_state.open_settings = not st.session_state.open_settings

    with st.expander("Customize look & layout", expanded=st.session_state.open_settings):
        st.markdown("### 🎨 Theme & Typography")
        grad = st.selectbox("Header gradient", ['Sunset','Peach','Mango','Rose','Coral'], index=0)
        density = st.radio("Card density", ['Cozy','Compact','Roomy'], index=0, horizontal=True)
        header_scale = st.slider("Header size", 1.2, 2.8, 2.2, 0.1)
        base_font_scale = st.slider("Base font scale", 0.9, 1.3, 1.0, 0.05)

        font_family = st.selectbox("Font", [
            'Inter','SF Pro Display','Segoe UI','Roboto','Nunito','Lato','Montserrat','Poppins','System UI'
        ], index=0)
        font_color = st.color_picker("Font color", value="#EDEFF7")
        header_text = st.color_picker("Header text color", value="#2b1b14")

        st.markdown("### 🧱 Surfaces & Colors")
        base_bg = st.color_picker("Dashboard background", value="#0f1220")
        card_hex = st.color_picker("Card base color", value="#141935")
        card_alpha = st.slider("Card transparency", 0.4, 1.0, 0.92, 0.02)
        accent = st.color_picker("Accent color", value="#FF8A65")

        st.markdown("### 🎯 Section accent colors")
        sc_personal = st.color_picker("Personal", value="#F6A36D")
        sc_education = st.color_picker("Education", value="#FFB74D")
        sc_experience = st.color_picker("Experience", value="#FF8A65")
        sc_publications = st.color_picker("Publications", value="#70E1F5")
        sc_achievements = st.color_picker("Achievements", value="#FAD961")
        sc_training = st.color_picker("Training", value="#8BC34A")
        sc_projects = st.color_picker("Projects", value="#B39DDB")
        sc_meta = st.color_picker("Application/Vacancy details", value="#FFA726")

        st.markdown("### 🧩 Section order")
        default_sections = [
            'Personal Details','Education','Experience','Publications & Citations',
            'Achievements','Trainings & Certifications','Projects / Research Grants', 'Application / Vacancy Details'
        ]
        order = st.multiselect("Choose order (reselect to change)", default_sections, default=default_sections)

    # Apply theme after settings
    set_theme(
        header_grad=grad if 'grad' in locals() else 'Sunset',
        density=density if 'density' in locals() else 'Cozy',
        header_scale=header_scale if 'header_scale' in locals() else 2.2,
        accent=accent if 'accent' in locals() else '#FF8A65',
        base_bg=base_bg if 'base_bg' in locals() else '#0f1220',
        card_hex=card_hex if 'card_hex' in locals() else '#141935',
        card_alpha=card_alpha if 'card_alpha' in locals() else 0.92,
        font_family=font_family if 'font_family' in locals() else 'Inter',
        font_color=font_color if 'font_color' in locals() else '#EDEFF7',
        base_font_scale=base_font_scale if 'base_font_scale' in locals() else 1.0,
        header_text=header_text if 'header_text' in locals() else '#2b1b14',
        section_colors={
            'personal': sc_personal if 'sc_personal' in locals() else '#F6A36D',
            'education': sc_education if 'sc_education' in locals() else '#FFB74D',
            'experience': sc_experience if 'sc_experience' in locals() else '#FF8A65',
            'publications': sc_publications if 'sc_publications' in locals() else '#70E1F5',
            'achievements': sc_achievements if 'sc_achievements' in locals() else '#FAD961',
            'training': sc_training if 'sc_training' in locals() else '#8BC34A',
            'projects': sc_projects if 'sc_projects' in locals() else '#B39DDB',
            'meta': sc_meta if 'sc_meta' in locals() else '#FFA726',
        }
    )

# ---------------- Selected row & helper ----------------
row = df[df[name_col]==sel].iloc[0]

def F(*keys, default='—'):
    col = find_col(df, *keys)
    if not col:
        return default
    val = row.get(col, default)
    if pd.isna(val) or str(val).strip()=='' or str(val).strip().lower() in {'na','n/a','nil','none'}:
        return default
    return val

# ---------------- Header ----------------
applied_pos = F('Applied Position')
st.markdown(f"""
<div class='header warm'>
  <div class='title' style='font-size: calc(1.55rem * var(--header-scale)); color: var(--header-text);'>🪪 {sel}</div>
  <div class='subtitle' style='color: var(--header-text); opacity: .95'>{applied_pos}</div>
  <div class='header-pills'>
    <span class='pill'>🎂 {F('Date of Birth')}</span>
    <span class='pill'>🧮 Age: {F('Your age')}</span>
    <span class='pill'>💍 {F('Marital Status')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------- KPI Row (requested fields) ----------------
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

# ---------------- Section order logic ----------------
sections = {
    'Personal Details': 'personal',
    'Education': 'education',
    'Experience': 'experience',
    'Publications & Citations': 'publications',
    'Achievements': 'achievements',
    'Trainings & Certifications': 'training',
    'Projects / Research Grants': 'projects',
    'Application / Vacancy Details': 'meta'
}

# Use chosen order from settings if available
try:
    chosen = order if len(order)>0 else list(sections.keys())
except NameError:
    chosen = list(sections.keys())

# Append missing sections if any
for s in sections.keys():
    if s not in chosen:
        chosen.append(s)

# --------------- RENDERERS ---------------

def render_personal():
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


def render_education():
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


def render_experience():
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

    # Optional timeline toggle respected if set earlier
    try:
        show_tl = show_timeline
    except NameError:
        show_tl = True

    if roles:
        if show_tl:
            rdf = pd.DataFrame(roles)
            try:
                fig = px.timeline(rdf, x_start='Start', x_end='End', y='Organization', color='Role', hover_data=['Role','Organization'])
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Timeline not available due to inconsistent dates; showing list below.")

        for r in roles:
            start_txt = r['Start'].date().isoformat() if isinstance(r['Start'], pd.Timestamp) else '—'
            end_txt = r['End'].date().isoformat() if r['RawEnd']!='—' and r['End'] else 'Present'
            months = r['Months']
            tenure = f"{months//12}y {months%12}m" if months is not None else '—'
            # Left: tenure & dates | Right: Organization (prominent)
            st.markdown(
                f"""
                <div class='role'>
                  <div class='role-top'>
                    <div class='role-left'>
                      <div class='role-title'>{r['Role']}</div>
                      <div class='role-meta'>
                        <span class='chip'>📅 {start_txt} → {end_txt}</span>
                        <span class='chip'>⏱️ {tenure}</span>
                      </div>
                    </div>
                    <div class='role-org'>🏢 {r['Organization']}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # Summary metrics (Department removed as requested)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='qa mini'><div class='qa-label'>Total Experience</div><div class='qa-value'>{F('Total Experience (In Years & months)')}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='qa mini'><div class='qa-label'>Student Feedback (last year %)</div><div class='qa-value'>{F('Annual Student Feedback - Last year (in %)')}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_publications():
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
        fig = px.bar(cat_df, x='Type', y='Count', color='Type', text='Count', height=340,
                     color_discrete_sequence=['#FF7043','#BDBDBD','#FFB74D','#FF8A65','#FFCC80'])
        fig.update_traces(textposition='outside', textfont_size=16)
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        try:
            show_pie = show_if_pie
        except NameError:
            show_pie = True
        if show_pie:
            pie_df = pd.DataFrame({'Type':['Impact','Non-Impact'], 'Count':[pub_if, pub_nonif]})
            pie = px.pie(pie_df, values='Count', names='Type', color='Type',
                         color_discrete_map={'Impact':'#FF7043','Non-Impact':'#BDBDBD'})
            pie.update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(pie, use_container_width=True)

        # Big stats
        st.markdown("""
        <div class='stat'>
          <div class='stat-label'>Impact Factor (Σ)</div>
          <div class='stat-value'>{{}}</div>
        </div>
        """.format(F('Total Impact Factor (number)')), unsafe_allow_html=True)
        st.markdown("""
        <div class='stat'>
          <div class='stat-label'>Conference Papers</div>
          <div class='stat-value'>{{}}</div>
        </div>
        """.format(F('Total No. of Conference Papers')), unsafe_allow_html=True)

        st.write("**MS/MPhil Supervision**", F('MS/MPhil Supervision Load'))
        st.write("**PhD Supervision**", F('PhD Supervision Load'))
        st.write("**Citation Index**", F('Citation Index'))
        st.write("**Scopus Index Articles**", F('Scopus Index Article'))

    st.markdown("</div>", unsafe_allow_html=True)


def render_achievements():
    st.markdown("<div class='card section-achievements'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🏅 Achievements</div>", unsafe_allow_html=True)
    ach = F('ACADEMIC/PROFESSIONAL ACHIEVEMENTS/INITIATIVES (If Any)')
    items = [i.strip(' \t\n-•') for i in re.split(r'[\n\r;]+', str(ach)) if i and i.strip() and i.strip().lower() not in {'—','na','n/a','nil'}]
    if items:
        st.markdown("\n".join([f"- {i}" for i in items]))
    else:
        st.info("No achievements listed.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_training():
    st.markdown("<div class='card section-training'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎓 Trainings & Certifications</div>", unsafe_allow_html=True)
    tr = F('Training/Workshop/Diploma/Certificate (Related to Post Applied If Any)')
    titems = [i.strip(' \t\n-•') for i in re.split(r'[\n\r;]+', str(tr)) if i and i.strip() and i.strip().lower() not in {'—','na','n/a','nil'}]
    if titems:
        st.markdown("\n".join([f"- {i}" for i in titems]))
    else:
        st.info("No trainings/certifications listed.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_projects():
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


def render_meta():
    st.markdown("<div class='card section-meta'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📄 Application / Vacancy Details</div>", unsafe_allow_html=True)
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

# --------------- Analysis + PDF Export ---------------

def render_analysis_and_export():
    st.markdown("<div class='card section-analysis'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧠 Candidate Insight & Analysis</div>", unsafe_allow_html=True)

    # Simple heuristics
    total_pubs = get_num(F('Total No. of Publications')) or 0
    if_pubs = get_num(F('Total No. of Publications With Impact Factor')) or 0
    conf = get_num(F('Total No. of Conference Papers')) or 0
    w = get_num(F('No. of W Category Publications')) or 0
    x = get_num(F('No. of X Category Publications')) or 0
    y = get_num(F('No. of Y Category Publications')) or 0
    exp_str = str(F('Total Experience (In Years & months)'))

    # Compute strengths
    strengths = []
    if if_pubs >= max(1, 0.3*max(1,total_pubs)):
        strengths.append("Solid share of Impact Factor publications")
    if w + x >= 5:
        strengths.append("Track record in W/X category journals")
    if conf >= 10:
        strengths.append("Active conference contributor")

    # Risks / gaps
    gaps = []
    if total_pubs < 5:
        gaps.append("Limited overall publications; consider ramping up outputs")
    if if_pubs == 0 and total_pubs > 0:
        gaps.append("Convert some outputs into IF journals")

    salary_now = get_num(F('What is Your Current Salary')) or 0
    salary_exp = get_num(F('What is Your Expected Salary')) or 0
    delta = None
    try:
        delta = salary_exp - salary_now if (salary_exp and salary_now) else None
    except Exception:
        delta = None

    st.markdown("**Highlights**")
    bullets = [
        f"Publications: **{int(total_pubs)}** total; IF: **{int(if_pubs)}**; Conf: **{int(conf)}**",
        f"W/X/Y: **{int(w)} / {int(x)} / {int(y)}**",
        f"Experience: **{exp_str}**",
        f"Salary gap (expected - current): **{delta:,.0f}**" if delta is not None else "Salary gap: —"
    ]
    st.markdown("\n".join([f"- {b}" for b in bullets]))

    if strengths:
        st.markdown("**Strengths**\n" + "\n".join([f"- ✅ {s}" for s in strengths]))
    if gaps:
        st.markdown("**Development Areas**\n" + "\n".join([f"- ⚠️ {g}" for g in gaps]))

    # PDF Export (simple textual card using PyMuPDF)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open()
        page = doc.new_page()  # A4 by default
        y = 40
        def write(text, size=14, bold=False):
            nonlocal y
            font = "helv" if not bold else "helvB"
            page.insert_text((40, y), text, fontsize=size, fontname=font)
            y += size + 8
        write(f"Candidate: {sel}", 20, True)
        write(f"Position: {applied_pos}", 14)
        write("")
        write("Key Numbers", 16, True)
        write(f"Total Publications: {int(total_pubs)}")
        write(f"IF Publications: {int(if_pubs)}")
        write(f"Conference Papers: {int(conf)}")
        write(f"W/X/Y: {int(w)}/{int(x)}/{int(y)}")
        write("")
        write("Experience", 16, True)
        write(f"{exp_str}")
        if delta is not None:
            write("")
            write("Compensation", 16, True)
            write(f"Current: {salary_now:,.0f} | Expected: {salary_exp:,.0f} | Gap: {delta:,.0f}")
        # Achievements (first 5 lines)
        ach = str(F('ACADEMIC/PROFESSIONAL ACHIEVEMENTS/INITIATIVES (If Any)'))
        alines = [ln for ln in re.split(r'[\n\r;]+', ach) if ln.strip()][:5]
        if alines:
            write("")
            write("Achievements", 16, True)
            for ln in alines:
                write(f"• {ln[:100]}")

        pdf_bytes = doc.tobytes()
        doc.close()
        st.download_button("📄 Export profile PDF", data=pdf_bytes, file_name=f"{sel.replace(' ','_')}_profile.pdf", mime='application/pdf')
    except Exception as e:
        st.info("PDF export requires PyMuPDF (pymupdf). If not available, add it to requirements.txt.")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------- Render by order ---------------
render_map = {
    'Personal Details': render_personal,
    'Education': render_education,
    'Experience': render_experience,
    'Publications & Citations': render_publications,
    'Achievements': render_achievements,
    'Trainings & Certifications': render_training,
    'Projects / Research Grants': render_projects,
    'Application / Vacancy Details': render_meta
}

for sec in chosen:
    render_map.get(sec, lambda: None)()

# Resume
with st.expander("📎 Resume / CV"):
    st.write(F("Candidates' Resume/CV"))

# Analysis & export at end
render_analysis_and_export()

st.caption("Made with ❤️ – Pro UI: warm gradients, customizable fonts/colors, movable sections, robust experience, and insights")
