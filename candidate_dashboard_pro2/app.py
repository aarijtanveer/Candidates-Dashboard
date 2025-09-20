import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from pathlib import Path
import re

st.set_page_config(page_title="Candidate Profiles – Pro UI v2", page_icon="🌈", layout="wide")

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
    # Harden parsing for various odd formats
    if val is None:
        return None
    s = str(val).strip()
    if s == '' or s.lower() in {'na','n/a','nil','none','-','—','\\-'}:
        return None
    # Fix common short patterns like 'Dec-14'
    s = re.sub(r'([A-Za-z]{3})-(\d{2})$', r'\1 20\2', s)
    try:
        return pd.to_datetime(s, dayfirst=True, errors='coerce')
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
    if value in ('—', '', None):
        return
    label = str(label).strip()
    val = str(value)
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

    # Settings toggle
    if 'open_settings' not in st.session_state:
        st.session_state.open_settings = True
    if st.button("⚙️ UI Settings"):
        st.session_state.open_settings = not st.session_state.open_settings

    with st.expander("Customize look & layout", expanded=st.session_state.open_settings):
        tab1, tab2, tab3 = st.tabs(["Theme & Fonts", "Surfaces & Colors", "Sections & Layout"]) 
        with tab1:
            grad = st.selectbox("Header gradient", ['Sunset','Peach','Mango','Rose','Coral'], index=0)
            density = st.radio("Card density", ['Cozy','Compact','Roomy'], index=1, horizontal=True)
            header_scale = st.slider("Header size", 1.2, 2.8, 2.2, 0.1)
            base_font_scale = st.slider("Base font scale", 0.9, 1.3, 1.0, 0.05)
            font_family = st.selectbox("Font", ['Inter','SF Pro Display','Segoe UI','Roboto','Nunito','Lato','Montserrat','Poppins','System UI'], index=0)
            font_color = st.color_picker("Font color", value="#EDEFF7")
            header_text = st.color_picker("Header text color", value="#1a0f0a")
        with tab2:
            base_bg = st.color_picker("Dashboard background", value="#0f1220")
            card_hex = st.color_picker("Card base color", value="#141935")
            card_alpha = st.slider("Card transparency", 0.40, 1.0, 0.92, 0.02)
            accent = st.color_picker("Accent color", value="#FF8A65")
            st.markdown("**Section accents**")
            sc_personal = st.color_picker("Personal", value="#F6A36D")
            sc_education = st.color_picker("Education", value="#FFB74D")
            sc_experience = st.color_picker("Experience", value="#FF8A65")
            sc_publications = st.color_picker("Publications", value="#70E1F5")
            sc_achievements = st.color_picker("Achievements", value="#FAD961")
            sc_training = st.color_picker("Training", value="#8BC34A")
            sc_projects = st.color_picker("Projects", value="#B39DDB")
            sc_meta = st.color_picker("Application/Vacancy", value="#FFA726")
        with tab3:
            st.markdown("**Compact layout (reduce scrolling)**")
            compact = st.toggle("Enable compact mosaic layout", value=True)
            kpi_cols = st.slider("KPI tiles per row", 3, 5, 5)
            grid_cols = st.slider("Mosaic columns", 2, 3, 3)
            collapse_long_lists = st.toggle("Collapse long lists (Achievements/Training)", value=True)
            default_sections = [
                'Personal Details','Education','Experience','Publications & Citations',
                'Achievements','Trainings & Certifications','Projects / Research Grants', 'Application / Vacancy Details'
            ]
            order = st.multiselect("Section order", default_sections, default=default_sections)

    # Apply theme immediately after settings
    set_theme(
        header_grad=grad,
        density=density,
        header_scale=header_scale,
        accent=accent,
        base_bg=base_bg,
        card_hex=card_hex,
        card_alpha=card_alpha,
        font_family=font_family,
        font_color=font_color,
        base_font_scale=base_font_scale,
        header_text=header_text,
        section_colors={
            'personal': sc_personal,
            'education': sc_education,
            'experience': sc_experience,
            'publications': sc_publications,
            'achievements': sc_achievements,
            'training': sc_training,
            'projects': sc_projects,
            'meta': sc_meta,
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
  <div class='title' style='font-size: calc(1.55rem * var(--header-scale)); color: var(--header-text); text-shadow:0 1px 2px rgba(0,0,0,.25);'>🪪 {sel}</div>
  <div class='subtitle' style='color: var(--header-text); opacity: .95; max-width:72ch; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{applied_pos}</div>
  <div class='header-pills'>
    <span class='pill' title='Date of Birth'>🎂 {F('Date of Birth')}</span>
    <span class='pill' title='Age'>🧮 {F('Your age')}</span>
    <span class='pill' title='Marital Status'>💍 {F('Marital Status')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------- KPI Row ----------------
k1 = F('Total No. of Publications')
k2 = F('Total No. of Publications With Impact Factor')
k3 = F('No. of W Category Publications')
k4 = F('No. of X Category Publications')
k5 = F('No. of Y Category Publications')

# Dynamic KPI columns
try:
    kcols = st.columns(kpi_cols)
except Exception:
    kcols = st.columns(5)

for c, v, label, tip in [
    (kcols[0], k1, 'Total Publications', 'Total No. of Publications'),
    (kcols[1], k2, 'With Impact Factor', 'Total No. of Publications With Impact Factor'),
    (kcols[2], k3, 'W Category', 'No. of W Category Publications'),
    (kcols[3], k4, 'X Category', 'No. of X Category Publications'),
    (kcols[4], k5, 'Y Category', 'No. of Y Category Publications'),
]:
    with c:
        st.markdown(f"<div class='kpi' title='{tip}'><div class='value'>{v}</div><div class='label'>{label}</div></div>", unsafe_allow_html=True)

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
try:
    chosen = order if len(order)>0 else list(sections.keys())
except Exception:
    chosen = list(sections.keys())
for s in sections.keys():
    if s not in chosen:
        chosen.append(s)

# ---------------- Renderers ----------------

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
        qa("Experience Cities", cities if cities else None)
        qa("Relatives at Riphah", F('Any Relative(s) working in Riphah?'))
        qa("Health (if any)", F('Any Prolong illness / disease'))
    st.markdown("</div>", unsafe_allow_html=True)


def render_education():
    st.markdown("<div class='card section-education'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎓 Education</div>", unsafe_allow_html=True)

    def degree_card(level, title, result, start, end, uni, wrank, arank):
        # Skip empty level & title cards
        if level in ('—', '', None) and title in ('—','',None):
            return
        # hide ranks if missing
        wr = '' if wrank in ('—','',None) else f"<div><span class='deg-label'>World Rank</span><span class='deg-val'>{wrank}</span></div>"
        ar = '' if arank in ('—','',None) else f"<div><span class='deg-label'>Asia Rank</span><span class='deg-val'>{arank}</span></div>"
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
                    {wr}
                    {ar}
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
        None, None
    )
    inproc = F('Qualification/Degree In Process (If Any)')
    if str(inproc) not in {'—','', 'NA','N/A','Nil','nil'}:
        degree_card(
            inproc + ' (In Process)',
            F('Major/Specialization of Degree (In Process)'),
            F('GPA/Grade/Percentage'),
            F('In Process Degree Start Date'),
            F('End Date (Expected)'),
            '—', None, None
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

    try:
        show_tl = compact is False  # show timeline only in non-compact to save vertical space
    except Exception:
        show_tl = True

    if roles:
        if show_tl:
            rdf = pd.DataFrame(roles)
            try:
                fig = px.timeline(rdf, x_start='Start', x_end='End', y='Organization', color='Role', hover_data=['Role','Organization'])
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(height=280, margin=dict(l=8,r=8,t=24,b=8))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Timeline not available due to inconsistent dates; showing list below.")

        for r in roles:
            start_txt = r['Start'].date().isoformat() if isinstance(r['Start'], pd.Timestamp) else '—'
            end_txt = r['End'].date().isoformat() if r['RawEnd']!='—' and r['End'] else 'Present'
            months = r['Months']
            tenure = f"{months//12}y {months%12}m" if months is not None else '—'
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
                    <div class='role-org' title='{r['Organization']}'>🏢 {r['Organization']}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

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
        cat_df = pd.DataFrame({'Type':['Impact Factor','Non-Impact Factor','W Category','X Category','Y Category'], 'Count':[pub_if, pub_nonif, w, x, y]})
        fig = px.bar(cat_df, x='Type', y='Count', color='Type', text='Count', height=300, color_discrete_sequence=['#FF7043','#BDBDBD','#FFB74D','#FF8A65','#FFCC80'])
        fig.update_traces(textposition='outside', textfont_size=15, cliponaxis=False)
        fig.update_layout(margin=dict(l=8,r=8,t=8,b=8))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if (pub_if + pub_nonif) > 0 and min(pub_if, pub_nonif) > 0:
            pie_df = pd.DataFrame({'Type':['Impact','Non-Impact'], 'Count':[pub_if, pub_nonif]})
            pie = px.pie(pie_df, values='Count', names='Type', color='Type', color_discrete_map={'Impact':'#FF7043','Non-Impact':'#BDBDBD'})
            pie.update_layout(height=300, margin=dict(l=8,r=8,t=8,b=8))
            st.plotly_chart(pie, use_container_width=True)
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
    ach = str(F('ACADEMIC/PROFESSIONAL ACHIEVEMENTS/INITIATIVES (If Any)'))
    items = [i.strip(' \t\n-•') for i in re.split(r'[\n\r;]+', ach) if i and i.strip() and i.strip().lower() not in {'—','na','n/a','nil'}]
    if items:
        if 'collapse_long_lists' in globals() and collapse_long_lists and len(items)>6:
            with st.expander(f"Show {len(items)} items", expanded=False):
                st.markdown("\n".join([f"- {i}" for i in items]))
        else:
            st.markdown("\n".join([f"- {i}" for i in items]))
    else:
        st.info("No achievements listed.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_training():
    st.markdown("<div class='card section-training'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎓 Trainings & Certifications</div>", unsafe_allow_html=True)
    tr = str(F('Training/Workshop/Diploma/Certificate (Related to Post Applied If Any)'))
    titems = [i.strip(' \t\n-•') for i in re.split(r'[\n\r;]+', tr) if i and i.strip() and i.strip().lower() not in {'—','na','n/a','nil'}]
    if titems:
        if 'collapse_long_lists' in globals() and collapse_long_lists and len(titems)>6:
            with st.expander(f"Show {len(titems)} items", expanded=False):
                st.markdown("\n".join([f"- {i}" for i in titems]))
        else:
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

def insight_ratio(a,b):
    tot = (a or 0) + (b or 0)
    return (a or 0)/tot if tot else 0.0


def render_analysis_and_export():
    st.markdown("<div class='card section-analysis'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧠 Candidate Insight & Analysis</div>", unsafe_allow_html=True)

    total_pubs = get_num(F('Total No. of Publications')) or 0
    if_pubs = get_num(F('Total No. of Publications With Impact Factor')) or 0
    conf = get_num(F('Total No. of Conference Papers')) or 0
    w = get_num(F('No. of W Category Publications')) or 0
    x = get_num(F('No. of X Category Publications')) or 0
    y = get_num(F('No. of Y Category Publications')) or 0
    non_if = max(total_pubs - if_pubs, 0)
    if_share = insight_ratio(if_pubs, non_if)
    exp_str = str(F('Total Experience (In Years & months)'))

    # Basic star-score out of 5 based on simple weighted metrics
    score = 0
    score += min(total_pubs/20, 2.0)      # up to 2 points
    score += min(if_share*2.0, 1.5)       # up to 1.5 points
    score += min((w+x)/10, 1.0)           # up to 1 point
    score += min(conf/20, 0.5)            # up to 0.5 point
    score = round(score, 2)

    st.markdown(f"**Highlights**")
    bullets = [
        f"Publications: **{int(total_pubs)}** total; IF: **{int(if_pubs)}** ({if_share:.0%})",
        f"W/X/Y: **{int(w)} / {int(x)} / {int(y)}**",
        f"Experience: **{exp_str}**",
        f"Composite research score: **{score}/5** (heuristic)"
    ]
    st.markdown("\n".join([f"- {b}" for b in bullets]))

    strengths = []
    if if_share >= 0.25:
        strengths.append("Healthy proportion of IF publications")
    if (w + x) >= 5:
        strengths.append("Strong record in W/X category journals")
    if conf >= 8:
        strengths.append("Active conference presence (≥ 8)")

    gaps = []
    if total_pubs < 10:
        gaps.append("Low overall publication count — target ≥ 10")
    if if_pubs == 0 and total_pubs > 0:
        gaps.append("Convert some outputs to Impact Factor journals")
    if (w + x + y) == 0 and total_pubs > 0:
        gaps.append("No W/X/Y categories — aim for recognized venues")

    if strengths:
        st.markdown("**Strengths**\n" + "\n".join([f"- ✅ {s}" for s in strengths]))
    if gaps:
        st.markdown("**Development Areas**\n" + "\n".join([f"- ⚠️ {g}" for g in gaps]))

    # PDF via PyMuPDF if installed
    try:
        import fitz  # type: ignore
        doc = fitz.open()
        page = doc.new_page()
        y = 40
        def write(text, size=14, bold=False):
            nonlocal y
            font = "helvB" if bold else "helv"
            page.insert_text((40, y), text, fontsize=size, fontname=font)
            y += size + 8
        write(f"Candidate: {sel}", 20, True)
        write(f"Position: {F('Applied Position')}", 14)
        write("")
        write("Key Numbers", 16, True)
        write(f"Total Publications: {int(total_pubs)}")
        write(f"IF Publications: {int(if_pubs)} (share {if_share:.0%})")
        write(f"Conference Papers: {int(conf)}")
        write(f"W/X/Y: {int(w)}/{int(x)}/{int(y)}")
        write("")
        write("Experience", 16, True)
        write(f"{exp_str}")
        write("")
        write("Heuristic Research Score", 16, True)
        write(f"{score}/5")
        if strengths:
            write("")
            write("Strengths", 16, True)
            for s in strengths:
                write(f"• {s}")
        if gaps:
            write("")
            write("Development Areas", 16, True)
            for g in gaps:
                write(f"• {g}")
        pdf_bytes = doc.tobytes(); doc.close()
        st.download_button("📄 Export profile PDF", data=pdf_bytes, file_name=f"{sel.replace(' ','_')}_profile.pdf", mime='application/pdf')
    except Exception:
        st.info("PDF export requires PyMuPDF (pymupdf) which is included in requirements.txt. If running in an environment without it, install first.")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------- Compact mosaic renderer ---------------
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

try:
    use_compact = compact
except Exception:
    use_compact = True

if use_compact:
    # Mosaic columns
    try:
        cols = st.columns(grid_cols)
    except Exception:
        cols = st.columns(3)
    # Distribute sections round-robin into columns to reduce scrolling
    idx = 0
    for sec in chosen:
        with cols[idx % len(cols)]:
            render_map.get(sec, lambda: None)()
        idx += 1
else:
    for sec in chosen:
        render_map.get(sec, lambda: None)()

# Resume (collapsed)
with st.expander("📎 Resume / CV", expanded=False):
    st.write(F("Candidates' Resume/CV"))

# Analysis & export at end (full width)
render_analysis_and_export()

st.caption("Pro UI v2 – Compact mosaic layout, working settings, stronger contrast, and in-depth analysis")
