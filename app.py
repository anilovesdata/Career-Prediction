import streamlit as st
import pandas as pd
import os
import joblib
import sys

import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList:
        pass
    _ct._RemainderColsList = _RemainderColsList

st.set_page_config(
    page_title="TechPath : A Career Recommendation App",
    page_icon="🧭",
    layout="centered",
)

# Subject: a tool that maps someone's skills and interests onto
# a tech career path. The visual language leans on that idea 
# a route, waypoints, a destination  rather than a generic
# "AI dashboard" look.
#
# Color
#   --ink       #0B1220  background
#   --panel     #121A2B  card surface
#   --border    #232D45  hairline borders
#   --text      #E8ECF4  primary text
#   --muted     #8A93A6  secondary text
#   --gold      #E8A93E  the path — in-progress / structure
#   --teal      #2DD4BF  the destination — result / confidence
#   --danger    #F2545B  errors
#
# Type
#   Display : Space Grotesk  (headings, the route labels)
#   Body    : Inter          (form labels, copy)
#   Utility : JetBrains Mono (numbers, ratings, scores)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --ink: #0B1220;
    --panel: #121A2B;
    --border: #232D45;
    --text: #E8ECF4;
    --muted: #8A93A6;
    --gold: #E8A93E;
    --teal: #2DD4BF;
    --danger: #F2545B;
}

.stApp {
    background: var(--ink);
    color: var(--text);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header {visibility: hidden;}

/* ---------- Headings & display type ---------- */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}

.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.4rem;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.15;
    color: var(--text);
    margin-bottom: 0.5rem;
}

.hero-sub {
    font-family: 'Inter', sans-serif;
    color: var(--muted);
    font-size: 1rem;
    max-width: 42rem;
    margin-bottom: 1.6rem;
}

/* ---------- Route map (signature element) ---------- */
.route-wrap {
    margin: 0.5rem 0 2rem 0;
}
.route-caption {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-align: center;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
}

/* ---------- Section headers acting as waypoints ---------- */
.section-block {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1.5rem 1.1rem 1.5rem;
    margin-bottom: 1.1rem;
}
.section-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--gold);
    letter-spacing: 0.1em;
    margin-bottom: 0.15rem;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.3rem;
}
.section-hint {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 0.8rem;
}

/* ---------- Form widgets ---------- */
[data-testid="stForm"] {
    border: none;
    padding: 0;
    background: transparent;
}

label, .stMarkdown p {
    color: var(--text) !important;
}

[data-testid="stWidgetLabel"] p {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: var(--text) !important;
    font-weight: 500;
}

/* Sliders */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--gold) !important;
    border-color: var(--gold) !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background: var(--gold) !important;
}

/* Number inputs, select boxes, text inputs */
[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div,
[data-testid="stTextInput"] input {
    background-color: var(--ink) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Radio pills */
[data-testid="stRadio"] label {
    background: var(--ink);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    margin-right: 0.3rem;
}
[data-testid="stRadio"] input:checked + div {
    color: var(--gold) !important;
}

/* Submit button — the "start the journey" action */
[data-testid="stFormSubmitButton"] button {
    background: var(--gold);
    color: var(--ink);
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1.6rem;
    width: 100%;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 0 0 3px rgba(232, 169, 62, 0.25);
    transform: translateY(-1px);
}
[data-testid="stFormSubmitButton"] button:focus-visible {
    outline: 2px solid var(--teal);
    outline-offset: 2px;
}

/* ---------- Result / destination card ---------- */
.destination-card {
    background: linear-gradient(180deg, rgba(45,212,191,0.10), rgba(18,26,43,0.4));
    border: 1px solid var(--teal);
    border-radius: 12px;
    padding: 1.6rem;
    margin-top: 1rem;
    margin-bottom: 1.2rem;
}
.destination-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    color: var(--teal);
    margin-bottom: 0.3rem;
}
.destination-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--text);
}

.profile-caption {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 1.4rem 0 0.4rem 0;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 8px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

SECTIONS = [
    {"key": "profile", "label": "Profile", "eyebrow": "WAYPOINT 01",
     "title": "Who you are", "hint": "The basics — used only to contextualize your results."},
    {"key": "education", "label": "Education", "eyebrow": "WAYPOINT 02",
     "title": "Education & certifications", "hint": "Your formal background in tech."},
    {"key": "skills", "label": "Skills", "eyebrow": "WAYPOINT 03",
     "title": "Skill ratings", "hint": "Rate yourself honestly — 1 is novice, 10 is expert."},
    {"key": "stack", "label": "Stack", "eyebrow": "WAYPOINT 04",
     "title": "Technical knowledge", "hint": "Languages and tools you're comfortable with."},
    {"key": "interests", "label": "Interests", "eyebrow": "WAYPOINT 05",
     "title": "Career interests", "hint": "What pulls your attention when you're free to choose."},
    {"key": "experience", "label": "Experience", "eyebrow": "WAYPOINT 06",
     "title": "Experience & work preferences", "hint": "How you work now, and how you'd like to."},
]


def render_route_map(active_index=None):
    """Draws the waypoint map: the product's signature visual.
    Numbered, because the form genuinely is a sequence the user
    moves through — the numbers carry real order information."""
    n = len(SECTIONS)
    width = 640
    margin = 60
    gap = (width - 2 * margin) / (n - 1)
    y = 34

    circles = []
    labels = []
    for i, sec in enumerate(SECTIONS):
        cx = margin + i * gap
        is_active = (active_index is not None and i == active_index)
        is_done = (active_index is not None and i < active_index)
        fill = "var(--teal)" if is_active else ("var(--gold)" if is_done else "#1B2540")
        stroke = "var(--teal)" if is_active else "var(--gold)"
        circles.append(
            f'<circle cx="{cx}" cy="{y}" r="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            f'<text x="{cx}" y="{y+4}" font-family="JetBrains Mono, monospace" font-size="9" '
            f'text-anchor="middle" fill="{"var(--ink)" if (is_active or is_done) else "var(--muted)"}">{i+1}</text>'
        )
        labels.append(
            f'<text x="{cx}" y="{y+26}" font-family="Inter, sans-serif" font-size="10.5" '
            f'text-anchor="middle" fill="var(--muted)">{sec["label"]}</text>'
        )

    line = f'<line x1="{margin}" y1="{y}" x2="{width - margin}" y2="{y}" stroke="var(--border)" stroke-width="2"/>'

    svg = f"""
    <div class="route-wrap">
        <svg viewBox="0 0 {width} {y+40}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
            {line}
            {''.join(circles)}
            {''.join(labels)}
        </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)


def section_header(sec):
    st.markdown(
        f'<div class="section-eyebrow">{sec["eyebrow"]}</div>'
        f'<div class="section-title">{sec["title"]}</div>'
        f'<div class="section-hint">{sec["hint"]}</div>',
        unsafe_allow_html=True,
    )


def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


model = load_model()

st.markdown('<div class="hero-eyebrow">TECHPATH</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Find the tech career your skills already point to</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Answer six short sections about your background, skills, and preferences. '
    'The model reads them the way a mentor would, and points to the career path with the strongest match.</div>',
    unsafe_allow_html=True,
)

render_route_map(active_index=None)
st.markdown('<div class="route-caption">SIX WAYPOINTS TO YOUR RECOMMENDATION</div>', unsafe_allow_html=True)

if model is None:
    st.warning(
        "**model.pkl not found.** Place your trained model file in the same folder as this "
        "app, then refresh — the form below will still let you explore the inputs."
    )


with st.form("career_prediction_form"):

    # --- Waypoint 1: Profile ---
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section_header(SECTIONS[0])
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=18, max_value=80, value=28, step=1)
    with col2:
        gender = st.selectbox("Gender", options=["Male", "Female", "Non-binary", "Other", "Prefer not to say"])
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Waypoint 2: Education ---
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section_header(SECTIONS[1])
    col3, col4 = st.columns(2)
    with col3:
        education = st.selectbox(
            "Education level (1 = High School, 5 = PhD)",
            options=[1, 2, 3, 4, 5], index=3,
        )
    with col4:
        education_field = st.selectbox("Education field", options=[
            "Computer Science", "Information Technology", "Engineering",
            "Mathematics", "Physics", "Business", "Other"
        ])
    highest_certification = st.selectbox("Highest certification", options=[
        "None", "AWS Machine Learning", "AWS Cloud Practitioner", "GCP Professional Data Engineer",
        "Azure Fundamentals", "CISSP", "CompTIA Security+", "Other"
    ], index=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Waypoint 3: Skills ---
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section_header(SECTIONS[2])
    col5, col6 = st.columns(2)
    with col5:
        coding_skill = st.slider("Coding skill", 1, 10, 9)
        mathematics_rating = st.slider("Mathematics", 1, 10, 7)
        creativity_rating = st.slider("Creativity", 1, 10, 6)
        security_awareness = st.slider("Security awareness", 1, 10, 5)
    with col6:
        systems_thinking = st.slider("Systems thinking", 1, 10, 8)
        data_analysis = st.slider("Data analysis", 1, 10, 8)
        communication = st.slider("Communication", 1, 10, 6)
        design_sensibility = st.slider("Design sensibility", 1, 10, 4)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Waypoint 4: Stack ---
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section_header(SECTIONS[3])
    col7, col8, col9 = st.columns(3)
    with col7:
        knows_python = st.radio("Python?", options=["Yes", "No"], index=0)
        knows_java = st.radio("Java?", options=["Yes", "No"], index=1)
    with col8:
        knows_javascript = st.radio("JavaScript?", options=["Yes", "No"], index=0)
        knows_bash_linux = st.radio("Bash / Linux?", options=["Yes", "No"], index=0)
    with col9:
        knows_sql = st.radio("SQL?", options=["Yes", "No"], index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Waypoint 5: Interests ---
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section_header(SECTIONS[4])
    col10, col11, col12 = st.columns(3)
    with col10:
        interested_in_ai = st.radio("AI?", options=["Yes", "No"], index=0)
        interested_in_web_dev = st.radio("Web dev?", options=["Yes", "No"], index=0)
    with col11:
        interested_in_security = st.radio("Security?", options=["Yes", "No"], index=1)
        interested_in_cloud = st.radio("Cloud computing?", options=["Yes", "No"], index=0)
    with col12:
        interested_in_data_science = st.radio("Data science?", options=["Yes", "No"], index=0)
        interested_in_mobile_dev = st.radio("Mobile dev?", options=["Yes", "No"], index=1)
        interested_in_ux_design = st.radio("UX design?", options=["Yes", "No"], index=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Waypoint 6: Experience ---
    st.markdown('<div class="section-block">', unsafe_allow_html=True)
    section_header(SECTIONS[5])
    col13, col14 = st.columns(2)
    with col13:
        years_of_experience = st.number_input("Years of experience", min_value=0, max_value=50, value=5, step=1)
        hours_coding = st.number_input("Hours coding / week", min_value=0, max_value=168, value=25, step=1)
        num_side_projects = st.number_input("Side projects", min_value=0, max_value=100, value=2, step=1)
    with col14:
        remote_work = st.selectbox("Remote work preference", options=["Fully Remote", "Hybrid", "In-Office"])
        company_size = st.selectbox("Company size preference", options=["Startup", "Mid-size", "Enterprise", "Corporate"])
        open_source = st.radio("Open source contributor?", options=["Yes", "No"], index=0)

    st.markdown('<div class="profile-caption">Work style</div>', unsafe_allow_html=True)
    col15, col16, col17 = st.columns(3)
    with col15:
        problem_solving_style = st.selectbox("Problem solving (1–5)", options=[1, 2, 3, 4, 5], index=1)
    with col16:
        work_style = st.selectbox("Work style (1–5)", options=[1, 2, 3, 4, 5], index=2)
    with col17:
        learning_preference = st.selectbox("Learning preference (1–5)", options=[1, 2, 3, 4, 5], index=2)

    st.markdown('<div class="profile-caption">Job needs</div>', unsafe_allow_html=True)
    col18, col19, col20 = st.columns(3)
    with col18:
        job_satisfaction = st.slider("Job satisfaction (current)", 1, 10, 4)
    with col19:
        work_life_balance = st.slider("Work-life balance", 1, 10, 3)
    with col20:
        learning_agility = st.slider("Learning agility", 1, 10, 4)
    st.markdown('</div>', unsafe_allow_html=True)

    submit_button = st.form_submit_button("Find my career path →")


if submit_button:
    user_inputs = {
        "Age": age,
        "Gender": gender,
        "Education": education,
        "EducationField": education_field,
        "YearsOfExperience": years_of_experience,
        "CodingSkillRating": coding_skill,
        "MathematicsRating": mathematics_rating,
        "CreativityRating": creativity_rating,
        "SecurityAwarenessRating": security_awareness,
        "SystemsThinkingRating": systems_thinking,
        "DataAnalysisRating": data_analysis,
        "CommunicationRating": communication,
        "DesignSensibilityRating": design_sensibility,
        "InterestedInAI": interested_in_ai,
        "InterestedInWebDev": interested_in_web_dev,
        "InterestedInSecurity": interested_in_security,
        "InterestedInDataScience": interested_in_data_science,
        "InterestedInCloudComputing": interested_in_cloud,
        "InterestedInMobileDev": interested_in_mobile_dev,
        "InterestedInUXDesign": interested_in_ux_design,
        "KnowsPython": knows_python,
        "KnowsJavaScript": knows_javascript,
        "KnowsSQL": knows_sql,
        "KnowsJava": knows_java,
        "KnowsBashLinux": knows_bash_linux,
        "ProblemSolvingStyle": problem_solving_style,
        "WorkStyle": work_style,
        "LearningPreference": learning_preference,
        "JobSatisfactionRating": job_satisfaction,
        "WorkLifeBalanceRating": work_life_balance,
        "LearningAgility": learning_agility,
        "RemoteWorkPreference": remote_work,
        "CompanySizePreference": company_size,
        "HighestCertification": highest_certification,
        "HoursSpentCodingPerWeek": hours_coding,
        "NumSideProjects": num_side_projects,
        "OpenSourceContributor": open_source,
    }

    input_df = pd.DataFrame([user_inputs])

    render_route_map(active_index=len(SECTIONS))  # all waypoints lit — journey complete

    st.markdown('<div class="profile-caption">Your profile</div>', unsafe_allow_html=True)
    st.dataframe(input_df, use_container_width=True)

    if model is None:
        st.error(
            "No model loaded, so there's nothing to predict from yet. "
            "Add model.pkl next to this script and submit again."
        )
    else:
        try:
            prediction = model.predict(input_df)
            predicted_career = prediction[0]

            st.markdown(
                f'<div class="destination-card">'
                f'<div class="destination-eyebrow">YOUR DESTINATION</div>'
                f'<div class="destination-title">{predicted_career}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if hasattr(model, "predict_proba"):
                st.markdown('<div class="profile-caption">Confidence across top matches</div>', unsafe_allow_html=True)
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_
                prob_df = (
                    pd.DataFrame({"Career": classes, "Probability": probabilities})
                    .sort_values(by="Probability", ascending=False)
                    .head(5)
                )
                st.bar_chart(prob_df.set_index("Career"), y_label="Confidence", color="#2DD4BF")

        except Exception as e:
            st.error(f"The model couldn't score this profile: {e}")
            st.markdown(
                "This usually means the model expects preprocessing (encoding categorical "
                "fields, scaling numeric ones) that isn't happening here. If your model.pkl "
                "is a bare estimator rather than a full pipeline, add that preprocessing step "
                "before calling `model.predict`."
            )
