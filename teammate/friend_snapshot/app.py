"""app.py - live demo for the viva. Run with:  streamlit run app.py"""
import pandas as pd
import streamlit as st
from pipeline import RiskPipeline

AGE = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64",
       "65-69", "70-74", "75-79", "80+"]
INCOME = ["< $10k", "$10-15k", "$15-20k", "$20-25k", "$25-35k", "$35-50k", "$50-75k", "$75k+"]
EDU = ["Never attended", "Grades 1-8", "Grades 9-11", "Grade 12 / GED", "Some college", "College graduate"]
HEALTH = ["Excellent", "Very good", "Good", "Fair", "Poor"]
ACTION = {"Minimal": "Routine care. Re-screen in 3 years.",
          "Watch": "Lifestyle advice; re-screen in 1 year.",
          "Elevated": "Offer a blood test (HbA1c or fasting glucose) at the next visit.",
          "Priority": "Prioritise a blood test now."}
BASE = dict(Age=6, Sex=0, BMI=27, HighBP=0, HighChol=0, CholCheck=1, Smoker=0, Stroke=0,
            HeartDiseaseorAttack=0, PhysActivity=1, Fruits=1, Veggies=1, HvyAlcoholConsump=0,
            AnyHealthcare=1, NoDocbcCost=0, GenHlth=2, MentHlth=0, PhysHlth=0, DiffWalk=0,
            Education=5, Income=6)
PERSONAS = {"Young, healthy weight": {**BASE, "Age": 2, "BMI": 22},
            "Middle-aged, model unsure": {**BASE, "Age": 8, "BMI": 23, "HighChol": 1, "GenHlth": 3},
            "Senior, obese, inactive": {**BASE, "Age": 11, "BMI": 36, "HighBP": 1, "HighChol": 1,
                                        "PhysActivity": 0, "Fruits": 0, "Veggies": 0, "GenHlth": 4}}

st.set_page_config(page_title="Diabetes screening triage", layout="centered")
pipe = st.cache_resource(RiskPipeline)()
st.title("Non-invasive diabetes screening triage")
st.caption("Research demo, not a diagnosis. It suggests who should be offered a blood test first. "
           "Trained on self-reported BRFSS 2015 survey answers.")
choice = st.sidebar.radio("Start from a persona", ["Enter my own"] + list(PERSONAS))
d = PERSONAS.get(choice, BASE)
yes = lambda label, key: int(st.checkbox(label, value=bool(d[key])))

with st.form("answers"):
    c1, c2 = st.columns(2)
    a = dict(d)
    a["Age"] = c1.selectbox("Age", range(1, 14), index=d["Age"] - 1, format_func=lambda i: AGE[i - 1])
    a["Sex"] = c2.radio("Sex", [0, 1], index=d["Sex"], format_func=lambda i: ["Female", "Male"][i])
    a["BMI"] = c1.number_input("BMI", 12, 98, d["BMI"])
    a["GenHlth"] = c2.selectbox("General health", range(1, 6), index=d["GenHlth"] - 1,
                                format_func=lambda i: HEALTH[i - 1])
    a["Income"] = c1.selectbox("Household income", range(1, 9), index=d["Income"] - 1,
                               format_func=lambda i: INCOME[i - 1])
    a["Education"] = c2.selectbox("Education", range(1, 7), index=d["Education"] - 1,
                                  format_func=lambda i: EDU[i - 1])
    a["MentHlth"] = c1.slider("Days of poor mental health (last 30)", 0, 30, d["MentHlth"])
    a["PhysHlth"] = c2.slider("Days of poor physical health (last 30)", 0, 30, d["PhysHlth"])
    for key, label in [("HighBP", "Told you have high blood pressure"), ("HighChol", "Told you have high cholesterol"),
                       ("CholCheck", "Cholesterol checked in the last 5 years"), ("Smoker", "Smoked 100+ cigarettes ever"),
                       ("Stroke", "Ever had a stroke"), ("HeartDiseaseorAttack", "Heart disease or heart attack"),
                       ("PhysActivity", "Physical activity outside work (last 30 days)"),
                       ("Fruits", "Fruit at least once a day"), ("Veggies", "Vegetables at least once a day"),
                       ("HvyAlcoholConsump", "Heavy drinking"), ("AnyHealthcare", "Any health coverage"),
                       ("NoDocbcCost", "Skipped a doctor because of cost (last year)"),
                       ("DiffWalk", "Serious difficulty walking or climbing stairs")]:
        a[key] = yes(label, key)
    go = st.form_submit_button("Assess")

if go:
    r = pipe.assess(pd.DataFrame([a])).iloc[0]
    st.metric("Risk tier", r.tier)
    st.write(f"Model estimate: about **{round(100 * r.p_diabetes)} in 100** people who answered like this "
             f"reported a diabetes or prediabetes diagnosis.")
    st.info(ACTION[r.tier])
    st.subheader("Why this tier")
    for line in r.trace.split(" | "):
        st.write("- " + line)
    st.subheader("What could change")
    if r.largest_modifiable != "none":
        st.write(f"Of a risk score of {r.risk_score:.0f}/100, about {r.modifiable_part:.0f} points are linked to "
                 f"modifiable factors; the largest is **{r.largest_modifiable}**.")
    else:
        st.write("No modifiable factor changes the score noticeably.")
    st.caption("Associations in cross-sectional survey data, not proven effects of changing behaviour.")
