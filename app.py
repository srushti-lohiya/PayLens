"""Step 9: demo UI.  Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st
from src.service import predict_job
from src.llm import llm_available

st.set_page_config(page_title="Salary AI", page_icon="💼", layout="wide")
st.title("💼 Data Science Salary Estimator")
st.caption("Trained on 2019 US Glassdoor estimates · 80% conformal range · grounded in similar real postings")

with st.form("job"):
    c1, c2 = st.columns(2)
    title = c1.text_input("Job title", "Senior Data Scientist")
    location = c2.text_input("Location (City, ST)", "San Francisco, CA")
    desc = st.text_area("Paste the job description", height=220,
                        value="We need 5+ years of Python, SQL and Spark. Experience with AWS and deep learning. PhD preferred.")
    c3, c4, c5 = st.columns(3)
    rating = c3.slider("Company rating", 1.0, 5.0, 3.8, 0.1)
    founded = c4.number_input("Company founded", 1800, 2020, 2005)
    sector = c5.selectbox("Sector", ["Information Technology", "Biotech & Pharmaceuticals", "Business Services",
                                     "Finance", "Health Care", "Insurance", "Manufacturing", "Aerospace & Defense", "-1"])
    explain = st.checkbox("Explain with LLM", value=llm_available(), disabled=not llm_available(),
                          help="Needs ANTHROPIC_API_KEY")
    go = st.form_submit_button("Estimate salary")

if go:
    with st.spinner("Estimating..."):
        r = predict_job({"job_title": title, "job_description": desc, "location": location,
                         "company_rating": rating, "company_founded": int(founded), "sector": sector},
                        with_explanation=explain)
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimate", f"${r['salary_k']:.0f}K")
    m2.metric("Low (P10)", f"${r['low_k']:.0f}K")
    m3.metric("High (P90)", f"${r['high_k']:.0f}K")
    if "explanation" in r:
        st.subheader("Why"); st.write(r["explanation"])
    st.subheader("Most similar real postings")
    st.dataframe(pd.DataFrame(r["similar_jobs"]), use_container_width=True)
