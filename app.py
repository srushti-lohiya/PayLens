
"""Step 9: PayLens Streamlit demo UI.

Run:
    python -m streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src.service import predict_job
from src.llm import llm_available


st.set_page_config(
    page_title="PayLens | AI Salary Estimator",
    page_icon="💼",
    layout="wide",
)

st.title("💼 PayLens — AI Salary Estimator")

st.caption(
    "Historical 2019 US Glassdoor salary data · "
    "80% conformal prediction interval · "
    "AI explanations grounded in similar job postings"
)

with st.form("job"):
    c1, c2 = st.columns(2)

    title = c1.text_input(
        "Job title",
        value="Senior Data Scientist",
    )

    location = c2.text_input(
        "Location (City, State)",
        value="San Francisco, CA",
    )

    desc = st.text_area(
        "Paste the job description",
        height=220,
        value=(
            "We need 5+ years of Python, SQL and Spark. "
            "Experience with AWS and deep learning. "
            "PhD preferred."
        ),
    )

    c3, c4, c5 = st.columns(3)

    rating = c3.slider(
        "Company rating",
        min_value=1.0,
        max_value=5.0,
        value=3.8,
        step=0.1,
    )

    founded = c4.number_input(
        "Company founded",
        min_value=1800,
        max_value=2020,
        value=2005,
    )

    sector = c5.selectbox(
        "Sector",
        [
            "Information Technology",
            "Biotech & Pharmaceuticals",
            "Business Services",
            "Finance",
            "Health Care",
            "Insurance",
            "Manufacturing",
            "Aerospace & Defense",
            "-1",
        ],
    )

    explain = st.checkbox(
        "Generate AI explanation",
        value=llm_available(),
        disabled=not llm_available(),
        help="Requires an OPENAI_API_KEY in your environment.",
    )

    go = st.form_submit_button(
        "Estimate salary",
        type="primary",
        use_container_width=True,
    )


if go:
    if not title.strip() or not desc.strip():
        st.error("Please enter a job title and job description.")
        st.stop()

    with st.spinner("Analyzing job details and estimating salary..."):
        try:
            result = predict_job(
                {
                    "job_title": title,
                    "job_description": desc,
                    "location": location,
                    "company_rating": rating,
                    "company_founded": int(founded),
                    "sector": sector,
                },
                with_explanation=explain,
            )
        except Exception as exc:
            st.error(f"Unable to generate salary estimate: {exc}")
            st.stop()

    st.divider()
    st.subheader("Estimated annual salary")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Predicted salary",
        f"${result['salary_k']:,.0f}K",
    )

    m2.metric(
        "Lower bound",
        f"${result['low_k']:,.0f}K",
    )

    m3.metric(
        "Upper bound",
        f"${result['high_k']:,.0f}K",
    )

    st.caption(
        "The lower and upper bounds represent an 80% conformal "
        "prediction interval, calibrated on historical data. "
        "They are not guaranteed salary percentiles."
    )

    if result.get("explanation"):
        st.divider()
        st.subheader("Why this salary?")

        # Render as plain text so dollar signs do not trigger LaTeX.
        st.text(result["explanation"])

    st.divider()
    st.subheader("Most similar real job postings")

    similar_jobs = pd.DataFrame(result.get("similar_jobs", []))

    if not similar_jobs.empty:
        similar_jobs = similar_jobs.rename(
            columns={
                "Job Title": "Job title",
                "company": "Company",
                "salary": "Salary ($K)",
                "similarity": "Similarity",
            }
        )

        st.dataframe(
            similar_jobs,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No similar job postings were found.")

    st.caption(
        "PayLens is a portfolio demonstration trained on historical "
        "US salary data. Estimates should not be treated as current "
        "market salary offers."
    )
