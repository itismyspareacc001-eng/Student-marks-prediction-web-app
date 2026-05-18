# Students Marks Prediction App
# This file contains the main application logic for predicting student marks

# ==============================
# IMPORT LIBRARIES
# ==============================

import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="AI Student Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)


# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("📌 Navigation Menu")

menu = st.sidebar.selectbox(
    "Choose Section",
    [
        "Home",
        "Dataset",
        "Prediction",
        "Analytics",
        "About"
    ]
)


# ==============================
# DATASET
# ==============================

data = {
    "Student": [
        "Rahul",
        "Amit",
        "Priya",
        "Neha",
        "Karan",
        "Anjali",
        "Riya",
        "Vikas"
    ],

    "StudyHours": [1,2,3,4,5,6,7,8],

    "Marks": [20,35,50,60,72,80,90,96]
}


df = pd.DataFrame(data)


# ==============================
# MACHINE LEARNING MODEL
# ==============================

X = df[["StudyHours"]]

y = df["Marks"]


model = LinearRegression()

model.fit(X, y)


# ==============================
# HOME SECTION
# ==============================

if menu == "Home":

    st.title("🎓 AI Student Analytics Dashboard")

    st.write(
        """
        Welcome to the AI-powered Student Analytics System.

        This project demonstrates:
        - Machine Learning
        - Streamlit Deployment
        - Data Visualization
        - AI Prediction Systems
        """
    )

    st.image(
        "https://images.unsplash.com/photo-1522202176988-66273c2fd55f",
        use_container_width=True
    )

    st.subheader("📈 Dashboard Overview")


    # ---------------- COLUMNS ---------------- #

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Students",
            value=len(df)
        )

    with col2:
        st.metric(
            label="Average Marks",
            value=round(df["Marks"].mean(), 2)
        )

    with col3:
        st.metric(
            label="Highest Marks",
            value=df["Marks"].max()
        )


# ==============================
# DATASET SECTION
# ==============================

elif menu == "Dataset":

    st.title("📂 Student Dataset")

    st.dataframe(df)

    st.subheader("Dataset Statistics")

    st.write(df.describe())


# ==============================
# PREDICTION SECTION
# ==============================

elif menu == "Prediction":

    st.title("🤖 AI Marks Predictor")


    # ---------------- FORM ---------------- #

    with st.form("prediction_form"):

        hours = st.number_input(
            "Enter Study Hours",
            min_value=0.0,
            max_value=24.0,
            step=0.5
        )

        submit = st.form_submit_button(
            "Predict Marks"
        )


    # ---------------- PREDICTION ---------------- #

    if submit:

        input_data = pd.DataFrame({
            "StudyHours": [hours]
        })

        prediction = model.predict(input_data)

        predicted_marks = prediction[0]


        # ---------------- RESULT DISPLAY ---------------- #

        st.success(
            f"🎯 Predicted Marks = {predicted_marks:.2f}"
        )


        # ---------------- PERFORMANCE MESSAGE ---------------- #

        if predicted_marks >= 90:

            st.balloons()

            st.success("Excellent Performance Expected!")

        elif predicted_marks >= 70:

            st.info("Good Performance Expected!")

        elif predicted_marks >= 40:

            st.warning("Average Performance")

        else:

            st.error("Needs Improvement")


# ==============================
# ANALYTICS SECTION
# ==============================

elif menu == "Analytics":

    st.title("📊 Data Analytics")


    # ---------------- LINE CHART ---------------- #

    st.subheader("Study Hours vs Marks")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        df["StudyHours"],
        df["Marks"],
    )

    ax.plot(
        df["StudyHours"],
        model.predict(X)
    )

    ax.set_xlabel("Study Hours")

    ax.set_ylabel("Marks")

    ax.set_title("Linear Regression Analysis")

    ax.grid(True)

    st.pyplot(fig)


    # ---------------- BAR CHART ---------------- #

    st.subheader("Student Marks Comparison")

    fig2, ax2 = plt.subplots(figsize=(8,5))

    ax2.bar(
        df["Student"],
        df["Marks"]
    )

    ax2.set_xlabel("Students")

    ax2.set_ylabel("Marks")

    ax2.set_title("Student Performance")

    st.pyplot(fig2)


    # ---------------- HISTOGRAM ---------------- #

    st.subheader("Marks Distribution")

    fig3, ax3 = plt.subplots(figsize=(8,5))

    ax3.hist(
        df["Marks"],
        bins=5
    )

    ax3.set_xlabel("Marks")

    ax3.set_ylabel("Frequency")

    ax3.set_title("Distribution Analysis")

    st.pyplot(fig3)


# ==============================
# ABOUT SECTION
# ==============================

elif menu == "About":

    st.title("ℹ About Project")

    st.write(
        """
        This AI Dashboard Project demonstrates:

        ✅ Machine Learning

        ✅ Linear Regression

        ✅ Streamlit Deployment

        ✅ Data Analytics

        ✅ Visualization

        ✅ Interactive Prediction System
        """
    )

    st.info(
        "Built using Python, Streamlit, Pandas, Matplotlib, and Scikit-learn"
    )


# ==============================
# FOOTER
# ==============================

st.markdown("---")

st.caption("AI Student Analytics Dashboard | Built with Streamlit")
