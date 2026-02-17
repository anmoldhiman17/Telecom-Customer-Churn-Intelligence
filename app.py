import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Telco Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# LOAD MODEL (Logistic Regression Pipeline)
# ----------------------------
model = joblib.load("telco_model.pkl")

# ----------------------------
# CUSTOM CSS (Clean Enterprise Dark Theme)
# ----------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg,#7b2ff7,#f107a3);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 220px;
    font-weight: bold;
    border: none;
}
.stDownloadButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 220px;
    font-weight: bold;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("⚙ Control Panel")

tenure = st.sidebar.slider("📅 Tenure (Months)", 0, 120, 12)
monthly = st.sidebar.slider("💰 Monthly Charges", 0, 500, 70)
total = st.sidebar.slider("🧾 Total Charges", 0, 20000, 800)

st.sidebar.markdown("---")
st.sidebar.info("""
### 📌 About Project

AI-powered churn prediction system with:

- ROC-AUC Optimization  
- Hyperparameter Tuning  
- Threshold Control  
- Production Deployment  
                
Built by Anmol | 2026
""")

# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
<h1 style='text-align:center;'>📊 Telco Customer Churn Intelligence</h1>
<p style='text-align:center;'>Enterprise AI System for Customer Retention Analytics</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# PREDICTION SECTION
# ----------------------------
st.subheader("📌 Customer Input")

if st.button("🚀 Predict Churn"):

    input_data = np.array([[tenure, monthly, total]])

    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= 0.4 else 0

    st.markdown("---")

    # RESULT MESSAGE
    if prediction == 1:
        st.error("⚠ Customer is LIKELY to churn")
    else:
        st.success("✅ Customer is NOT likely to churn")

    st.markdown(f"### 📊 Churn Probability: {probability*100:.2f}%")

    # ----------------------------
    # RISK CATEGORY BADGE
    # ----------------------------
    if probability < 0.4:
        st.success("🟢 Risk Level: LOW")
        risk_label = "LOW"
    elif probability < 0.7:
        st.warning("🟡 Risk Level: MEDIUM")
        risk_label = "MEDIUM"
    else:
        st.error("🔴 Risk Level: HIGH")
        risk_label = "HIGH"

    # ----------------------------
    # GAUGE + MINI ANALYTICS
    # ----------------------------
    col1, col2 = st.columns([2,1])

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Churn Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Mini Analytics")

        metrics = pd.DataFrame({
            "Metric": ["Model Accuracy", "ROC-AUC", "Precision", "Recall"],
            "Value": [0.84, 0.83, 0.80, 0.75]
        })

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True
        )

    # ----------------------------
    # PDF REPORT GENERATION
    # ----------------------------
    def generate_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []

        styles = getSampleStyleSheet()

        elements.append(Paragraph("Telco Customer Churn Report", styles["Heading1"]))
        elements.append(Spacer(1, 20))

        content = f"""
        Customer Tenure: {tenure} months <br/>
        Monthly Charges: ₹{monthly} <br/>
        Total Charges: ₹{total} <br/><br/>
        Churn Probability: {probability*100:.2f}% <br/>
        Risk Level: {risk_label}
        """

        elements.append(Paragraph(content, styles["Normal"]))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf_file = generate_pdf()

    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_file,
        file_name="churn_report.pdf",
        mime="application/pdf"
    )

st.markdown("---")
st.markdown(
    "<center>Built with ❤️ by Anmol | Telco ML Project 2026</center>",
    unsafe_allow_html=True
)
