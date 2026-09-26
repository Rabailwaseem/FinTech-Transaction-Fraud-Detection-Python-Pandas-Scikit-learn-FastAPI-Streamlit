"""Streamlit UI for the fraud detection model."""

import streamlit as st

from src.predict import predict_transaction

st.set_page_config(
    page_title="FinTech Fraud Detection",
    page_icon="🔎",
    layout="centered",
)

st.title("FinTech Transaction Fraud Detection")
st.caption("Synthetic-data ML portfolio project")

st.write(
    "Enter transaction and account signals to estimate the probability "
    "that a transaction may be fraudulent."
)

with st.form("transaction_form"):
    amount = st.number_input("Transaction amount", min_value=0.01, value=150.0)
    hour = st.slider("Transaction hour", min_value=0, max_value=23, value=14)
    merchant_category = st.selectbox(
        "Merchant category",
        ["grocery", "fuel", "restaurant", "electronics", "travel", "fashion", "online_services"],
    )
    country_mismatch = st.selectbox("Country mismatch", [0, 1])
    device_trust = st.slider("Device trust", 0.0, 1.0, 0.80)
    transactions_24h = st.number_input(
        "Transactions in previous 24h", min_value=0, value=2
    )
    card_age_days = st.number_input("Card age (days)", min_value=1, value=900)
    account_age_days = st.number_input("Account age (days)", min_value=1, value=1500)
    previous_chargebacks = st.number_input(
        "Previous chargebacks", min_value=0, value=0
    )

    submitted = st.form_submit_button("Analyze Transaction")

if submitted:
    transaction = {
        "amount": amount,
        "hour": hour,
        "merchant_category": merchant_category,
        "country_mismatch": country_mismatch,
        "device_trust": device_trust,
        "transactions_24h": transactions_24h,
        "card_age_days": card_age_days,
        "account_age_days": account_age_days,
        "previous_chargebacks": previous_chargebacks,
    }

    try:
        result = predict_transaction(transaction)
        probability = result["fraud_probability"]

        st.metric("Fraud probability", f"{probability:.1%}")
        st.info(f"Risk classification: **{result['risk']}**")
        st.progress(probability)
    except FileNotFoundError as exc:
        st.error(str(exc))
