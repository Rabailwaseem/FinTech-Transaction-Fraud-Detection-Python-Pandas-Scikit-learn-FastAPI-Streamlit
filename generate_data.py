"""Generate a reproducible synthetic fintech transaction dataset."""

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_ROWS = 15000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_transactions(n_rows=N_ROWS, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)

    amount = np.round(rng.lognormal(mean=3.4, sigma=1.0, size=n_rows), 2)
    hour = rng.integers(0, 24, size=n_rows)

    merchant_categories = np.array(
        ["grocery", "fuel", "restaurant", "electronics", "travel", "fashion", "online_services"]
    )
    merchant_category = rng.choice(
        merchant_categories,
        size=n_rows,
        p=[0.18, 0.12, 0.17, 0.12, 0.08, 0.15, 0.18],
    )

    country_mismatch = rng.binomial(1, 0.08, size=n_rows)
    device_trust = np.clip(rng.beta(5, 2, size=n_rows), 0, 1)
    transactions_24h = rng.poisson(2.2, size=n_rows)
    card_age_days = rng.integers(15, 3000, size=n_rows)
    account_age_days = rng.integers(30, 5000, size=n_rows)
    previous_chargebacks = rng.poisson(0.12, size=n_rows)

    # Create a realistic-ish synthetic fraud signal.
    unusual_hour = ((hour <= 4) | (hour >= 23)).astype(int)
    high_amount = (amount > np.quantile(amount, 0.90)).astype(int)
    high_velocity = (transactions_24h >= 7).astype(int)
    low_trust = (device_trust < 0.35).astype(int)
    risky_merchant = np.isin(merchant_category, ["electronics", "travel", "online_services"]).astype(int)

    logit = (
        -4.2
        + 0.95 * unusual_hour
        + 1.15 * high_amount
        + 1.00 * high_velocity
        + 1.25 * country_mismatch
        + 1.40 * low_trust
        + 0.55 * risky_merchant
        + 0.60 * np.minimum(previous_chargebacks, 3)
        - 0.00008 * card_age_days
        + rng.normal(0, 0.55, size=n_rows)
    )

    fraud_probability = sigmoid(logit)
    is_fraud = rng.binomial(1, fraud_probability)

    df = pd.DataFrame(
        {
            "amount": amount,
            "hour": hour,
            "merchant_category": merchant_category,
            "country_mismatch": country_mismatch,
            "device_trust": np.round(device_trust, 4),
            "transactions_24h": transactions_24h,
            "card_age_days": card_age_days,
            "account_age_days": account_age_days,
            "previous_chargebacks": previous_chargebacks,
            "is_fraud": is_fraud,
        }
    )
    return df


if __name__ == "__main__":
    output = Path("data/transactions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_transactions()
    df.to_csv(output, index=False)
    print(f"Created {output} with shape {df.shape}")
    print(df["is_fraud"].value_counts(normalize=True).rename("proportion"))
