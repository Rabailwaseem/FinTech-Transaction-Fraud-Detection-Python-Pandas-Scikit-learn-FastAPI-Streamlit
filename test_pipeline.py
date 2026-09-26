import pandas as pd

from src.generate_data import generate_transactions


def test_generated_dataset_has_expected_columns():
    df = generate_transactions(n_rows=100, random_state=7)

    expected = {
        "amount",
        "hour",
        "merchant_category",
        "country_mismatch",
        "device_trust",
        "transactions_24h",
        "card_age_days",
        "account_age_days",
        "previous_chargebacks",
        "is_fraud",
    }

    assert expected.issubset(df.columns)
    assert len(df) == 100


def test_generated_dataset_has_both_classes():
    df = generate_transactions(n_rows=1000, random_state=7)

    assert set(df["is_fraud"].unique()) == {0, 1}
