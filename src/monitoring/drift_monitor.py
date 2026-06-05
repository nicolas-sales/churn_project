import os
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

REFERENCE_DATA_PATH = "data/Telco-Customer-Churn.csv"
CURRENT_DATA_PATH = "monitoring/predictions.csv"
REPORT_PATH = "monitoring/drift_report.html"

def load_reference_data():

    df = pd.read_csv(
        REFERENCE_DATA_PATH
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.dropna()

    df = df.drop(
        columns=[
            "customerID",
            "Churn"
        ],
        errors="ignore"
    )

    return df

def load_current_data():

    df = pd.read_csv(CURRENT_DATA_PATH)

    # Colonnes ajoutées par le monitoring, non présentes dans le dataset de référence

    df = df.drop(columns=["timestamp","churn_probability","prediction","risk_level"],errors="ignore")

    return df

def generate_drift_report():

    reference_df = load_reference_data()
    current_df = load_current_data()

    # Garder uniquement les colonnes communes

    common_columns = [
        col for col in reference_df.columns if col in current_df.columns
    ]

    reference_df = reference_df[common_columns]
    current_df = current_df[common_columns]

    report = Report([
        DataDriftPreset()
    ])

    result = report.run(
        current_df,reference_df
    )

    os.makedirs("monitoring", exist_ok=True)

    result.save_html(REPORT_PATH)

    print(f"Drift report saved to {REPORT_PATH}")


if __name__=="__main__":

    generate_drift_report()