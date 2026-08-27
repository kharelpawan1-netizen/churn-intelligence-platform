# pipelines/statistical_analysis.py
# Formal hypothesis tests: which features are statistically associated with churn?

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

df = pd.read_csv("data/processed/telco_churn_clean.csv")


def chi_square_test(df: pd.DataFrame, column: str) -> tuple:
    """
    Tests whether a categorical column and Churn are independent.
    Null hypothesis: the column has NO relationship with churn.
    A small p-value (< 0.05) means we reject that — the relationship is real.
    """
    contingency_table = pd.crosstab(df[column], df["Churn"])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2, p_value


def t_test(df: pd.DataFrame, column: str) -> tuple:
    """
    Tests whether the average of a numeric column differs between
    churned and non-churned customers.
    """
    churned = df[df["Churn"] == 1][column]
    not_churned = df[df["Churn"] == 0][column]
    t_stat, p_value = ttest_ind(churned, not_churned)
    return t_stat, p_value


if __name__ == "__main__":
    print("=== Chi-square tests (categorical features vs Churn) ===")
    categorical_features = ["Contract", "PaymentMethod", "InternetService", "gender"]
    for col in categorical_features:
        chi2, p = chi_square_test(df, col)
        significant = "significant" if p < 0.05 else "NOT significant"
        print(f"{col:20s} chi2={chi2:8.2f}  p={p:.6f}  -> {significant}")

    print("\n=== T-tests (numeric features vs Churn) ===")
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_features:
        t_stat, p = t_test(df, col)
        significant = "significant" if p < 0.05 else "NOT significant"
        print(f"{col:20s} t={t_stat:8.2f}  p={p:.6f}  -> {significant}")