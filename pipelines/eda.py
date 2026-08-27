# pipelines/eda.py
# Exploratory analysis on the cleaned churn data: what drives churn?

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/telco_churn_clean.csv")

# --- Churn rate by contract type ---
# Business question: are month-to-month customers more likely to leave
# than customers on longer contracts?
churn_by_contract = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
print("Churn rate by contract type:")
print(churn_by_contract)

# --- Churn rate by tenure ---
# Business question: do newer customers churn more than long-term ones?
print(f"\nAvg tenure, churned customers:     {df[df['Churn'] == 1]['tenure'].mean():.1f} months")
print(f"Avg tenure, non-churned customers: {df[df['Churn'] == 0]['tenure'].mean():.1f} months")

# --- Save a simple visualization ---
plt.figure(figsize=(8, 5))
sns.barplot(x=churn_by_contract.index, y=churn_by_contract.values)
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate")
plt.xlabel("Contract Type")
plt.tight_layout()
plt.savefig("docs/churn_by_contract.png")
print("\nSaved chart to docs/churn_by_contract.png")