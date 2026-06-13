import pandas as pd

df = pd.read_csv("telco_churn_processed.csv")

print("Shape:", df.shape)
print("\nKolom:")
print(df.columns.tolist())