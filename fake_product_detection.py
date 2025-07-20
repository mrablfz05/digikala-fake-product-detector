import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("data/digikala-products.csv")

df = df.dropna(subset=["Seller"])
df = df.dropna(subset=["Category2"])

df["min_price_last_month"] = df["min_price_last_month"].replace(0, np.nan)

df['title_fa'] = df['title_fa'].apply(lambda x: x[:15] + '...' if isinstance(x, str) and len(x) > 40 else x)

col = "sub_category"
print(df[col].apply(type).value_counts())

# All columns are almost clean but "min_price_last_month" have about 75% NaN values but not that bad because we used CatBoost algorithm(good handle text-based and null values)
col = "min_price_last_month"
null_percentage = df[col].isnull().mean() * 100
print(f"{col}: {null_percentage:.2f}% null values")

df.isnull().mean().sort_values(ascending=False).plot(kind="bar", figsize=(12, 5))
plt.title("Null values")
plt.ylabel("Between 0 and 1")
plt.grid(True)
plt.show()

sns.boxplot(data=df[["Price", "min_price_last_month"]], color="red")
plt.title("random")
plt.show()

sns.scatterplot(x="Price", y="Rate", hue="Is_Fake", data=df)
plt.title("Price to Rate for showing is fake or not")
plt.show()

df["min_price_last_month"].hist(bins=100)
plt.title("Scattering min price for last month")
plt.xlabel("Price")
plt.ylabel("Quantity")
plt.show()

sns.countplot(x='Is_Fake', data=df)
plt.title("Production of real and fake products")
plt.show()

df['Rate_per_vote'] = df['Rate'] / (df['Rate_cnt'] + 1)

numeric_cols = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Solidarity")
plt.show()

df[["Price", "min_price_last_month"]].corr()
df["Rate_per_vote"]

# Feature Engineering
df["Price_delta"] = df["Price"] - df["min_price_last_month"]

df.drop(columns=["min_price_last_month"],inplace=True)

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", linewidths="1", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

df[(df["Rate_cnt"] == 0) & (df["Rate"] != 0)]

print(df[(df['Rate_cnt'] == 0)][['Rate', 'Rate_cnt']].head(50))

print(f"non-zero Rate_cnt: {(len(df[df['Rate_cnt'] > 0]) / len(df)) * 100:.2f}%")

df['is_Rate_Zero'] = (df['Rate'] == 0).astype(int)
df['is_Rate_cnt_Zero'] = (df['Rate_cnt'] == 0).astype(int)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Rate_cnt", y="Rate", alpha=0.5)
plt.xscale('log')
plt.title("Rate vs Rate_cnt (Log Scale)")
plt.xlabel("Rate_cnt (Log)")
plt.ylabel("Rate")
plt.show()

# Training
print(df["Is_Fake"].value_counts(normalize=True))

result_file = "./logs/text/classification_comparison.csv"
run_history = []

X = df.drop("Is_Fake", axis=1)
y = df["Is_Fake"]

cat_features = ["title_fa", "Category1", "Category2", "Brand", "sub_category", "Seller"]
cat_features = [col for col in cat_features if col in X.columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if os.path.exists(result_file):
    run_history = pd.read_csv(result_file).to_dict('records')

run_id = len(run_history) + 1

model = CatBoostClassifier(
    iterations= 800,
    learning_rate= 0.2,
    depth= 7,
    verbose= 100,
    l2_leaf_reg= 5,
    border_count= 128,
    early_stopping_rounds= 50,
    random_seed=42,
    class_weights= [1, 5],
    # auto_class_weights= "Balanced",
    cat_features= cat_features
)

model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)[:, 1]
threshold = 0.9
y_pred_adjusted = (y_pred >= threshold).astype(int)
report = classification_report(y_test, y_pred_adjusted, output_dict=True)

current_run = {
    "Run": run_id,
    "Iterations": 800,
    "Depth": 7,
    "l2_leaf_reg": 5,
    "Threshold": 0.9,
    "Precision_False": report['False']['precision'],
    "Recall_False": report['False']['recall'],
    "F1_False": report['False']['f1-score'],
    "Precision_True": report['True']['precision'],
    "Recall_True": report['True']['recall'],
    "F1_True": report['True']['f1-score'],
    "Accuracy": report['accuracy']
}
run_history.append(current_run)

with open(result_file, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=current_run.keys())
    if run_id == 1:
        writer.writeheader()
    writer.writerows(run_history)
comparison_df = pd.DataFrame(run_history)
print(comparison_df.to_string(index=False))