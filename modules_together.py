import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/digikala-products.csv")

# Check if all data is loaded
print(f"Total rows loaded: {len(df)}")

df = df.dropna(subset=["Seller", "Category2"])

df["min_price_last_month"] = df["min_price_last_month"].replace(0, np.nan)

df['title_fa'] = df['title_fa'].apply(lambda x: x[:15] + '...' if isinstance(x, str) and len(x) > 40 else x)

col = "sub_category"
print(df[col].apply(type).value_counts())

col = "min_price_last_month"
null_percentage = df[col].isnull().mean() * 100
print(f"{col}: {null_percentage:.2f}% null values")

df.isnull().mean().sort_values(ascending=False).plot(kind="bar", figsize=(12, 5))
plt.title("Null values")
plt.ylabel("Between 0 and 1")
plt.grid(True)
plt.show()

sns.boxplot(data=df[["Price", "min_price_last_month"]], color="red")
plt.title("Boxplot of Price and min_price_last_month")
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
plt.title("Correlation Matrix")
plt.show()

print(df[["Price", "min_price_last_month"]].corr())
print(df["Rate_per_vote"].describe())

df["Price_delta"] = df["Price"] - df["min_price_last_month"]

df.drop(columns=["min_price_last_month"], inplace=True)

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

print(df["Is_Fake"].value_counts(normalize=True))

cat_features = ["title_fa", "Category1", "Category2", "Brand", "sub_category", "Seller"]
cat_features = [col for col in cat_features if col in df.columns]

# Split based on IDs to avoid leakage and keep all data
train_ids = set(df.sample(frac=0.8, random_state=32)["id"])
X_train = df[df["id"].isin(train_ids)].drop(columns=["Is_Fake"])
X_test = df[~df["id"].isin(train_ids)].drop(columns=["Is_Fake"])
y_train = df[df["id"].isin(train_ids)]["Is_Fake"]
y_test = df[~df["id"].isin(train_ids)]["Is_Fake"]

common_ids = set(X_train["id"]).intersection(set(X_test["id"]))
print("common id's after fix:", len(common_ids))
print(f"new olverlap: {(len(common_ids) / len(X_train)) * 100:.2f}%")

# Use class_weights to handle imbalance
model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.1,
    depth=4,
    l2_leaf_reg=5,
    early_stopping_rounds=50,
    class_weights=[1, 20],  # Increased for better balance
    random_seed=32,
    border_count=128,
    verbose=100,
    cat_features=cat_features
)
model.fit(X_train, y_train, eval_set=(X_test, y_test))

y_pred = model.predict_proba(X_test)[:, 1]
threshold = 0.3  # Adjusted to improve Precision and F1 for True
y_pred_adjusted = (y_pred >= threshold).astype(int)
print("گزارش طبقه‌بندی:")
print(classification_report(y_test, y_pred_adjusted))

cm = confusion_matrix(y_test, y_pred_adjusted)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0.5, 1.5], ['Real', 'Fake'])
plt.yticks([0.5, 1.5], ['Real', 'Fake'])
plt.show()
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(X_train['Price'], kde=True, color='blue', label='Train')
sns.histplot(X_test['Price'], kde=True, color='red', label='Test')
plt.title('distribution Price')
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(X_train['Rate'], kde=True, color='blue', label='Train')
sns.histplot(X_test['Rate'], kde=True, color='red', label='Test')
plt.title('distribution Rate')
plt.legend()
plt.show()