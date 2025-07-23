import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

cat_features = ["title_fa", "Category1", "Category2", "Brand", "sub_category", "Seller"]
cat_features = [col for col in cat_features if col in df.columns]

# solve data leakage
train_ids = set(df.sample(frac=0.8, random_state=42)["id"])
X_train = df[df["id"].isin(train_ids)].drop("Is_Fake", axis=1)
X_test = df[~df["id"].isin(train_ids)].drop("Is_Fake", axis=1)
y_train = df[df["id"].isin(train_ids)]["Is_Fake"]
y_test = df[~df["id"].isin(train_ids)]["Is_Fake"]

common_ids = set(X_train["id"]).intersection(set(X_test["id"]))
print("common id's after fix:", len(common_ids))
print(f"new olverlap: {(len(common_ids) / len(X_train)) * 100:.2f}%")


def objective(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 550, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2, 10),
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 50, 300),
        'class_weight_true': trial.suggest_int('class_weight_true', 3, 15),
        'random_seed': 42,
        'border_count': 128,
        'verbose': 100,
        'cat_features': cat_features
    }

    class_weights = [1, param['class_weight_true']]
    del param['class_weight_true']

    model = CatBoostClassifier(**param)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    threshold = trial.suggest_float('threshold', 0.5, 0.99)
    y_pred_adjusted = (y_pred >= threshold).astype(int)

    report = classification_report(y_test, y_pred_adjusted, output_dict=True)
    f1_true = report['True']['f1-score']

    return f1_true

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print("best parameters: ", study.best_params)

best_params = study.best_params
class_weights = [1, best_params['class_weight_true']]
del best_params['class_weight_true']

model = CatBoostClassifier(
    iterations=best_params['iterations'],
    learning_rate=best_params['learning_rate'],
    depth=best_params['depth'],
    l2_leaf_reg=best_params['l2_leaf_reg'],
    early_stopping_rounds=best_params['early_stopping_rounds'],
    border_count=128,
    verbose=100,
    random_seed=42,
    class_weights=class_weights,
    cat_features=cat_features
)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)[:, 1]
threshold = best_params['threshold']
y_pred_adjusted = (y_pred >= threshold).astype(int)
print("گزارش طبقه‌بندی با بهترین پارامترها:")
print(classification_report(y_test, y_pred_adjusted))

cm = confusion_matrix(y_test, y_pred_adjusted)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1, linecolor='black', annot_kws={"size": 16})
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.xticks([0.5, 1.5], ['Real', 'Fake'], fontsize=12)
plt.yticks([0.5, 1.5], ['Real', 'Fake'], fontsize=12, rotation=0)
plt.tight_layout()
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