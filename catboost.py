import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("digikala-products.csv")

df["Is_Fake"].unique()

df = df.dropna(subset=["Seller"])
df = df.dropna(subset=["Category2"])

df["min_price_last_month"] = df["min_price_last_month"].replace(0, np.nan)

df['title_fa'] = df['title_fa'].apply(lambda x: x[:15] + '...' if isinstance(x, str) and len(x) > 40 else x)

col = "sub_category"
print(df[col].apply(type).value_counts())
df.columns
df.shape

df["Price"].min()
df["Price"].max()
df["Price"].mean()

df.columns

# All columns are almost clean but "min_price_last_month" have about 75% NaN values but not that bad because we used CatBoost algorithm(good handle text-based and null values)
col = "min_price_last_month"
null_percentage = df[col].isnull().mean() * 100
print(f"{col}: {null_percentage:.2f}% null values")

df.isnull().sum()

df["Rate"].min()
df["Rate"].max()

df["min_price_last_month"].min()
df["min_price_last_month"].mean()
df["min_price_last_month"].max()

df["Brand"].unique()
df["Is_Fake"].unique()
df["sub_category"].unique()

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

df['Rate_per_vote'] = df['Rate'] / (df['Rate_cnt'] + 1e-5)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Rate_cnt", y="Rate_per_vote", alpha=0.5)
plt.title("Rate ratio Rate count")
plt.xlabel("Rates")
plt.ylabel("Ratio")
plt.show()


# -------------- Big Issue -------------------
# df_filtered = df[(df['Rate_cnt'] > 5) & (df['Rate_per_vote'] < 10)]

# plt.figure(figsize=(10,6))
# sns.scatterplot(data=df_filtered, x='Rate_cnt', y='Rate_per_vote', alpha=0.5)
# plt.title("Points to votes ratio (with filter)")
# plt.xlabel("Rate_cnt")
# plt.ylabel("(Rate / Rate_cnt)")
# plt.grid(True)
# plt.show()

# df['Rate_cnt'].hist(bins=100, figsize=(10,5))
# plt.title("توزیع تعداد رأی‌ها (Rate_cnt)")
# plt.xlabel("تعداد رأی‌ها")
# plt.ylabel("تعداد محصولات")
# plt.grid(True)
# plt.show()

numeric_cols = df.select_dtypes(include=['float64', 'int64'])

corr_matrix = numeric_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Solidarity")
plt.show()