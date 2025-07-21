from imports import plt, sns

def plot_nulls(df):
    df.isnull().mean().sort_values(ascending=False).plot(kind="bar", figsize=(12, 5))
    plt.title("Null values")
    plt.ylabel("Between 0 and 1")
    plt.grid(True)
    plt.show()

def plot_price_box(df):
    sns.boxplot(data=df[["Price", "Price_delta"]], color="red")
    plt.title("Price vs Price_delta")
    plt.show()

def plot_scatter_price_rate(df):
    sns.scatterplot(x="Price", y="Rate", hue="Is_Fake", data=df)
    plt.title("Price to Rate: Fake vs Real")
    plt.show()

def plot_hist_price(df):
    df["Price_delta"].hist(bins=100)
    plt.title("Price Delta Distribution")
    plt.xlabel("Price Delta")
    plt.ylabel("Count")
    plt.show()

def plot_fake_distribution(df):
    sns.countplot(x='Is_Fake', data=df)
    plt.title("Real vs Fake Product Count")
    plt.show()

def plot_corr_heatmap(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_cols.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

def plot_rate_log(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Rate_cnt", y="Rate", alpha=0.5)
    plt.xscale('log')
    plt.title("Rate vs Rate_cnt (Log Scale)")
    plt.xlabel("Rate_cnt (Log)")
    plt.ylabel("Rate")
    plt.show()

def plot_confusion(cm):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1, linecolor='black', annot_kws={"size": 16})
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks([0.5, 1.5], ['Real', 'Fake'], fontsize=12)
    plt.yticks([0.5, 1.5], ['Real', 'Fake'], fontsize=12, rotation=0)
    plt.tight_layout()
    plt.show()
