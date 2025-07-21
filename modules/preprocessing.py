from imports import pd, np

def load_and_clean_data(path="../data/digikala-products.csv"):
    df = pd.read_csv(path)
    
    df = df.dropna(subset=["Seller", "Category2"])
    df["min_price_last_month"] = df["min_price_last_month"].replace(0, np.nan)

    df['title_fa'] = df['title_fa'].apply(lambda x: x[:15] + '...' if isinstance(x, str) and len(x) > 40 else x)

    df['Rate_per_vote'] = df['Rate'] / (df['Rate_cnt'] + 1)
    df['Price_delta'] = df["Price"] - df["min_price_last_month"]
    df['is_Rate_Zero'] = (df['Rate'] == 0).astype(int)
    df['is_Rate_cnt_Zero'] = (df['Rate_cnt'] == 0).astype(int)

    df.drop(columns=["min_price_last_month"], inplace=True)

    return df
