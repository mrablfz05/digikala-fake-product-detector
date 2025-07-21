from imports import CatBoostClassifier, train_test_split, classification_report, confusion_matrix

def train_model(df):
    X = df.drop("Is_Fake", axis=1)
    y = df["Is_Fake"]

    cat_features = ["title_fa", "Category1", "Category2", "Brand", "sub_category", "Seller"]
    cat_features = [col for col in cat_features if col in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostClassifier(
        iterations=750,
        learning_rate=0.2,
        depth=7,
        verbose=100,
        l2_leaf_reg=5,
        border_count=128,
        early_stopping_rounds=80,
        random_seed=42,
        class_weights=[1, 5],
        cat_features=cat_features
    )

    model.fit(X_train, y_train)
    return model, X_test, y_test