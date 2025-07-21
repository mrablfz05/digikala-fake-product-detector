from preprocessing import load_and_clean_data
from plots import *
from train import train_model
from sklearn.metrics import classification_report, confusion_matrix

def main():
    df = load_and_clean_data()

    # 👇 Visualization
    # plot_nulls(df)
    # plot_price_box(df)
    # plot_scatter_price_rate(df)
    # plot_hist_price(df)
    # plot_fake_distribution(df)
    plot_corr_heatmap(df)
    # plot_rate_log(df)

    # 👇 Training
    print(df["Is_Fake"].value_counts(normalize=True))
    model, X_test, y_test = train_model(df)

    y_pred = model.predict_proba(X_test)[:, 1]
    y_pred_adjusted = (y_pred >= 0.9).astype(int)

    print(classification_report(y_test, y_pred_adjusted))

    cm = confusion_matrix(y_test, y_pred_adjusted)
    plot_confusion(cm)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()