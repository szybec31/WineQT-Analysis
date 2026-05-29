# src/results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_results_tables(results, save_dir="charts/"):

    results_df = pd.DataFrame(results)

    print("\n==============================")
    print(" RESULTS TABLES ")
    print("==============================")

    resampling_methods = results_df["resampling"].unique()

    for method in resampling_methods:

        print(f"\n===== {method.upper()} =====")

        table = results_df[
            results_df["resampling"] == method
        ].copy()

        table = table.sort_values(
            by="f1_mean",
            ascending=False
        )

        table = table.reset_index(drop=True)

        # USUŃ kolumny techniczne
        display_table = table.drop(
            columns=["f1_mean", "bal_acc_mean"],
            errors="ignore"
        )

        print(display_table)

        save_path = f"{save_dir}{method}_results.csv"

        display_table.to_csv(save_path, index=False)

        print(f"\nSaved: {save_path}")

    return results_df


def plot_results(results_df, metric="f1_mean"):

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="resampling",
        y=metric,
        hue="model"
    )

    plt.title(f"Models comparison (f1_macro)")
    plt.grid(True)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"charts/is_resampling_better.png")
    plt.show()