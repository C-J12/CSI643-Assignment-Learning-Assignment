"""Entry point for the CSI643 student performance ML assignment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loading import load_student_data, inspect_data
from classification import run_classification
from regression import run_regression
from evaluation import plot_eda, plot_best_r1, save_results


def main():
    output_dir = ROOT / "results"
    df = load_student_data()
    inspect_data(df)
    print("\nAverage final grade:", round(df["G3"].mean(), 2))
    print("Highest final grade:", df["G3"].max())
    print("Lowest final grade:", df["G3"].min())
    plot_eda(df, output_dir)
    df["Pass"] = (df["G3"] >= 10).astype(int)
    print("\nPass/fail counts:\n", df["Pass"].value_counts().sort_index())
    print("\nPass/fail percentages:\n", (df["Pass"].value_counts(normalize=True) * 100).round(2))
    classification_table, best_classification_models = run_classification(df)
    regression_table, best_regression_models, test_sets = run_regression(df)
    plot_best_r1(regression_table, best_regression_models, test_sets, output_dir)
    save_results(classification_table, regression_table, output_dir)
    print("\nFinished. Plots and CSV result tables are in the results/ folder.")


if __name__ == "__main__":
    main()
