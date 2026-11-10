import os
import sys
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# Find the main project folder
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from sentiment import SentimentAnalyzer


def main():

    print("\n========================================")
    print("       SENTIMENT MODEL EVALUATION")
    print("========================================\n")

    # Load test dataset
    data_path = os.path.join(
        os.path.dirname(__file__),
        "test_data.csv"
    )

    df = pd.read_csv(data_path)

    print("Test samples:", len(df))

    # Load sentiment analyzer
    analyzer = SentimentAnalyzer()

    y_true = []
    y_pred = []

    print("\nRunning predictions...\n")

    for index, row in df.iterrows():

        text = row["text"]
        actual = row["label"]

        result = analyzer.analyze(text)

        predicted = result["sentiment"]

        y_true.append(actual)
        y_pred.append(predicted)

        print(
            f"{index + 1:02d}. "
            f"Actual: {actual:<8} "
            f"Predicted: {predicted:<8}"
        )

    # Calculate evaluation metrics
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Display results
    print("\n========================================")
    print("              RESULTS")
    print("========================================")

    print(f"\nAccuracy :  {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall   :  {recall:.4f}")
    print(f"F1-Score :  {f1:.4f}")

    # Classification report
    print("\n========================================")
    print("       CLASSIFICATION REPORT")
    print("========================================\n")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[
                "Positive",
                "Neutral",
                "Negative"
            ],
            zero_division=0
        )
    )

    # Confusion matrix
    labels = [
        "Positive",
        "Neutral",
        "Negative"
    ]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print("Confusion Matrix:")
    print(cm)

    # Create confusion matrix visualization
    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(
        "Sentiment Classification Confusion Matrix"
    )

    plt.tight_layout()

    # Save confusion matrix
    output_path = os.path.join(
        os.path.dirname(__file__),
        "confusion_matrix.png"
    )

    plt.savefig(output_path)

    print(
        f"\nConfusion matrix saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()