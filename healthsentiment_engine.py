import re

import html

import time

from pathlib import Path



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from textblob import TextBlob

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



from sklearn.model_selection import (

  train_test_split,

  StratifiedKFold

)

from sklearn.metrics import (

  accuracy_score,

  precision_score,

  recall_score,

  f1_score,

  classification_report,

  confusion_matrix,

  ConfusionMatrixDisplay

)



# ==========================================================

# HEALTHSENTIMENT AI

# Comparative Healthcare Sentiment Analysis Agent

# ==========================================================



OUTPUT_FOLDER = Path.cwd() / "healthsentiment_results"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)



MAX_DRUG_REVIEWS = 5000

RANDOM_STATE = 42



THRESHOLDS = {

  "TextBlob": np.round(np.arange(-0.30, 0.301, 0.01), 3),

  "VADER": np.round(np.arange(-0.50, 0.501, 0.01), 3)

}



DEFAULT_THRESHOLDS = {

  "TextBlob": 0.00,

  "VADER": 0.05

}





class SentimentTextblob:



  def __init__(self):

    self.datasets = {}

    self.scored_data = {}

    self.results = []

    self.final_predictions = {}

    self.vader = SentimentIntensityAnalyzer()



  # ------------------------------------------------------

  # Find datasets

  # ------------------------------------------------------

  def find_file(self, filename):



    current = Path.cwd()



    folders = [

      current,

      current / "healthsentiment_results",

      current.parent,

      current.parent / "healthsentiment_results"

    ]



    for folder in folders:

      path = folder / filename



      if path.is_file():

        return path



    for folder in [current, current.parent]:

      try:

        for path in folder.rglob(filename):

          if path.is_file():

            return path

      except (OSError, PermissionError):

        pass



    raise FileNotFoundError(

      f"Could not locate {filename}. "

      f"Current notebook folder: {current}"

    )



  # ------------------------------------------------------

  # Clean text

  # ------------------------------------------------------

  def clean_text(self, text):



    text = html.unescape(str(text))

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"\s+", " ", text)



    return text.strip()



  # ------------------------------------------------------

  # Load datasets

  # ------------------------------------------------------

  def load_data(self):



    print("\n========== LOADING DATASETS ==========")



    drug_path = self.find_file("drugsComTest_raw.csv")

    patient_path = self.find_file(

      "patient_feedback_dataset.xlsx"

    )



    drug = pd.read_csv(drug_path)

    patient = pd.read_excel(patient_path)



    drug.columns = drug.columns.str.strip()

    patient.columns = patient.columns.str.strip()



    print("Drug Reviews original:", drug.shape)

    print("Patient Feedback original:", patient.shape)



    # Drug reviews: 1-4 negative, 6-10 positive

    drug["rating"] = pd.to_numeric(

      drug["rating"], errors="coerce"

    )



    drug = drug.dropna(

      subset=["review", "rating"]

    ).copy()



    drug = drug[

      drug["rating"].between(1, 4) |

      drug["rating"].between(6, 10)

    ].copy()



    drug["Sentiments"] = (

      drug["rating"] >= 6

    ).astype(int)



    drug["Text"] = drug["review"].astype(str)



    # Patient feedback

    patient = patient.dropna(

      subset=["Feedback", "Sentiment"]

    ).copy()



    patient["Sentiments"] = pd.to_numeric(

      patient["Sentiment"], errors="coerce"

    )



    patient = patient[

      patient["Sentiments"].isin([0, 1])

    ].copy()



    patient["Sentiments"] = patient[

      "Sentiments"

    ].astype(int)



    patient["Text"] = patient["Feedback"].astype(str)



    self.datasets = {

      "Drug Reviews": drug,

      "Patient Feedback": patient

    }



    for name, df in self.datasets.items():



      df = df.copy()



      df["Clean_Text"] = df["Text"].apply(

        self.clean_text

      )



      df["Duplicate_Key"] = (

        df["Clean_Text"]

        .str.lower()

        .str.strip()

      )



      df = df[

        df["Duplicate_Key"].str.len() > 0

      ].copy()



      original_count = len(df)



      # Detect conflicting labels before deduplication

      conflicts = (

        df.groupby("Duplicate_Key")["Sentiments"]

        .nunique()

      )



      if (conflicts > 1).any():

        print(

          f"Warning: {name} contains "

          f"{(conflicts > 1).sum()} texts "

          "with conflicting labels."

        )



      # Remove exact duplicates

      df = df.drop_duplicates(

        subset=["Duplicate_Key"]

      ).copy()



      print(

        f"\n{name}: removed "

        f"{original_count - len(df)} duplicates"

      )



      # Sample drug reviews only

      if (

        name == "Drug Reviews"

        and MAX_DRUG_REVIEWS is not None

        and len(df) > MAX_DRUG_REVIEWS

      ):



        df, _ = train_test_split(

          df,

          train_size=MAX_DRUG_REVIEWS,

          stratify=df["Sentiments"],

          random_state=RANDOM_STATE

        )



      df = df.reset_index(drop=True)



      self.datasets[name] = df



      print("Records:", len(df))

      print("Label distribution:")

      print(df["Sentiments"].value_counts())



      df.to_csv(

        OUTPUT_FOLDER /

        f"{name.lower().replace(' ', '_')}_clean.csv",

        index=False

      )



  # ------------------------------------------------------

  # Compute sentiment scores

  # ------------------------------------------------------

  def compute_sentiment(self):



    print("\n========== COMPUTING SENTIMENT ==========")



    for name, df in self.datasets.items():



      df = df.copy()



      print(f"\nAnalyzing {name}...")



      start = time.perf_counter()



      blob = df["Clean_Text"].apply(

        lambda text: TextBlob(text).sentiment

      )



      df["Polarity_score"] = blob.apply(

        lambda x: x.polarity

      )



      df["Subjectivity_score"] = blob.apply(

        lambda x: x.subjectivity

      )



      blob_runtime = time.perf_counter() - start



      start = time.perf_counter()



      df["VADER_Compound"] = df[

        "Clean_Text"

      ].apply(

        lambda text: self.vader

        .polarity_scores(text)["compound"]

      )



      vader_runtime = time.perf_counter() - start



      self.scored_data[name] = df



      print(

        f"TextBlob runtime: {blob_runtime:.3f}s"

      )

      print(

        f"VADER runtime: {vader_runtime:.3f}s"

      )



  # ------------------------------------------------------

  # Optimize thresholds on training/validation data

  # ------------------------------------------------------

  def best_threshold(self, scores, labels, model):



    default = DEFAULT_THRESHOLDS[model]



    best = default

    highest = -1



    for threshold in THRESHOLDS[model]:



      predictions = (

        scores > threshold

      ).astype(int)



      accuracy = accuracy_score(

        labels, predictions

      )



      if accuracy > highest + 1e-12:

        highest = accuracy

        best = threshold



      elif np.isclose(accuracy, highest):

        if (

          abs(threshold - default)

          < abs(best - default)

        ):

          best = threshold



    return float(best), float(highest)



  # ------------------------------------------------------

  # Drug Reviews: validation and independent test set

  # ------------------------------------------------------

  def evaluate_drug_reviews(self):



    name = "Drug Reviews"

    df = self.scored_data[name]



    development, test = train_test_split(

      df,

      test_size=0.20,

      stratify=df["Sentiments"],

      random_state=RANDOM_STATE

    )



    candidates = []



    for model, column in [

      ("TextBlob", "Polarity_score"),

      ("VADER", "VADER_Compound")

    ]:



      threshold, score = self.best_threshold(

        development[column].to_numpy(),

        development["Sentiments"].to_numpy(),

        model

      )



      candidates.append({

        "Model": model,

        "Column": column,

        "Threshold": threshold,

        "Selection_Accuracy": score

      })



      print(

        f"{model}: validation accuracy "

        f"{score:.2%}, threshold {threshold:.3f}"

      )



    selected = max(

      candidates,

      key=lambda x: x["Selection_Accuracy"]

    )



    test = test.copy()



    test["Predicted"] = (

      test[selected["Column"]]

      > selected["Threshold"]

    ).astype(int)



    self.save_evaluation(

      name,

      selected,

      test,

      "Held-out test"

    )



  # ------------------------------------------------------

  # Patient Feedback: nested 5-fold cross-validation

  # ------------------------------------------------------

  def evaluate_patient_feedback(self):



    name = "Patient Feedback"

    df = self.scored_data[name].reset_index(drop=True)



    labels = df["Sentiments"].to_numpy()



    outer_cv = StratifiedKFold(

      n_splits=5,

      shuffle=True,

      random_state=RANDOM_STATE

    )



    predictions = np.full(

      len(df), -1, dtype=int

    )



    fold_results = []

    model_selections = []



    for fold, (train_idx, test_idx) in enumerate(

      outer_cv.split(df, labels), start=1

    ):



      train = df.iloc[train_idx]

      test = df.iloc[test_idx]



      # Inner CV selects the model and threshold.

      inner_cv = StratifiedKFold(

        n_splits=3,

        shuffle=True,

        random_state=RANDOM_STATE + fold

      )



      candidates = []



      for model, column in [

        ("TextBlob", "Polarity_score"),

        ("VADER", "VADER_Compound")

      ]:



        best_threshold = None

        best_score = -1



        for threshold in THRESHOLDS[model]:



          inner_predictions = np.full(

            len(train), -1, dtype=int

          )



          for _, val_idx in inner_cv.split(

            train, train["Sentiments"]

          ):



            inner_predictions[val_idx] = (

              train.iloc[val_idx][column]

              .to_numpy() > threshold

            ).astype(int)



          score = accuracy_score(

            train["Sentiments"],

            inner_predictions

          )



          default = DEFAULT_THRESHOLDS[model]



          if (

            score > best_score + 1e-12

            or (

              np.isclose(score, best_score)

              and (

                best_threshold is None

                or abs(threshold - default)

                < abs(best_threshold - default)

              )

            )

          ):

            best_score = score

            best_threshold = threshold



        candidates.append({

          "Model": model,

          "Column": column,

          "Threshold": float(best_threshold),

          "Selection_Accuracy": float(best_score)

        })



      selected = max(

        candidates,

        key=lambda x: x["Selection_Accuracy"]

      )



      fold_pred = (

        test[selected["Column"]].to_numpy()

        > selected["Threshold"]

      ).astype(int)



      predictions[test_idx] = fold_pred



      fold_accuracy = accuracy_score(

        test["Sentiments"], fold_pred

      )



      fold_results.append({

        "Fold": fold,

        "Selected_Model": selected["Model"],

        "Threshold": selected["Threshold"],

        "Accuracy": fold_accuracy

      })



      model_selections.append(selected["Model"])



      print(

        f"Fold {fold}: {selected['Model']}, "

        f"threshold {selected['Threshold']:.3f}, "

        f"accuracy {fold_accuracy:.2%}"

      )



    evaluated = df.copy()

    evaluated["Predicted"] = predictions



    # These are out-of-fold predictions, not training scores.

    most_selected = pd.Series(

      model_selections

    ).value_counts().index[0]



    selected_summary = {

      "Model": most_selected,

      "Threshold": np.nan,

      "Selection_Accuracy": np.nan

    }



    self.save_evaluation(

      name,

      selected_summary,

      evaluated,

      "Nested 5-fold cross-validation"

    )



    pd.DataFrame(fold_results).to_csv(

      OUTPUT_FOLDER /

      "patient_feedback_cv_folds.csv",

      index=False

    )



    # Select one model/threshold for future new feedback.

    final_candidates = []



    for model, column in [

      ("TextBlob", "Polarity_score"),

      ("VADER", "VADER_Compound")

    ]:



      threshold, score = self.best_threshold(

        df[column].to_numpy(),

        df["Sentiments"].to_numpy(),

        model

      )



      final_candidates.append({

        "Model": model,

        "Threshold": threshold,

        "Development_Accuracy": score

      })



    final_selected = max(

      final_candidates,

      key=lambda x: x["Development_Accuracy"]

    )



    print(

      "\nFinal model for future patient feedback:",

      final_selected["Model"]

    )

    print(

      "Final threshold:",

      final_selected["Threshold"]

    )



    print(

      "Note: The cross-validation accuracy above "

      "is the evaluation result. The final "

      "threshold was fitted using all 20 statements."

    )



  # ------------------------------------------------------

  # Save metrics and predictions

  # ------------------------------------------------------

  def save_evaluation(

    self, name, selected, evaluated, method

  ):



    actual = evaluated["Sentiments"]

    predicted = evaluated["Predicted"]



    accuracy = accuracy_score(actual, predicted)



    precision = precision_score(

      actual, predicted, zero_division=0

    )



    recall = recall_score(

      actual, predicted, zero_division=0

    )



    f1 = f1_score(

      actual, predicted, zero_division=0

    )



    self.results.append({

      "Dataset": name,

      "Best_Model": selected["Model"],

      "Evaluation": method,

      "Accuracy": accuracy,

      "Precision": precision,

      "Recall": recall,

      "F1_Score": f1,

      "Records": len(evaluated)

    })



    self.final_predictions[name] = evaluated



    print(f"\n========== {name.upper()} RESULTS ==========")

    print("Selected model:", selected["Model"])

    print("Evaluation:", method)

    print(f"Accuracy: {accuracy:.2%}")

    print(f"Precision: {precision:.2%}")

    print(f"Recall: {recall:.2%}")

    print(f"F1-score: {f1:.2%}")



    print("\nClassification Report:")

    print(

      classification_report(

        actual,

        predicted,

        labels=[0, 1],

        target_names=["Negative", "Positive"],

        zero_division=0

      )

    )



    safe_name = name.lower().replace(" ", "_")



    evaluated.to_csv(

      OUTPUT_FOLDER /

      f"{safe_name}_predictions.csv",

      index=False

    )



  # ------------------------------------------------------

  # Visualize results

  # ------------------------------------------------------

  def visualize_results(self):



    results = pd.DataFrame(self.results)



    print("\n========== FINAL MODEL COMPARISON ==========")

    print(results.round(4).to_string(index=False))



    results.to_csv(

      OUTPUT_FOLDER / "final_model_comparison.csv",

      index=False

    )



    fig, ax = plt.subplots(figsize=(9, 5))



    bars = ax.bar(

      results["Dataset"],

      results["Accuracy"] * 100

    )



    for bar, (_, row) in zip(

      bars, results.iterrows()

    ):

      ax.text(

        bar.get_x() + bar.get_width() / 2,

        bar.get_height() + 1,

        f"{row['Accuracy']:.2%}",

        ha="center"

      )



    ax.set_title("HealthSentiment AI - Model Accuracy")

    ax.set_ylabel("Accuracy (%)")

    ax.set_ylim(0, 100)



    plt.tight_layout()

    plt.savefig(

      OUTPUT_FOLDER / "model_accuracy.png",

      dpi=300

    )

    plt.show()



    fig, ax = plt.subplots(figsize=(9, 5))



    bars = ax.bar(

      results["Dataset"],

      results["F1_Score"] * 100

    )



    for bar, (_, row) in zip(

      bars, results.iterrows()

    ):

      ax.text(

        bar.get_x() + bar.get_width() / 2,

        bar.get_height() + 1,

        f"{row['F1_Score']:.2%}",

        ha="center"

      )



    ax.set_title("HealthSentiment AI - F1-Score")

    ax.set_ylabel("F1-Score (%)")

    ax.set_ylim(0, 100)



    plt.tight_layout()

    plt.savefig(

      OUTPUT_FOLDER / "model_f1_score.png",

      dpi=300

    )

    plt.show()



    for name, df in self.final_predictions.items():



      safe_name = name.lower().replace(" ", "_")



      cm = confusion_matrix(

        df["Sentiments"],

        df["Predicted"],

        labels=[0, 1]

      )



      disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=["Negative", "Positive"]

      )



      fig, ax = plt.subplots(figsize=(6, 5))

      disp.plot(ax=ax, cmap="Blues", values_format="d")



      ax.set_title(f"Confusion Matrix - {name}")



      plt.tight_layout()

      plt.savefig(

        OUTPUT_FOLDER /

        f"{safe_name}_confusion_matrix.png",

        dpi=300

      )

      plt.show()



      distribution = pd.DataFrame({

        "Actual": df["Sentiments"].value_counts(),

        "Predicted": df["Predicted"].value_counts()

      }).reindex([0, 1], fill_value=0)



      distribution.index = ["Negative", "Positive"]



      distribution.plot(

        kind="bar",

        figsize=(8, 5)

      )



      plt.title(f"Sentiment Distribution - {name}")

      plt.xlabel("Sentiment")

      plt.ylabel("Number of Records")

      plt.xticks(rotation=0)

      plt.tight_layout()



      plt.savefig(

        OUTPUT_FOLDER /

        f"{safe_name}_sentiment_distribution.png",

        dpi=300

      )

      plt.show()



      fig, ax = plt.subplots(figsize=(8, 5))



      scatter = ax.scatter(

        df["Polarity_score"],

        df["Subjectivity_score"],

        c=df["Sentiments"],

        cmap="bwr",

        alpha=0.6

      )



      ax.axvline(0, linestyle="--", color="gray")

      ax.set_title(

        f"Polarity vs Subjectivity - {name}"

      )

      ax.set_xlabel("Polarity")

      ax.set_ylabel("Subjectivity")



      plt.colorbar(

        scatter, ax=ax,

        label="Actual Sentiment (0=Negative, 1=Positive)"

      )



      plt.tight_layout()

      plt.savefig(

        OUTPUT_FOLDER /

        f"{safe_name}_polarity_subjectivity.png",

        dpi=300

      )

      plt.show()



  # ------------------------------------------------------

  # Automatic execution

  # ------------------------------------------------------

  def run(self):



    print("=" * 55)

    print("         HEALTHSENTIMENT AI")

    print("    Comparative Sentiment Analysis Agent")

    print("=" * 55)



    self.load_data()

    self.compute_sentiment()



    print("\n========== DRUG REVIEW EVALUATION ==========")

    self.evaluate_drug_reviews()



    print("\n========== PATIENT FEEDBACK EVALUATION ==========")

    self.evaluate_patient_feedback()



    self.visualize_results()



    print("\n========== ANALYSIS COMPLETED ==========")

    print("Results saved to:", OUTPUT_FOLDER.resolve())





# ==========================================================

# RUN THE AGENT

# ==========================================================



