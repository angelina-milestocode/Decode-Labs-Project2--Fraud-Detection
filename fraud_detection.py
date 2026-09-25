import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv("creditcard.csv")

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# --------------------------------------------------
# 2. CHECK THE DATA
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nFraud percentage:")
print(df["Class"].mean() * 100)


# --------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]


# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 5. LOGISTIC REGRESSION + SMOTE
# --------------------------------------------------

print("\nTraining Logistic Regression...")

logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_params = {
    "model__C": [0.1, 1, 10]
}

logistic_grid = GridSearchCV(
    logistic_pipeline,
    logistic_params,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

logistic_grid.fit(X_train, y_train)

logistic_model = logistic_grid.best_estimator_

print("Best Logistic Regression parameters:")
print(logistic_grid.best_params_)


# --------------------------------------------------
# 6. RANDOM FOREST + SMOTE
# --------------------------------------------------

print("\nTraining Random Forest...")

random_forest_pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])

random_forest_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10]
}

random_forest_grid = GridSearchCV(
    random_forest_pipeline,
    random_forest_params,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

random_forest_grid.fit(X_train, y_train)

random_forest_model = random_forest_grid.best_estimator_

print("Best Random Forest parameters:")
print(random_forest_grid.best_params_)


# --------------------------------------------------
# 7. LOGISTIC REGRESSION EVALUATION
# --------------------------------------------------

logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

logistic_precision = precision_score(
    y_test,
    logistic_predictions
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions
)

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)


# --------------------------------------------------
# 8. RANDOM FOREST EVALUATION
# --------------------------------------------------

random_forest_predictions = random_forest_model.predict(X_test)
random_forest_probabilities = random_forest_model.predict_proba(X_test)[:, 1]

random_forest_precision = precision_score(
    y_test,
    random_forest_predictions
)

random_forest_recall = recall_score(
    y_test,
    random_forest_predictions
)

random_forest_roc_auc = roc_auc_score(
    y_test,
    random_forest_probabilities
)


# --------------------------------------------------
# 9. PRINT RESULTS
# --------------------------------------------------

print("\n==============================")
print("MODEL RESULTS")
print("==============================")

print("\nLogistic Regression")
print("Precision:", round(logistic_precision, 4))
print("Recall:", round(logistic_recall, 4))
print("ROC-AUC:", round(logistic_roc_auc, 4))

print("\nRandom Forest")
print("Precision:", round(random_forest_precision, 4))
print("Recall:", round(random_forest_recall, 4))
print("ROC-AUC:", round(random_forest_roc_auc, 4))


# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Precision": [
        logistic_precision,
        random_forest_precision
    ],
    "Recall": [
        logistic_recall,
        random_forest_recall
    ],
    "ROC-AUC": [
        logistic_roc_auc,
        random_forest_roc_auc
    ]
})

results.to_csv(
    "model_results.csv",
    index=False
)

print("\nResults saved to model_results.csv")


# --------------------------------------------------
# 11. CLASS DISTRIBUTION GRAPH
# --------------------------------------------------

plt.figure(figsize=(6, 4))

df["Class"].value_counts().plot(
    kind="bar"
)

plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")

plt.xticks(
    [0, 1],
    ["Genuine", "Fraud"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "class_distribution.png"
)

plt.show()

print("\nProject completed successfully!")