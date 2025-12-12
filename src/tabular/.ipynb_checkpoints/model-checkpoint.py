import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

ESA_BINS = [0, 35, 50, 65, 80, 100]

def discretize_score(score):
    for i in range(len(ESA_BINS) - 1):
        if ESA_BINS[i] <= score <= ESA_BINS[i+1]:
            return i
    return None

class TabularESAModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, random_state=42)

    def train(self, df):
        feature_cols = [
            "perc_econ_disadv",
            "student_teacher_ratio",
            "perc_ell",
            "enrollment",
        ]

        df["ESA_class"] = df["ESA_score"].apply(discretize_score)

        X = df[feature_cols]
        y = df["ESA_class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        print(classification_report(y_test, y_pred, zero_division=0))
        print("Macro F1:", f1_score(y_test, y_pred, average="macro", zero_division=0))

    def predict(self, new_df):
        return self.model.predict(new_df)