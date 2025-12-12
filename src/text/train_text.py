import pandas as pd
from model_text import TextESAClassifier

def main():
    # IMPORTANT: Adjusted for execution from notebooks folder
    df = pd.read_csv("../data/processed/tabular_dataset.csv")

    # Temporary text input (fake)
    texts = df["school_name"].tolist()
    labels = df["ESA_class"].tolist()

    model = TextESAClassifier()
    model.train(texts, labels, epochs=1)
    model.evaluate(texts, labels)

    model.save("text_model")

if __name__ == "__main__":
    main()
