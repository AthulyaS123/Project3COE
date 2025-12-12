import pandas as pd
from model import TabularESAModel  # imports your class

def main():
    # Load your processed CSV
    df = pd.read_csv("data/processed/tabular_dataset.csv")

    # Create and train the model
    model = TabularESAModel()
    model.train(df)

if __name__ == "__main__":
    main()
