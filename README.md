# Project3COE
Multimodal Equity Risk Scoring for Austin ISD High Schools

This project develops a multimodal machine learning pipeline to assess educational equity risk for Austin ISD high schools by combining textual descriptions and tabular school-level indicators. The system demonstrates how integrating qualitative and quantitative data improves equity-focused risk assessment compared to single-modality models.

Project Overview

Educational inequity is influenced by both structural factors (staffing levels, school characteristics) and contextual narratives (school descriptions, reports, qualitative indicators). Traditional models often rely on a single data modality, limiting their ability to capture these interactions.

This project builds three models:

Text-based model (DistilBERT) to classify equity risk from school-related text.

Tabular model (Random Forest) using NCES and AISD indicators.

Multimodal fusion model combining text and tabular signals to predict final equity risk classes.<img width="669" height="502" alt="Screenshot 2025-12-13 at 6 29 56 AM" src="https://github.com/user-attachments/assets/ac0bc201-ac57-44ce-b9cf-617d53873a14" />

 Trained models are not committed to GitHub due to file size constraints. All results are reproducible by running the notebooks.

Environment Setup
1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

Key libraries:

PyTorch, Transformers
Scikit-learn
Pandas, NumPy
Matplotlib, Seaborn
How to Run the Project (Correct Order)

Run notebooks in the order below to reproduce all results.

1️. Text-Based Model
Notebook:
notebooks/text_model.ipynb
Purpose:
Classify schools into ESA and FCA risk classes using textual data.
Key steps:
Tokenization with DistilBERT
Multi-task classification (ESA + FCA)
Class imbalance handling
Confusion matrix and classification report
Output:
data/processed/text_predictions.csv

2️. Tabular Model
Notebook:
notebooks/tabular_model.ipynb
Data sources:
NCES Directory
NCES Staff counts
NCES School characteristics
AISD-derived metadata
Key steps:
Dataset merging and cleaning
Feature engineering (teacher counts, ratios)
Random Forest modeling
Regression and classification evaluation
Output:
data/processed/tabular_features.csv

3️. Multimodal Fusion Model
Notebook:
notebooks/fusion_model.ipynb
Purpose:
Combine text-based predictions with tabular features to predict final ESA risk class.
Fusion approach:
Text predictions used as additional features
Tabular + text feature concatenation
Random Forest classifier
Cross-validated evaluation
Evaluation includes:
Accuracy
Weighted F1-score
Confusion matrices comparing:
Text-only
Tabular-only
Multimodal fusion
Evaluation & Visualizations
Generated figures include:
Confusion matrices (Text / Tabular / Fusion)
Feature importance plots
Side-by-side modality comparisons

Saved in:
results/figures/


Reproducing All Results
jupyter notebook notebooks/text_model.ipynb
jupyter notebook notebooks/tabular_model.ipynb
jupyter notebook notebooks/fusion_model.ipynb
