import os
import re
import pandas as pd
import pdfplumber

# Resolve project root based on this file's location: src/text/extract_text.py
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))

LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "esa_fca_scores_cleaned.csv")
PDF_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "facility_pdfs")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "text_dataset.csv")


def safe_name(school_name: str) -> str:
    """
    Turn 'Liberal Arts & Science Academy (LASA)' into 'liberal_arts_science_academy_lasa'
    so we can match 'liberal_arts_science_academy_lasa.pdf'.
    Adjust if your filenames use a different pattern.
    """
    s = school_name.lower()
    s = s.replace("&", "and")
    for ch in ["(", ")", ",", ".", "'"]:
        s = s.replace(ch, "")
    s = "_".join(s.split())
    return s


def bin_score(score: float) -> int:
    """
    Simple 3-bin scheme for ESA/FCA:
      0: low quality (< 50)
      1: medium (50–69)
      2: high (>= 70)
    You can tweak thresholds later in your report if you want.
    """
    if score < 50:
        return 0
    elif score < 70:
        return 1
    else:
        return 2


def extract_clean_text(pdf_path: str) -> str:
    """
    Extract narrative text from a PDF and remove lines that might leak scores
    or are mostly numeric.
    """
    if not os.path.exists(pdf_path):
        print(f"[WARN] PDF not found: {pdf_path}")
        return ""

    kept_lines = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                # Skip lines with digits (likely scores, years, indices)
                if re.search(r"\d", line):
                    continue

                # Skip obvious leakage / meta lines
                if re.search(r"(esa|fca|score|rating|index|points|percent|%)",
                             line, flags=re.IGNORECASE):
                    continue

                kept_lines.append(line)

    return " ".join(kept_lines)


def main():
    labels_df = pd.read_csv(LABELS_PATH)

    rows = []
    for _, row in labels_df.iterrows():
        school = row["School"]
        esa_score = float(row["ESA_Score"])
        fca_score = float(row["FCA_Score"])

        pdf_filename = safe_name(school) + ".pdf"
        pdf_path = os.path.join(PDF_DIR, pdf_filename)

        print(f"[INFO] Processing {school} -> {pdf_filename}")
        text = extract_clean_text(pdf_path)

        if not text:
            print(f"[WARN] No usable text extracted for {school}")

        esa_class = bin_score(esa_score)
        fca_class = bin_score(fca_score)

        rows.append({
            "School": school,
            "ESA_Score": esa_score,
            "FCA_Score": fca_score,
            "ESA_Class": esa_class,
            "FCA_Class": fca_class,
            "text": text
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Saved text dataset to {OUTPUT_PATH}")
    print(out_df[["School", "ESA_Class", "FCA_Class"]].head())


if __name__ == "__main__":
    main()