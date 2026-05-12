import pandas as pd

df = pd.read_csv("data/processed/open_qa_dataset1.csv")

print("\nTOTAL")
print(len(df))

print("\nQUESTION TYPES")
print(df["question_type"].value_counts())

print("\nMATERIAL CLASS")
print(df["material_class"].value_counts())

print("\nDIFFICULTY")
print(df["difficulty"].value_counts())

print("\nSOURCE SPLIT")
print(df["source_split"].value_counts())