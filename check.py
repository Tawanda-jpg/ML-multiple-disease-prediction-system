from pathlib import Path

path = Path(r"C:/Users/user/Downloads/ML-multiple-disease-prediction-system/dataset/heart.csv")

print("Exists:", path.exists())
print("Suffix:", path.suffix)
print("First bytes:", path.read_bytes()[:50])