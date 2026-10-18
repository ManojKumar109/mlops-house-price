from sklearn.datasets import fetch_california_housing
from pathlib import Path

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

housing = fetch_california_housing(as_frame=True)
df = housing.frame

df.to_csv(data_dir / "house_price.csv", index=False)

print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head())