# train.py – Vehicle Price Prediction with Imputation

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_and_clean(path: str):
    """Load CSV, ensure price exists, clean target, and select feature columns."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset not found at: {path_obj.resolve()}")

    print(f"✅ Loading dataset from: {path_obj.resolve()}")
    df = pd.read_csv(path_obj)
    print(f"   Original rows: {len(df)}, columns: {list(df.columns)}")

    if "price" not in df.columns:
        raise ValueError("Your dataset must contain a 'price' column as the target.")

    # Ensure price is numeric and drop rows with invalid/missing price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["price"])
    print(f"   Dropped {before - len(df)} rows with missing/invalid price.")
    print(f"   Rows remaining for training: {len(df)}")

    # Candidate feature columns
    cat_candidates = [
        "make", "model", "fuel", "transmission", "body",
        "exterior_color", "interior_color", "drivetrain", "trim",
    ]
    num_candidates = ["year", "mileage", "cylinders", "doors"]

    cat_cols = [c for c in cat_candidates if c in df.columns]
    num_cols = [c for c in num_candidates if c in df.columns]

    if not (cat_cols or num_cols):
        raise ValueError(
            "No usable feature columns found. "
            "Expected some of: make, model, fuel, transmission, body, "
            "exterior_color, interior_color, drivetrain, trim, "
            "year, mileage, cylinders, doors."
        )

    print("✅ Using feature columns:")
    print(f"   Categorical: {cat_cols}")
    print(f"   Numeric    : {num_cols}")

    return df, cat_cols, num_cols


def train(csv_path: str, model_path: str = "vehicle_price_model.joblib"):
    df, cat_cols, num_cols = load_and_clean(csv_path)

    y = df["price"]
    X = df[cat_cols + num_cols]  # features only

    # Preprocessor with IMPUTATION
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    print("✅ Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✅ Training model (may take a moment)...")
    model.fit(X_train, y_train)

    print("✅ Evaluating model...")
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n📊 Evaluation metrics:")
    print(f"   MAE: {mae:.2f}")
    print(f"   R² : {r2:.4f}")

    out_path = Path(model_path).resolve()
    joblib.dump(model, out_path)
    print(f"\n💾 Model saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train vehicle price prediction model")
    parser.add_argument("--data", required=True, help="Path to dataset CSV (with 'price' column)")
    parser.add_argument(
        "--out",
        default="vehicle_price_model.joblib",
        help="Output model file name (default: vehicle_price_model.joblib)",
    )
    args = parser.parse_args()

    train(args.data, args.out)


if __name__ == "__main__":
    main()
