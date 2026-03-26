"""Feature engineering functions for the House Prices competition."""

import pandas as pd

# Quality scale mapping — ordinal encoding for quality/condition features
# WHY: These are ordinal categories with a natural order that models can exploit
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

EXPOSURE_MAP = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}

FINISH_MAP = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}

GARAGE_FINISH_MAP = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}

FENCE_MAP = {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}

# Columns to drop — low variance or redundant
# WHY: Utilities has 1 unique value in test, Street has 2 with 99.6% Pave
COLS_TO_DROP = ["Id", "Utilities", "Street", "PoolQC", "MiscFeature", "MiscVal", "Alley"]

# Quality columns to encode ordinally
QUALITY_COLS = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
]


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing columns.

    Args:
        df: DataFrame with raw features (after cleaning).

    Returns:
        DataFrame with new features added.
    """
    df = df.copy()

    # Total living area = above ground + basement
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]

    # Total bathrooms (full + half weighted at 0.5)
    df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]

    # Total porch area
    df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]

    # House age and remodel age at time of sale
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    # Was the house remodeled? (remodel year != build year)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    # Has feature flags
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

    return df


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ordinal categorical features as numeric.

    Args:
        df: DataFrame with categorical quality/condition columns.

    Returns:
        DataFrame with ordinal columns replaced by numeric values.
    """
    df = df.copy()

    for col in QUALITY_COLS:
        if col in df.columns:
            df[col] = df[col].map(QUALITY_MAP).fillna(0).astype(int)

    if "BsmtExposure" in df.columns:
        df["BsmtExposure"] = df["BsmtExposure"].map(EXPOSURE_MAP).fillna(0).astype(int)

    if "BsmtFinType1" in df.columns:
        df["BsmtFinType1"] = df["BsmtFinType1"].map(FINISH_MAP).fillna(0).astype(int)

    if "BsmtFinType2" in df.columns:
        df["BsmtFinType2"] = df["BsmtFinType2"].map(FINISH_MAP).fillna(0).astype(int)

    if "GarageFinish" in df.columns:
        df["GarageFinish"] = df["GarageFinish"].map(GARAGE_FINISH_MAP).fillna(0).astype(int)

    if "Fence" in df.columns:
        df["Fence"] = df["Fence"].map(FENCE_MAP).fillna(0).astype(int)

    # Functional: Typ=0 (typical) is best, higher = more deductions
    functional_map = {"Typ": 0, "Min1": 1, "Min2": 2, "Mod": 3, "Maj1": 4, "Maj2": 5, "Sev": 6, "Sal": 7}
    if "Functional" in df.columns:
        df["Functional"] = df["Functional"].map(functional_map).fillna(0).astype(int)

    # PavedDrive: ordinal
    paved_map = {"N": 0, "P": 1, "Y": 2}
    if "PavedDrive" in df.columns:
        df["PavedDrive"] = df["PavedDrive"].map(paved_map).fillna(0).astype(int)

    # CentralAir: binary
    if "CentralAir" in df.columns:
        df["CentralAir"] = (df["CentralAir"] == "Y").astype(int)

    # LotShape: ordinal (Reg is best)
    lot_shape_map = {"Reg": 0, "IR1": 1, "IR2": 2, "IR3": 3}
    if "LotShape" in df.columns:
        df["LotShape"] = df["LotShape"].map(lot_shape_map).fillna(0).astype(int)

    # LandSlope: ordinal
    slope_map = {"Gtl": 0, "Mod": 1, "Sev": 2}
    if "LandSlope" in df.columns:
        df["LandSlope"] = df["LandSlope"].map(slope_map).fillna(0).astype(int)

    return df


def drop_columns(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """Drop low-value columns.

    Args:
        df: DataFrame to process.
        target_col: Target column name to preserve (not dropped).

    Returns:
        DataFrame with specified columns removed.
    """
    df = df.copy()
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns and c != target_col]
    df = df.drop(columns=cols_to_drop)
    return df


def encode_remaining_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode remaining categorical columns.

    Args:
        df: DataFrame that may still have object/string columns.

    Returns:
        DataFrame with all categorical columns one-hot encoded.
    """
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not cat_cols:
        return df
    df = pd.get_dummies(df, columns=cat_cols, dtype=int)
    return df


def apply_feature_engineering(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "SalePrice",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the full feature engineering pipeline to train and test.

    This is the single entry point called from the notebook.
    Ensures identical transformations on both datasets.

    Args:
        train_df: Training DataFrame (with target column).
        test_df: Test DataFrame (without target column).
        target_col: Name of the target column.

    Returns:
        Tuple of (processed_train, processed_test).
    """
    # Step 1: Create new features
    train_df = create_new_features(train_df)
    test_df = create_new_features(test_df)

    # Step 2: Encode ordinal features
    train_df = encode_ordinal_features(train_df)
    test_df = encode_ordinal_features(test_df)

    # Step 3: Drop low-value columns
    train_df = drop_columns(train_df, target_col=target_col)
    test_df = drop_columns(test_df, target_col=target_col)

    # Step 4: One-hot encode remaining categoricals
    # Align columns between train and test after encoding
    target = train_df[target_col]
    train_df = train_df.drop(columns=[target_col])

    train_df = encode_remaining_categoricals(train_df)
    test_df = encode_remaining_categoricals(test_df)

    # Align columns: keep only columns present in both
    common_cols = train_df.columns.intersection(test_df.columns)
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]

    # Restore target
    train_df[target_col] = target.values

    return train_df, test_df
