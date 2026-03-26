"""Unit tests for feature engineering functions."""

import pandas as pd
import pytest

from src.features import (
    apply_feature_engineering,
    create_new_features,
    drop_columns,
    encode_ordinal_features,
    encode_remaining_categoricals,
)


@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking cleaned House Prices data."""
    return pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "GrLivArea": [1500, 2000, 1200],
            "TotalBsmtSF": [1000, 800, 0],
            "FullBath": [2, 2, 1],
            "HalfBath": [1, 0, 0],
            "BsmtFullBath": [1, 0, 0],
            "BsmtHalfBath": [0, 1, 0],
            "OpenPorchSF": [50, 100, 0],
            "EnclosedPorch": [0, 0, 30],
            "3SsnPorch": [0, 0, 0],
            "ScreenPorch": [0, 50, 0],
            "YrSold": [2010, 2009, 2010],
            "YearBuilt": [2000, 1990, 1960],
            "YearRemodAdd": [2005, 1990, 2000],
            "GarageArea": [500, 0, 300],
            "Fireplaces": [1, 0, 0],
            "2ndFlrSF": [500, 0, 400],
            "ExterQual": ["Gd", "TA", "Fa"],
            "BsmtQual": ["Ex", "None", "TA"],
            "KitchenQual": ["Gd", "TA", "Fa"],
            "FireplaceQu": ["Gd", "None", "None"],
            "ExterCond": ["TA", "TA", "Gd"],
            "BsmtCond": ["TA", "None", "Fa"],
            "HeatingQC": ["Ex", "Gd", "TA"],
            "GarageQual": ["TA", "None", "TA"],
            "GarageCond": ["TA", "None", "TA"],
            "BsmtExposure": ["Gd", "None", "No"],
            "BsmtFinType1": ["GLQ", "None", "Unf"],
            "BsmtFinType2": ["Unf", "None", "Unf"],
            "GarageFinish": ["Fin", "None", "RFn"],
            "Fence": ["GdPrv", "None", "MnPrv"],
            "Functional": ["Typ", "Min1", "Mod"],
            "PavedDrive": ["Y", "N", "P"],
            "CentralAir": ["Y", "N", "Y"],
            "LotShape": ["Reg", "IR1", "IR2"],
            "LandSlope": ["Gtl", "Mod", "Sev"],
            "Utilities": ["AllPub", "AllPub", "AllPub"],
            "Street": ["Pave", "Pave", "Grvl"],
            "PoolQC": ["None", "None", "None"],
            "MiscFeature": ["None", "Shed", "None"],
            "MiscVal": [0, 400, 0],
            "Alley": ["None", "Grvl", "None"],
            "Neighborhood": ["NAmes", "CollgCr", "OldTown"],
            "SalePrice": [200000, 300000, 150000],
        }
    )


class TestCreateNewFeatures:
    def test_total_sf(self, sample_df):
        result = create_new_features(sample_df)
        assert result["TotalSF"].iloc[0] == 1500 + 1000
        assert result["TotalSF"].iloc[2] == 1200 + 0

    def test_total_bath(self, sample_df):
        result = create_new_features(sample_df)
        # Row 0: 2 + 0.5*1 + 1 + 0.5*0 = 3.5
        assert result["TotalBath"].iloc[0] == 3.5
        # Row 1: 2 + 0 + 0 + 0.5*1 = 2.5
        assert result["TotalBath"].iloc[1] == 2.5

    def test_total_porch(self, sample_df):
        result = create_new_features(sample_df)
        assert result["TotalPorchSF"].iloc[0] == 50
        assert result["TotalPorchSF"].iloc[1] == 150  # 100 + 50

    def test_house_age(self, sample_df):
        result = create_new_features(sample_df)
        assert result["HouseAge"].iloc[0] == 10  # 2010 - 2000
        assert result["HouseAge"].iloc[2] == 50  # 2010 - 1960

    def test_is_remodeled(self, sample_df):
        result = create_new_features(sample_df)
        assert result["IsRemodeled"].iloc[0] == 1  # 2005 != 2000
        assert result["IsRemodeled"].iloc[1] == 0  # 1990 == 1990

    def test_has_flags(self, sample_df):
        result = create_new_features(sample_df)
        assert result["HasGarage"].iloc[0] == 1
        assert result["HasGarage"].iloc[1] == 0
        assert result["HasBasement"].iloc[2] == 0
        assert result["HasFireplace"].iloc[0] == 1
        assert result["Has2ndFloor"].iloc[1] == 0

    def test_does_not_modify_original(self, sample_df):
        original_cols = sample_df.columns.tolist()
        create_new_features(sample_df)
        assert sample_df.columns.tolist() == original_cols


class TestEncodeOrdinalFeatures:
    def test_quality_encoding(self, sample_df):
        result = encode_ordinal_features(sample_df)
        assert result["ExterQual"].iloc[0] == 4  # Gd
        assert result["ExterQual"].iloc[1] == 3  # TA
        assert result["ExterQual"].iloc[2] == 2  # Fa

    def test_bsmt_exposure_encoding(self, sample_df):
        result = encode_ordinal_features(sample_df)
        assert result["BsmtExposure"].iloc[0] == 4  # Gd
        assert result["BsmtExposure"].iloc[2] == 1  # No

    def test_central_air_binary(self, sample_df):
        result = encode_ordinal_features(sample_df)
        assert result["CentralAir"].iloc[0] == 1
        assert result["CentralAir"].iloc[1] == 0

    def test_does_not_modify_original(self, sample_df):
        original_vals = sample_df["ExterQual"].tolist()
        encode_ordinal_features(sample_df)
        assert sample_df["ExterQual"].tolist() == original_vals


class TestDropColumns:
    def test_drops_specified_columns(self, sample_df):
        result = drop_columns(sample_df)
        assert "Id" not in result.columns
        assert "Utilities" not in result.columns
        assert "Street" not in result.columns

    def test_preserves_target(self, sample_df):
        result = drop_columns(sample_df, target_col="SalePrice")
        assert "SalePrice" in result.columns

    def test_does_not_modify_original(self, sample_df):
        original_cols = sample_df.columns.tolist()
        drop_columns(sample_df)
        assert sample_df.columns.tolist() == original_cols


class TestEncodeRemainingCategoricals:
    def test_creates_dummy_columns(self, sample_df):
        result = encode_remaining_categoricals(sample_df)
        # Neighborhood should be one-hot encoded
        assert any(c.startswith("Neighborhood_") for c in result.columns)
        # No object columns remain
        assert len(result.select_dtypes(include=["object", "string"]).columns) == 0


class TestApplyFeatureEngineering:
    def test_no_missing_values(self, sample_df):
        test_df = sample_df.drop(columns=["SalePrice"])
        train_result, test_result = apply_feature_engineering(sample_df, test_df)
        assert train_result.isnull().sum().sum() == 0
        assert test_result.isnull().sum().sum() == 0

    def test_same_columns(self, sample_df):
        test_df = sample_df.drop(columns=["SalePrice"])
        train_result, test_result = apply_feature_engineering(sample_df, test_df)
        # Train has target, test doesn't
        train_cols = set(train_result.columns) - {"SalePrice"}
        test_cols = set(test_result.columns)
        assert train_cols == test_cols

    def test_target_preserved(self, sample_df):
        test_df = sample_df.drop(columns=["SalePrice"])
        train_result, _ = apply_feature_engineering(sample_df, test_df)
        assert "SalePrice" in train_result.columns
        assert len(train_result) == len(sample_df)

    def test_no_object_columns_remain(self, sample_df):
        test_df = sample_df.drop(columns=["SalePrice"])
        train_result, test_result = apply_feature_engineering(sample_df, test_df)
        assert len(train_result.select_dtypes(include=["object", "string"]).columns) == 0
        assert len(test_result.select_dtypes(include=["object", "string"]).columns) == 0
