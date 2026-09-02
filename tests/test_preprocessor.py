"""
Unit tests for data preprocessor and feature selection pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd
from src.data_loader import generate_synthetic_cic_ids
from src.preprocessor import NIDSPreprocessor


@pytest.fixture
def sample_data():
    return generate_synthetic_cic_ids(n_samples=2000, random_state=42)


def test_preprocessor_fit_transform(sample_data):
    top_k = 20
    preprocessor = NIDSPreprocessor(top_k_features=top_k)
    processed_df = preprocessor.fit_transform(sample_data, label_col="Label")
    
    # Assertions
    assert processed_df.shape[0] == 2000
    assert processed_df.shape[1] == top_k + 1  # Top-k features + Target
    assert "Target" in processed_df.columns
    assert len(preprocessor.selected_features) == top_k
    assert not processed_df.isnull().any().any()
    assert not np.isinf(processed_df.select_dtypes(include=[np.number])).any().any()


def test_preprocessor_serialization(sample_data, tmp_path):
    preprocessor = NIDSPreprocessor(top_k_features=15)
    preprocessor.fit(sample_data, label_col="Label")
    preprocessor.save(models_dir=tmp_path)
    
    # Reload and test consistency
    loaded_preprocessor = NIDSPreprocessor.load(models_dir=tmp_path)
    assert loaded_preprocessor.fitted is True
    assert loaded_preprocessor.selected_features == preprocessor.selected_features
    
    # Transform on unseen batch
    test_batch = sample_data.iloc[:50]
    orig_transformed = preprocessor.transform(test_batch, label_col="Label")
    loaded_transformed = loaded_preprocessor.transform(test_batch, label_col="Label")
    
    pd.testing.assert_frame_equal(orig_transformed, loaded_transformed)


def test_label_inverse_transformation(sample_data):
    preprocessor = NIDSPreprocessor(top_k_features=10)
    preprocessor.fit(sample_data, label_col="Label")
    
    original_classes = sample_data["Label"].unique()
    encoded_vals = preprocessor.label_encoder.transform(original_classes)
    decoded_classes = preprocessor.inverse_transform_labels(encoded_vals)
    
    np.testing.assert_array_equal(original_classes, decoded_classes)
