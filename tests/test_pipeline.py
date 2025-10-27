"""
Tests básicos del pipeline
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import clean_data, prepare_features, split_data


def test_clean_data():
    df = pd.DataFrame({
        'A': [1, 2, 2, 3],
        'B': [4, 5, 5, 6],
        'quality': [5, 6, 6, 7]
    })
    
    df_clean = clean_data(df, handle_missing=True)
    
    assert df_clean.shape[0] == 3
    assert df_clean.duplicated().sum() == 0


def test_prepare_features():
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'quality': [5, 6, 7, 5, 6]
    })
    
    X, y, scaler = prepare_features(df, 'quality', scale=False)
    
    assert X.shape[1] == 2
    assert len(y) == 5
    assert set(y.unique()) == {0, 1}


def test_split_data():
    X = pd.DataFrame({
        'A': range(100),
        'B': range(100, 200)
    })
    y = pd.Series([0, 1] * 50)
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20


def test_config_exists():
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "config.yaml"
    assert config_path.exists(), "config.yaml no existe"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
