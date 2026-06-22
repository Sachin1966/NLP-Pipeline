import pytest
import os
import pandas as pd
from src.ingestion.loader import DocumentIngester
from src.ingestion.validator import DataValidator
from src.ingestion.synthetic import generate_synthetic_dataset

def test_data_validator():
    validator = DataValidator()
    
    # Test valid review
    valid_rec = {"text": "I bought the Nexus Laptop last week and it is amazing."}
    res = validator.validate_record(valid_rec)
    assert res["is_valid"] is True
    
    # Test corrupted review (empty text)
    corrupted_rec = {"text": "   "}
    res = validator.validate_record(corrupted_rec)
    assert res["is_valid"] is False
    assert res["is_corrupted"] is True
    
    # Test non-English review (should be flagged)
    non_eng_rec = {"text": "Ce produit est vraiment magnifique et incroyable."}
    res = validator.validate_record(non_eng_rec)
    assert res["is_valid"] is False
    assert res["unsupported_lang"] is True

def test_data_validator_batch():
    validator = DataValidator()
    records = [
        {"text": "Excellent mouse with smooth glide"},
        {"text": "Excellent mouse with smooth glide"}, # Duplicate
        {"text": "   "}, # Corrupted
        {"text": "The keyboard switches feel great."}
    ]
    res = validator.validate_batch(records)
    assert len(res["valid_records"]) == 2
    assert res["report"]["duplicate_records"] == 1
    assert res["report"]["corrupted_records"] == 1

def test_synthetic_data_generator():
    test_csv = "data/raw/test_synth.csv"
    try:
        generate_synthetic_dataset(output_path=test_csv, count=100)
        assert os.path.exists(test_csv)
        df = pd.read_csv(test_csv)
        assert len(df) == 100
        assert "text" in df.columns
        assert "sentiment" in df.columns
        assert "category" in df.columns
    finally:
        if os.path.exists(test_csv):
            os.remove(test_csv)
