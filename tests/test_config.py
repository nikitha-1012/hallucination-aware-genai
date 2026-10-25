"""Tests for configuration loading and validation (Part 1)."""

import pytest
import yaml
from pathlib import Path

from src.config import AppConfig, ConfigError, load_config


def test_load_valid_default_config():
    """Verify that the repository's root config.yaml loads properly."""
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.chunk_size == 1000
    assert cfg.chunk_overlap == 150
    assert cfg.top_k == 5
    assert cfg.retry_top_k == 8
    assert cfg.strong_score == 0.45
    assert cfg.weak_score == 0.30
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.llm_model == "llama3.2:3b"
    assert cfg.temperature == 0.1
    assert cfg.llm_timeout_seconds == 120
    assert cfg.ollama_base_url == "http://localhost:11434"
    assert cfg.default_collection == "python_fundamentals"
    assert cfg.get_collection_path().exists()


def test_missing_config_file(tmp_path):
    """Missing config file must raise ConfigError."""
    fake_path = tmp_path / "non_existent.yaml"
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config(fake_path)


def test_missing_required_key(tmp_path):
    """Missing required key must raise ConfigError specifying the key."""
    # Create valid base config dict
    data_dir = tmp_path / "data" / "collections" / "python_fundamentals"
    data_dir.mkdir(parents=True)

    base_data = {
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "retry_top_k": 8,
        "strong_score": 0.45,
        "weak_score": 0.30,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "llama3.2:3b",
        "temperature": 0.1,
        "llm_timeout_seconds": 120,
        "ollama_base_url": "http://localhost:11434",
        "data_dir": "data/collections",
        "default_collection": "python_fundamentals",
    }

    # Delete 'llm_model'
    del base_data["llm_model"]

    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(base_data, f)

    with pytest.raises(ConfigError, match="Missing required configuration key: 'llm_model'"):
        load_config(cfg_file)


def test_bad_threshold_order(tmp_path):
    """weak_score >= strong_score must raise ConfigError."""
    data_dir = tmp_path / "data" / "collections" / "python_fundamentals"
    data_dir.mkdir(parents=True)

    bad_data = {
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "retry_top_k": 8,
        "strong_score": 0.30,
        "weak_score": 0.45,  # Invalid: weak > strong
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "llama3.2:3b",
        "temperature": 0.1,
        "llm_timeout_seconds": 120,
        "ollama_base_url": "http://localhost:11434",
        "data_dir": "data/collections",
        "default_collection": "python_fundamentals",
    }

    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(bad_data, f)

    with pytest.raises(ConfigError, match="weak_score .* must be less than strong_score"):
        load_config(cfg_file)


def test_bad_chunk_overlap(tmp_path):
    """chunk_overlap >= chunk_size must raise ConfigError."""
    data_dir = tmp_path / "data" / "collections" / "python_fundamentals"
    data_dir.mkdir(parents=True)

    bad_data = {
        "chunk_size": 500,
        "chunk_overlap": 500,  # Invalid: overlap >= size
        "top_k": 5,
        "retry_top_k": 8,
        "strong_score": 0.45,
        "weak_score": 0.30,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "llama3.2:3b",
        "temperature": 0.1,
        "llm_timeout_seconds": 120,
        "ollama_base_url": "http://localhost:11434",
        "data_dir": "data/collections",
        "default_collection": "python_fundamentals",
    }

    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(bad_data, f)

    with pytest.raises(ConfigError, match="chunk_overlap .* must be less than chunk_size"):
        load_config(cfg_file)


def test_missing_collection_folder(tmp_path):
    """Missing collection directory must raise ConfigError."""
    # Data directory exists, but collection folder does NOT exist
    data_dir = tmp_path / "data" / "collections"
    data_dir.mkdir(parents=True)

    bad_data = {
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "retry_top_k": 8,
        "strong_score": 0.45,
        "weak_score": 0.30,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "llama3.2:3b",
        "temperature": 0.1,
        "llm_timeout_seconds": 120,
        "ollama_base_url": "http://localhost:11434",
        "data_dir": "data/collections",
        "default_collection": "nonexistent_collection",
    }

    cfg_file = tmp_path / "config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(bad_data, f)

    with pytest.raises(ConfigError, match="Configured collection directory does not exist"):
        load_config(cfg_file)
