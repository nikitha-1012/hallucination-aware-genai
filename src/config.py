"""Configuration loader and validator module.

Loads settings from config.yaml, validates all required keys, numeric bounds,
and folder existence, and returns a typed AppConfig object.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import yaml


class ConfigError(Exception):
    """Raised when configuration values are missing, invalid, or inconsistent."""
    pass


REQUIRED_KEYS = [
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "retry_top_k",
    "strong_score",
    "weak_score",
    "embedding_model",
    "llm_model",
    "temperature",
    "llm_timeout_seconds",
    "ollama_base_url",
]


@dataclass(frozen=True)
class AppConfig:
    """Typed application configuration."""
    chunk_size: int
    chunk_overlap: int
    top_k: int
    retry_top_k: int
    strong_score: float
    weak_score: float
    embedding_model: str
    llm_model: str
    temperature: float
    llm_timeout_seconds: int
    ollama_base_url: str
    data_dir: str = "data/collections"
    indexes_dir: str = "indexes"
    default_collection: str = "python_fundamentals"
    root_dir: Path = Path(".")

    def get_collection_path(self, collection_name: Optional[str] = None) -> Path:
        """Return the resolved path for a given collection directory."""
        col = collection_name or self.default_collection
        return (self.root_dir / self.data_dir / col).resolve()

    def get_indexes_path(self) -> Path:
        """Return the resolved path for the Chroma indexes directory."""
        return (self.root_dir / self.indexes_dir).resolve()


def _get_project_root() -> Path:
    """Infer the project root directory from this file's location."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load, validate, and return the application configuration.

    Args:
        config_path: Optional explicit path to config.yaml. Defaults to
                     'config.yaml' located in the project root.

    Returns:
        AppConfig: A validated, typed configuration instance.

    Raises:
        ConfigError: If config file is missing, invalid YAML, missing required keys,
                     fails numeric range checks, or collection directories are missing.
    """
    if config_path is None:
        target_path = _get_project_root() / "config.yaml"
    else:
        target_path = Path(config_path).resolve()

    if not target_path.exists() or not target_path.is_file():
        raise ConfigError(f"Configuration file not found: {target_path}")

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
    except Exception as e:
        raise ConfigError(f"Failed to parse YAML from {target_path}: {e}")

    if not isinstance(raw_cfg, dict):
        raise ConfigError(f"Configuration file {target_path} must define a YAML mapping/dictionary.")

    # 1. Validate required keys
    for key in REQUIRED_KEYS:
        if key not in raw_cfg or raw_cfg[key] is None:
            raise ConfigError(f"Missing required configuration key: '{key}'")

    # 2. Extract values and validate types / numeric ranges
    try:
        chunk_size = int(raw_cfg["chunk_size"])
        chunk_overlap = int(raw_cfg["chunk_overlap"])
        top_k = int(raw_cfg["top_k"])
        retry_top_k = int(raw_cfg["retry_top_k"])
        strong_score = float(raw_cfg["strong_score"])
        weak_score = float(raw_cfg["weak_score"])
        temperature = float(raw_cfg["temperature"])
        llm_timeout_seconds = int(raw_cfg["llm_timeout_seconds"])
    except (ValueError, TypeError) as e:
        raise ConfigError(f"Invalid numeric configuration value: {e}")

    embedding_model = str(raw_cfg["embedding_model"]).strip()
    llm_model = str(raw_cfg["llm_model"]).strip()
    ollama_base_url = str(raw_cfg["ollama_base_url"]).strip()

    if not embedding_model:
        raise ConfigError("Configuration key 'embedding_model' cannot be empty.")
    if not llm_model:
        raise ConfigError("Configuration key 'llm_model' cannot be empty.")
    if not ollama_base_url:
        raise ConfigError("Configuration key 'ollama_base_url' cannot be empty.")

    if chunk_size <= 0:
        raise ConfigError(f"chunk_size must be positive, got {chunk_size}")

    if chunk_overlap < 0:
        raise ConfigError(f"chunk_overlap must be non-negative, got {chunk_overlap}")

    if chunk_overlap >= chunk_size:
        raise ConfigError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )

    if top_k <= 0:
        raise ConfigError(f"top_k must be positive, got {top_k}")

    if retry_top_k <= 0:
        raise ConfigError(f"retry_top_k must be positive, got {retry_top_k}")

    if retry_top_k < top_k:
        raise ConfigError(
            f"retry_top_k ({retry_top_k}) must be greater than or equal to top_k ({top_k})"
        )

    if weak_score >= strong_score:
        raise ConfigError(
            f"weak_score ({weak_score}) must be less than strong_score ({strong_score})"
        )

    if temperature < 0.0:
        raise ConfigError(f"temperature must be non-negative, got {temperature}")

    if llm_timeout_seconds <= 0:
        raise ConfigError(f"llm_timeout_seconds must be positive, got {llm_timeout_seconds}")

    data_dir = str(raw_cfg.get("data_dir", "data/collections"))
    indexes_dir = str(raw_cfg.get("indexes_dir", "indexes"))
    default_collection = str(raw_cfg.get("default_collection", "python_fundamentals"))

    root_dir = target_path.parent

    # 3. Validate collection folder existence
    resolved_data_dir = (root_dir / data_dir).resolve()
    if not resolved_data_dir.exists():
        raise ConfigError(f"Data directory does not exist: {resolved_data_dir}")

    resolved_default_collection = (resolved_data_dir / default_collection).resolve()
    if not resolved_default_collection.exists():
        raise ConfigError(
            f"Configured collection directory does not exist: {resolved_default_collection}"
        )

    return AppConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        retry_top_k=retry_top_k,
        strong_score=strong_score,
        weak_score=weak_score,
        embedding_model=embedding_model,
        llm_model=llm_model,
        temperature=temperature,
        llm_timeout_seconds=llm_timeout_seconds,
        ollama_base_url=ollama_base_url,
        data_dir=data_dir,
        indexes_dir=indexes_dir,
        default_collection=default_collection,
        root_dir=root_dir,
    )
