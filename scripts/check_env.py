"""Environment and dependency diagnostic script for Hallucination-Aware GenAI Assistant.

Checks:
1. Python version (>= 3.10 required).
2. Reachability of Ollama service at the configured endpoint (default: http://localhost:11434).
3. Availability of the configured Ollama model (e.g. llama3.2:3b).
4. Loading of the configured Sentence Transformer embedding model (e.g. all-MiniLM-L6-v2).
"""

import sys
from pathlib import Path


def load_config() -> dict:
    """Load configuration from config.yaml or return defaults."""
    default_config = {
        "llm_model": "llama3.2:3b",
        "embedding_model": "all-MiniLM-L6-v2",
        "ollama_base_url": "http://localhost:11434",
    }
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"[-] config.yaml not found at {config_path}. Using internal defaults.")
        return default_config

    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            for k, v in default_config.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
    except ImportError:
        print("[-] pyyaml not installed. Using internal defaults.")
        return default_config
    except Exception as e:
        print(f"[-] Error loading config.yaml: {e}. Using internal defaults.")
        return default_config


def check_python_version() -> bool:
    """Check that Python version is at least 3.10."""
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    print(f"\n[1/4] Checking Python version...")
    print(f"      Detected: Python {version_str}")
    if (major, minor) < (3, 10):
        print(f"  [X] FAIL: Python 3.10 or newer is required.")
        print(f"      Instruction: Install Python 3.10+ from https://www.python.org/")
        return False
    print(f"  [+] PASS: Python version {version_str} meets requirements (>= 3.10).")
    return True


def check_ollama(base_url: str, model_name: str) -> tuple[bool, bool]:
    """Check if Ollama is reachable and if the configured model exists."""
    print(f"\n[2/4] Checking Ollama reachability at {base_url}...")
    try:
        import requests
    except ImportError:
        print("  [X] FAIL: 'requests' library is not installed.")
        print("      Instruction: Run 'pip install requests'.")
        return False, False

    tags_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = requests.get(tags_url, timeout=5)
        if response.status_code != 200:
            print(f"  [X] FAIL: Ollama returned HTTP status {response.status_code}.")
            print(f"      Instruction: Verify Ollama service health.")
            return False, False
        print(f"  [+] PASS: Ollama service is reachable at {base_url}.")
    except requests.exceptions.ConnectionError:
        print(f"  [X] FAIL: Unable to connect to Ollama at {base_url}.")
        print("      Instruction: Start the Ollama server:")
        print("                   1. Open terminal and run: ollama serve")
        print("                   2. Or launch the Ollama desktop application.")
        return False, False
    except Exception as e:
        print(f"  [X] FAIL: Error connecting to Ollama: {e}")
        return False, False

    print(f"\n[3/4] Checking configured Ollama model: '{model_name}'...")
    try:
        data = response.json()
        models = data.get("models", [])
        installed_names = [m.get("name", "") for m in models]

        # Normalization: match 'llama3.2:3b' or 'llama3.2:3b-instruct' or 'llama3.2:latest' if model_name is llama3.2
        found = False
        target = model_name.lower().strip()
        for inst in installed_names:
            inst_clean = inst.lower().strip()
            if inst_clean == target:
                found = True
                break
            # Match without tag if user specified e.g. llama3.2:3b and tag matches
            if inst_clean.split(":")[0] == target.split(":")[0]:
                if target in inst_clean or inst_clean in target:
                    found = True
                    break

        if found:
            print(f"  [+] PASS: Model '{model_name}' is available in Ollama.")
            return True, True
        else:
            print(f"  [X] FAIL: Model '{model_name}' not found in Ollama.")
            if installed_names:
                print(f"      Currently available models: {', '.join(installed_names)}")
            else:
                print("      No models currently installed in Ollama.")
            print(f"      Instruction: Download the model by running:")
            print(f"                   ollama pull {model_name}")
            return True, False
    except Exception as e:
        print(f"  [X] FAIL: Could not parse models from Ollama response: {e}")
        return True, False


def check_embedding_model(model_name: str) -> bool:
    """Check if the configured Sentence Transformer embedding model can be loaded."""
    print(f"\n[4/4] Checking embedding model: '{model_name}'...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  [X] FAIL: 'sentence-transformers' library is not installed.")
        print("      Instruction: Run 'pip install sentence-transformers'.")
        return False

    try:
        print("      Loading embedding model (this downloads weights on first run)...")
        model = SentenceTransformer(model_name)
        embedding = model.encode("Test embedding generation", convert_to_numpy=True)
        print(f"  [+] PASS: Successfully loaded '{model_name}' (vector dim: {len(embedding)}).")
        return True
    except Exception as e:
        print(f"  [X] FAIL: Failed to load embedding model '{model_name}': {e}")
        print("      Instruction: Ensure an internet connection is available to download the model,")
        print("                   or check that the model name is valid on Hugging Face.")
        return False


def main():
    print("=" * 60)
    print("  Hallucination-Aware GenAI Assistant - Environment Check")
    print("=" * 60)

    cfg = load_config()
    ollama_url = cfg.get("ollama_base_url", "http://localhost:11434")
    llm_model = cfg.get("llm_model", "llama3.2:3b")
    embedding_model = cfg.get("embedding_model", "all-MiniLM-L6-v2")

    print(f"Configuration loaded:")
    print(f"  - Ollama Base URL: {ollama_url}")
    print(f"  - LLM Model:       {llm_model}")
    print(f"  - Embedding Model: {embedding_model}")

    results = {}
    results["python"] = check_python_version()
    ollama_conn, model_ok = check_ollama(ollama_url, llm_model)
    results["ollama_conn"] = ollama_conn
    results["ollama_model"] = model_ok
    results["embedding"] = check_embedding_model(embedding_model)

    print("\n" + "=" * 60)
    print("  Summary:")
    print("=" * 60)
    print(f"  1. Python version (>= 3.10):        {'[PASS]' if results['python'] else '[FAIL]'}")
    print(f"  2. Ollama reachable:                {'[PASS]' if results['ollama_conn'] else '[FAIL]'}")
    print(f"  3. Ollama model '{llm_model}':       {'[PASS]' if results['ollama_model'] else '[FAIL]'}")
    print(f"  4. Embedding model loaded:          {'[PASS]' if results['embedding'] else '[FAIL]'}")
    print("=" * 60)

    all_passed = all(results.values())
    if all_passed:
        print("\nAll environment checks PASSED successfully.")
        sys.exit(0)
    else:
        print("\nSome environment checks did not pass. Review instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
