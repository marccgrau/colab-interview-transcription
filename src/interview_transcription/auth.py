import os

from dotenv import load_dotenv

load_dotenv()


def load_secrets(cfg):
    """Load HF token and optional LLM API key from .env file, Colab secrets, or environment.

    Returns (hf_token, llm_api_key). Raises ValueError if HF_TOKEN is missing.
    """
    hf_token = _get_secret("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found. Add it to a .env file, Colab Secrets, or set the env var. "
            "Get a token at https://huggingface.co/settings/tokens"
        )

    llm_api_key = ""
    if cfg.enable_llm_postprocessing:
        secret_name = (
            "ANTHROPIC_API_KEY" if cfg.llm_provider == "anthropic" else "OPENAI_API_KEY"
        )
        llm_api_key = _get_secret(secret_name)
        if not llm_api_key:
            print(f"WARNING: {secret_name} not found. LLM post-processing will be skipped.")

    return hf_token, llm_api_key


def _get_secret(name):
    """Try Colab secrets first, then environment variables (includes .env via dotenv)."""
    try:
        from google.colab import userdata

        return userdata.get(name)
    except ImportError:
        pass
    except Exception:
        pass
    return os.environ.get(name, "")
