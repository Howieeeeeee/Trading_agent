from __future__ import annotations

import importlib
import os
from typing import Any

from ..config import LLMConfig



def build_llm(config: LLMConfig):
    provider = config.provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.api_key_env:
            kwargs["api_key"] = os.getenv(config.api_key_env)
        if config.extra:
            kwargs.update(config.extra)
        return ChatOpenAI(**kwargs)

    if provider == "rule_based":
        from .rule_based import RuleBasedChatModel

        return RuleBasedChatModel()

    if provider == "custom":
        # expected format in llm.extra:
        # class_path: "pkg.module:ClassName"
        # init_kwargs: {...}
        if not config.extra or "class_path" not in config.extra:
            raise ValueError("custom provider requires llm.extra.class_path")
        class_path = config.extra["class_path"]
        init_kwargs = dict(config.extra.get("init_kwargs", {}))
        cls = _load_class(class_path)
        return cls(**init_kwargs)

    raise ValueError(f"Unsupported provider: {config.provider}")


def _load_class(path: str):
    if ":" not in path:
        raise ValueError(f"Invalid class path: {path}. expected module.path:ClassName")
    module_name, class_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
