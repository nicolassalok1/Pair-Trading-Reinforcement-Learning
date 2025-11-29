"""
Telegram-to-GPT sentiment pipeline for RL feature augmentation.

Functions:
    listen_telegram_channel(api_token, channel_id) -> str
    analyze_sentiment_with_gpt(message: str, api_key: str, model: str) -> dict
    sentiment_to_features(json_dict: dict) -> np.ndarray
    get_sentiment_features(...) -> np.ndarray
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"pair", "sentiment_score", "direction", "confidence", "strength"}


def listen_telegram_channel(api_token: str, channel_id: str, timeout: int = 10) -> str:
    """
    Fetch the latest text message from a Telegram channel using a bot token.
    """
    try:
        from telegram import Bot
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "python-telegram-bot is required for Telegram listening. "
            "Install with `pip install python-telegram-bot`."
        ) from exc

    bot = Bot(token=api_token)
    updates = bot.get_updates(timeout=timeout)
    if not updates:
        raise RuntimeError(f"No messages found for channel {channel_id}.")

    for update in reversed(updates):
        msg = update.message or update.channel_post
        if msg and str(msg.chat_id) == str(channel_id):
            text = (msg.text or msg.caption or "").strip()
            if not text:
                continue
            logger.info("Fetched Telegram message from channel %s", channel_id)
            return text

    raise RuntimeError(f"No text message found for channel {channel_id}.")


def analyze_sentiment_with_gpt(message: str, api_key: str, model: str = "gpt-4o") -> Dict:
    """
    Send a message to OpenAI GPT and return a structured sentiment dict.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "openai package is required for GPT analysis. Install with `pip install openai`."
        ) from exc

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a trading sentiment parser. "
        "Respond ONLY with valid JSON matching keys: "
        'pair, sentiment_score, direction, confidence, strength. '
        "direction must be one of: long, short, neutral."
    )
    user_prompt = (
        f"Message:\n{message}\n\n"
        "Return JSON exactly like:\n"
        '{"pair": "BTC/USD", "sentiment_score": 0.42, "direction": "long", '
        '"confidence": 0.71, "strength": 0.65}'
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    payload = json.loads(content)
    if not REQUIRED_FIELDS.issubset(payload.keys()):
        raise ValueError(f"GPT response missing required fields: {payload}")
    return payload


def _direction_to_numeric(direction: str) -> int:
    direction = (direction or "").strip().lower()
    if direction == "long":
        return 1
    if direction == "short":
        return -1
    return 0


def sentiment_to_features(json_dict: Dict) -> np.ndarray:
    """
    Convert GPT JSON to a numeric feature vector of shape (4,):
        [sentiment_score, confidence, direction_numeric, strength]
    """
    score = float(json_dict.get("sentiment_score", 0.0))
    confidence = float(json_dict.get("confidence", 0.0))
    strength = float(json_dict.get("strength", 0.0))
    direction_num = float(_direction_to_numeric(json_dict.get("direction", "neutral")))
    return np.array([score, confidence, direction_num, strength], dtype=np.float32)


def get_sentiment_features(
    api_token: str,
    channel_id: str,
    gpt_api_key: str,
    gpt_model: str = "gpt-4o",
) -> np.ndarray:
    """
    Full pipeline: Telegram -> GPT -> features.
    """
    try:
        raw_msg = listen_telegram_channel(api_token, channel_id)
        parsed = analyze_sentiment_with_gpt(raw_msg, api_key=gpt_api_key, model=gpt_model)
        feats = sentiment_to_features(parsed)
        return feats
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.warning("Failed to compute sentiment features: %s", exc)
        return np.zeros(4, dtype=np.float32)


__all__ = [
    "listen_telegram_channel",
    "analyze_sentiment_with_gpt",
    "sentiment_to_features",
    "get_sentiment_features",
]
