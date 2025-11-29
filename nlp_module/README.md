# NLP Sentiment Module

Pipeline : Telegram → GPT (JSON) → features (4-D) pour l’agent RL.

Fonctions principales (`nlp_module.sentiment_processor`) :
- `listen_telegram_channel(api_token, channel_id) -> str` : récupère le dernier message texte du canal.
- `analyze_sentiment_with_gpt(message, api_key, model="gpt-4o") -> dict` : impose un JSON structuré :
  ```json
  {
    "pair": "BTC/USD",
    "sentiment_score": 0.42,
    "direction": "long",
    "confidence": 0.71,
    "strength": 0.65
  }
  ```
- `sentiment_to_features(json_dict) -> np.ndarray` : renvoie `[score, confidence, direction_num, strength]` (shape (4,), direction_num = +1/-1/0).
- `get_sentiment_features(api_token, channel_id, gpt_api_key, gpt_model="gpt-4o") -> np.ndarray` : pipeline complet (retourne zéro en cas d’erreur).

Config :
- `config/nlp_config.yaml`
  ```
  telegram_api_token: "YOUR_TOKEN"
  telegram_channel_id: "YOUR_CHANNEL"
  gpt_model: "gpt-5.1"
  gpt_api_key: "OPENAI_KEY"
  ```

Features RL :
```
[ sentiment_score, sentiment_confidence, sentiment_direction, sentiment_strength ]
```
Shape = (4,). Sentiment_direction est mappé long=+1, short=-1, neutral=0.

Notes :
- Dépendances optionnelles : `python-telegram-bot`, `openai`.
- En cas d’échec (réseau/JSON), le module renvoie un vecteur de zéros pour ne pas bloquer l’agent.  
