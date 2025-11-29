# Streamlit UI (modular)

- Launch: `streamlit run streamlit_app/app.py` from repo root (after `pip install -r requirements.txt`).
- Config: edit NLP/Heston/RL in the **Configuration** tab (forms write to `config/*.yaml`). Sidebar lets you focus a module.
- Run: use sidebar to choose mode (`dry-run` or `real-run`) then click **Run pipeline**. Stages log to `logs/nlp.log`, `logs/heston.log`, `logs/rl.log`.
- Logs: **Execution Logs** tab shows live tails for each module; red highlights errors. Clear buttons wipe a module log.
- Files: new configs live under `config/`, UI helpers under `streamlit_app/`, outputs (mock features) in `STATICS/NLP` and `STATICS/Heston`.
- Notes: `dry-run` avoids network/heavy compute (mock GPT, mock Heston, short RL loop). `real-run` uses live modules; ensure API keys/CSV paths are set and CUDA if GPU is chosen.
