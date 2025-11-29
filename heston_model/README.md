# Heston data builder

Ce module construit une surface d'IV simple (ATM/OTM/ITM et maturités 30/60/90/180/365 jours) pour alimenter une calibration de modèle de Heston.

Entrée principale :
- `get_heston_dataset(symbol: str, source: str = "yahoo") -> pd.DataFrame`

Fonctionnement :
- Récupère le dernier prix spot.
- Liste les expirations d'options, choisit la plus proche des maturités cibles.
- Sélectionne les strikes ATM, OTM (+/-10%) et ITM (+/-10% inversé call/put).
- Construit un tableau avec `date, underlying_price, maturity, strike, option_type, implied_volatility` (+ colonne `symbol`).
- Sauvegarde le résultat dans `./data/heston_iv_surface.csv`.

Sources :
- `source="yahoo"` : utilise `DATA.API.YahooFinance` (dépendance optionnelle `yfinance`).
- `source="local"` : lit le spot dans `STATICS/PRICE/<symbol>.csv` (pas d’option chain auto, à compléter selon vos données).

Notes :
- Aucune dépendance n’est ajoutée aux requirements : installez `yfinance` si vous utilisez la source Yahoo.
- Le module réutilise le cache `./data/heston_iv_surface.csv` s’il est daté du jour et couvre toutes les maturités cibles.

## Calibration Heston (Carr–Madan)

Module : `heston_model.calibrate_heston`
- `calibrate_heston_from_df(df, r, q, max_iters, lr, device) -> (params, metrics, df_used)`
- `heston_params_to_features(params) -> np.ndarray shape (5,)` (kappa, theta, sigma, rho, v0)
- `get_heston_features_from_csv(csv_path, r, q, ...) -> (features, metrics)`
- `get_heston_features_from_market_df(df_options, r, q, ...) -> (features, metrics)`

Le module nécessite `heston_torch` (local ou sur PYTHONPATH) pour `HestonParams` et `carr_madan_call_torch`.
