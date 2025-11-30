# Setup, tests et Streamlit

Guide rapide depuis un clone git pour installer l'environnement, lancer les tests et ouvrir l'UI Streamlit.

## Prerequis
- Windows + PowerShell
- Miniconda/Anaconda avec `conda` dans le PATH
- GPU Nvidia (optionnel) si vous voulez utiliser CUDA

## Installation de l'environnement
1. Cloner puis se placer dans le dossier du projet.
2. Lancer le script (creer ou mettre a jour l'env `rl-pytorch-cuda`, puis verifier Torch/CUDA) :
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
   ```
3. Activer l'environnement :
   ```powershell
   conda activate rl-pytorch-cuda
   ```
4. Optionnel (deja fait par le script) : relancer la verif CUDA si besoin.
   ```powershell
   python verify_pytorch_cuda.py
   ```

## Lancer les tests
Executer depuis la racine du repo, avec l'env actif :
```powershell
pytest
```

## Lancer l'UI Streamlit
Toujours depuis la racine, apres activation de l'env :
```powershell
streamlit run streamlit_app/app.py
```

- Pour l'ancienne demo simple, vous pouvez aussi lancer : `streamlit run streamlit_app.py`.
- Ajustez les chemins de donnees/config dans l'UI si necessaire (dossiers `CONFIG`, `STATICS`, etc.).
