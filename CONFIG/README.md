# Pair-Trading-Reinforcement-Learning


## Si tu veux relancer plus tard dans une nouvelle console :

## Ouvre PowerShell, configure CUDA 11.2 pour la session :

deactivate  # si besoin
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"
conda activate rlpair-gpu-py310
./test-app-gpu.ps1 -SkipInstall -UseCurrentEnv -PythonExe "python"








<p align="center">
  <img width="600" src="Structure.PNG">
</p>
<p align="justify">

A TensorFlow implemention in Reinforcement Learning and Pairs Trading. The current status of the project covers implementation of RL in cointegration pair trading based on 1-minute stock market data. For the Reinforcement Learning here we use the N-armed bandit approach. The code is expandable so you can plug any strategies, data API or machine learning algorithms into the tool if you follow the style.

## Data Source
* [Tiingo](https://www.tiingo.com/) - A financial research platform that provides data including news, fundamentals and prices. We can extract the intraday stock market data through its REST IEX API that retrieves TOPS data (top of book, last sale data and top bid and ask quotes) from the IEX Exchange.

## Examples
See the folder EXAMPLE for more detail. Please initiate your own virtual environment before running the code.

## Disclaimer
The article and the relevant codes and content are purely informative and none of the information provided constitutes any recommendation regarding any security, transaction or investment strategy for any specific person. The implementation described in the article could be risky and the market condition could be volatile and differ from the period covered above. All trading strategies and tools are implemented at the users’ own risk.





$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
$env:PATH="$env:CUDA_PATH\bin;$env:CUDA_PATH\libnvvp;$env:PATH"
nvcc --version  # doit afficher 11.2


python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"



















Voici un pas-à-pas “from scratch” pour un env GPU fonctionnel (Windows, RTX 4060).

Pré-requis système

Driver NVIDIA déjà OK (576.57).
Installe CUDA Toolkit 11.2 (Windows).
Installe cuDNN 8.1 pour CUDA 11.x, copie les dossiers bin/include/lib du zip dans C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2.
Mets CUDA 11.2 en tête du PATH dans ta session PowerShell :

$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2"
$env:PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp;$env:PATH"
nvcc --version   # doit afficher 11.2 (et non 12.9)



Création d’un nouvel env propre

Crée un env Python 3.10 :
conda create -n rlpair-gpu-py310 python=3.10 -y
conda activate rlpair-gpu-py310
python -m pip install --upgrade pip
Installe TensorFlow GPU compatible Windows/CUDA 11.2 :
python -m pip install "tensorflow==2.10.1"
Installe le reste des deps (versions compatibles py3.10 & TF 2.10) :
python -m pip install pandas==1.5.3 numpy==1.23.5 scipy==1.10.1 scikit-learn==1.2.2 matplotlib==3.7.3 aiohttp==3.9.5 statsmodels==0.14.2 PyYAML==6.0.1 seaborn==0.12.2 fastcluster==1.2.6


(ne pas relancer pip install -r requirements-gpu.txt brut, il tirerait un TF trop récent/CPU-only).


Vérification GPU TensorFlo

python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
Tu dois voir au moins une entrée GPU.




Lancer le test avec le bon Python
Depuis le repo :




./test-app-gpu.ps1 -SkipInstall -PythonExe "<chemin-vers-env-conda>\python.exe"
(ou active l’env puis -PythonExe "python").




Si tu veux, je peux ajouter un flag “fallback CPU” dans le script GPU, mais le chemin ci-dessus te donne un env GPU propre.















