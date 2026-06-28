# Projet DT SC — Marketing ROI Dashboard

## 📌 Description
Ce projet est une application d’analyse et de simulation du ROI marketing pour des campagnes multicanal.
Il combine un tableau de bord interactif Streamlit avec une API FastAPI pour l’inférence de modèles de prédiction.

L’objectif est de proposer :
- une exploration des données marketing,
- une comparaison de modèles de machine learning,
- un simulateur de budget marketing,
- une analyse d’impact marginal et d’interprétabilité.

## 🧩 Structure du projet

- `dashboard.py` : application Streamlit principale.
- `api/api.py` : API FastAPI pour servir les prédictions et vérifier l’état du service.
- `data/Dummy_Data_HSS.csv` : jeu de données marketing utilisé pour l’analyse.
- `models/` : répertoire des modèles entraînés stockés en `joblib`.
- `models_output/` : résultats de comparaison des modèles et d’impact marginal.
- `pipeline/` : éléments de pipeline tels que `feature_names.json` et les jeux de données préparer.
- `notebooks/` : notebooks Jupyter pour l’analyse exploratoire, le preprocessing, la modélisation et SHAP.
- `shap_output/` : sortie dédiée aux résultats SHAP.

## 🛠️ Installation

1. Créez et activez un environnement virtuel Python :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

> Note : le projet utilise `tensorflow-macos` et `tensorflow-metal` pour macOS, ainsi que Streamlit et FastAPI.

## 🚀 Utilisation

### 1. Lancer l’API

Depuis la racine du projet :

```bash
uvicorn api.api:app --reload
```

L’API sera disponible sur `http://127.0.0.1:8000`.

### 2. Lancer le dashboard Streamlit

Depuis la racine du projet :

```bash
streamlit run dashboard.py
```

### 3. Endpoints utiles

- `GET /health` : vérifie que l’API est active et affiche les modèles disponibles.
- `POST /predict` : envoie un scénario budgétaire pour prédire les ventes et le ROI.
- `GET /model-info` : récupère des informations sur le modèle déployé.

## 📊 Fonctionnalités principales

- Visualisation des budgets marketing par canal
- Calcul du ROI et indicateurs clés
- Comparaison entre plusieurs modèles de machine learning
- Simulation de scénarios budgétaires
- Analyse d’impact marginal
- Support de requêtes d’inférence via API

## 🧪 Notebooks

Les notebooks permettent de suivre toutes les étapes du projet :

- `0_requirements.ipynb` : exigences et préparation du projet
- `1_eda_marketing.ipynb` : exploration des données
- `2_preprocessing_marketing.ipynb` : préparation et nettoyage des données
- `3_modeling_marketing.ipynb` : entraînement et comparaison des modèles
- `4_shap_marketing.ipynb` : interprétabilité avec SHAP

## 🔧 Conseils

- Vérifiez que les modèles existent dans `models/` avant de lancer l’API.
- Si des fichiers `*.pkl` manquent, il faudra réentraîner les modèles à partir des notebooks.
- Pour un affichage complet de l’application, lancez d’abord l’API puis le dashboard.

## 📁 À propos
Ce projet est réalisé dans le cadre d’un TP/Projet Data Science, avec un focus sur l’optimisation du ROI marketing et la mise en production d’un modèle via une API.
