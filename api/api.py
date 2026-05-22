"""
API REST — Optimisation du ROI Marketing
Endpoints : /health, /predict, /model-info
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Literal
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

root_dir = Path.cwd().parent

# ─────────────────────────────────────────
# Initialisation de l'app
# ─────────────────────────────────────────
app = FastAPI(
    title="Marketing ROI — API d'inférence",
    description="""
    API REST permettant de prédire les ventes et estimer le ROI
    à partir d'une combinaison de budgets marketing multicanal.

    ## Endpoints
    - **GET /health** — Vérifie que le service est actif
    - **POST /predict** — Prédit les ventes pour un scénario budgétaire
    - **GET /model-info** — Informations sur le modèle déployé
    """,
    version="1.0.0"
)

# ─────────────────────────────────────────
# Chargement du modèle et preprocessor
# ─────────────────────────────────────────
MODELS = {}
MODEL_FILES = {
    'Régression Linéaire': root_dir / 'models' / 'model_linear_regression.pkl',
    'Ridge':               root_dir / 'models' / 'model_ridge.pkl',
    'Random Forest':       root_dir / 'models' / 'model_random_forest.pkl',
    'XGBoost':             root_dir / 'models' / 'model_xgboost.pkl',
    'MLP':                 root_dir / 'models' / 'model_mlp.pkl',
}
for name, path in MODEL_FILES.items():
    try:
        MODELS[name] = joblib.load(path)
        print(f'✅ Modèle chargé : {name}')
    except Exception as e:
        print(f'⚠️  Modèle non trouvé : {name} ({e})')

MODEL_LOADED = len(MODELS) > 0
preprocessor = None
feature_names = None

try:
    preprocessor = joblib.load(root_dir / 'pipeline' / 'preprocessor.pkl')

    with open(root_dir / 'pipeline' / 'feature_names.json') as f:
        feature_names = json.load(f)

    MODEL_LOADED = True
    print(f'✅ Preprocessor chargé')
    print(f'✅ Features : {feature_names}')

except Exception as e:
    print(f'❌ Erreur de chargement : {e}')


# ─────────────────────────────────────────
# Schémas de données (Pydantic)
# ─────────────────────────────────────────
class PredictRequest(BaseModel):
    """Données d'entrée pour la prédiction."""
    TV: float = Field(
        ...,
        gt=0,
        description="Budget TV en millions d'euros",
        example=50.0
    )
    Radio: float = Field(
        ...,
        gt=0,
        description="Budget Radio en millions d'euros",
        example=20.0
    )
    Social_Media: float = Field(
        ...,
        gt=0,
        description="Budget Social Media en millions d'euros",
        example=5.0
    )
    Influencer: Literal['Mega', 'Macro', 'Micro', 'Nano'] = Field(
        ...,
        description="Type d'influenceur mobilisé",
        example="Mega"
    )
    model: Literal['Régression Linéaire', 'Ridge', 'Random Forest', 'XGBoost', 'MLP'] = Field(
        default='XGBoost',
        description="Modèle à utiliser pour la prédiction",
        example="XGBoost"
    )

    class Config:
        schema_extra = {
            "example": {
                "TV": 80.0,
                "Radio": 30.0,
                "Social_Media": 5.0,
                "Influencer": "Mega",
                "model": "XGBoost"
            }
        }

    @validator('TV')
    def tv_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Le budget TV doit être positif')
        if v > 10000:
            raise ValueError('Le budget TV semble anormalement élevé (> 10 000 M€)')
        return round(v, 4)

    @validator('Radio')
    def radio_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Le budget Radio doit être positif')
        if v > 10000:
            raise ValueError('Le budget Radio semble anormalement élevé (> 10 000 M€)')
        return round(v, 4)

    @validator('Social_Media')
    def social_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Le budget Social Media doit être positif')
        if v > 10000:
            raise ValueError('Le budget Social Media semble anormalement élevé (> 10 000 M€)')
        return round(v, 4)

    class Config:
        schema_extra = {
            "example": {
                "TV": 80.0,
                "Radio": 30.0,
                "Social_Media": 5.0,
                "Influencer": "Mega"
            }
        }


class PredictResponse(BaseModel):
    """Réponse de la prédiction."""
    status: str
    prediction: dict
    input_received: dict
    model_used: str
    timestamp: str


class HealthResponse(BaseModel):
    """Réponse du health check."""
    status: str
    model_loaded: bool
    available_models: list[str]
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Informations sur le modèle."""
    model_name: str
    model_type: str
    features: list
    influencer_types: list
    description: str
    version: str


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

# ── GET /health ──────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérification de l'état du service",
    tags=["Service"]
)
def health_check():
    """
    Vérifie que l'API est active et que le modèle est correctement chargé.

    Retourne :
    - **status** : 'ok' si tout fonctionne, 'degraded' sinon
    - **model_loaded** : True si le modèle est chargé
    - **model_name** : nom du modèle déployé
    """
    return HealthResponse(
    status="ok" if MODEL_LOADED else "degraded",
    model_loaded=MODEL_LOADED,
    available_models=list(MODELS.keys()),
    timestamp=datetime.now().isoformat()
)


# ── POST /predict ─────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Prédiction"])
def predict(request: PredictRequest):

    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Aucun modèle disponible.")

    # Vérification que le modèle demandé est disponible
    if request.model not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle '{request.model}' non disponible. "
                   f"Modèles disponibles : {list(MODELS.keys())}"
        )

    try:
        selected_model = MODELS[request.model]

        input_df = pd.DataFrame({
            'TV':           [request.TV],
            'Radio':        [request.Radio],
            'Social Media': [request.Social_Media],
            'Influencer':   [request.Influencer]
        })

        input_processed     = preprocessor.transform(input_df)
        prediction_value    = float(selected_model.predict(input_processed)[0])
        prediction_value    = round(max(0, prediction_value), 4)
        total_budget        = request.TV + request.Radio + request.Social_Media
        roi_estimated       = round(prediction_value / total_budget, 4)

        return PredictResponse(
            status="success",
            prediction={
                "sales_predicted_M€": prediction_value,
                "roi_estimated":       roi_estimated,
                "total_budget_M€":     round(total_budget, 4),
                "revenue_per_euro":    round(roi_estimated, 4)
            },
            input_received={
                "TV":           request.TV,
                "Radio":        request.Radio,
                "Social_Media": request.Social_Media,
                "Influencer":   request.Influencer,
                "model":        request.model
            },
            model_used=request.model,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

# ── GET /model-info ───────────────────────
@app.get("/model-info", tags=["Service"])
def model_info():
    return {
        "models_available": list(MODELS.keys()),
        "default_model":    "XGBoost",
        "features":         feature_names,
        "influencer_types": ["Mega", "Macro", "Micro", "Nano"],
        "version":          "1.0.0"
    }


# ─────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
