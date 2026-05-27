# =============================================================================
#  HR Analytics — API REST FastAPI
#  Auteurs   : Seye Kiné | Bindia Adeline Thiara
#  Encadrant : M. Aidara — UCAO 2025-2026
#  Lancement : uvicorn 6_fastapi:app --reload
#  URL       : http://localhost:8000
#  Swagger   : http://localhost:8000/docs
# =============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import pickle
import uvicorn

# ── Chargement du modèle ──────────────────────────────────────────────────────
with open("mon_modele_rh.pkl", "rb") as f:
    data = pickle.load(f)

modele        = data["cerveau_ia"]
preprocesseur = data["traitement"]
seuil         = data["reglage_seuil"]
FEATURES      = data["features"]
F1            = data["f1"]
AUC           = data["auc"]

# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "HR Turnover Analytics API",
    description = "Prédiction du risque de départ des employés — UCAO 2025-2026",
    version     = "1.0.0",
)

# ── Modèle de données (ce qu'on reçoit) ──────────────────────────────────────
class EmployeInput(BaseModel):
    # Variables numériques
    Age                     : int   = 35
    DailyRate               : int   = 800
    DistanceFromHome        : int   = 10
    HourlyRate              : int   = 60
    MonthlyIncome           : int   = 5000
    MonthlyRate             : int   = 15000
    NumCompaniesWorked      : int   = 2
    PercentSalaryHike       : int   = 12
    StockOptionLevel        : int   = 0
    TotalWorkingYears       : int   = 10
    TrainingTimesLastYear   : int   = 2
    YearsAtCompany          : int   = 4
    YearsInCurrentRole      : int   = 2
    YearsSinceLastPromotion : int   = 1
    YearsWithCurrManager    : int   = 2

    # Variables catégorielles
    BusinessTravel           : str = "Travel_Rarely"
    Department               : str = "Sales"
    Education                : str = "Bachelor"
    EducationField           : str = "Marketing"
    EnvironmentSatisfaction  : str = "High"
    Gender                   : str = "Male"
    JobInvolvement           : str = "High"
    JobLevel                 : str = "Junior Level"
    JobRole                  : str = "Sales Executive"
    JobSatisfaction          : str = "High"
    MaritalStatus            : str = "Single"
    OverTime                 : str = "No"
    PerformanceRating        : str = "Excellent"
    RelationshipSatisfaction : str = "High"
    WorkLifeBalance          : str = "Good"

# ── Fonction de prédiction ────────────────────────────────────────────────────
def predire(employe_dict: dict):
    df_in   = pd.DataFrame([employe_dict])
    X       = preprocesseur.transform(df_in[FEATURES])
    proba   = float(modele.predict_proba(X)[0][1])
    predict = int(proba >= seuil)

    if proba >= 0.70:
        niveau = "🔴 Critique"
        action = "Intervention immédiate — Entretien sous 48h"
    elif proba >= 0.50:
        niveau = "🟠 Élevé"
        action = "Entretien individuel sous 2 semaines"
    elif proba >= seuil:
        niveau = "🟡 Modéré"
        action = "Inclure dans le suivi mensuel RH"
    else:
        niveau = "🟢 Faible"
        action = "Profil stable — Entretiens annuels habituels"

    return {
        "probabilite"      : round(proba, 4),
        "probabilite_pct"  : round(proba * 100, 1),
        "prediction"       : predict,
        "niveau_risque"    : niveau,
        "recommandation"   : action,
        "seuil_utilise"    : seuil,
    }

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/")
def accueil():
    """Page d'accueil de l'API."""
    return {
        "message"     : "HR Turnover Analytics API",
        "description" : "Prédiction du risque de départ des employés",
        "auteurs"     : "Seye Kiné & Bindia Adeline Thiara",
        "encadrant"   : "M. Aidara — UCAO 2025-2026",
        "routes"      : {
            "/"              : "Cette page",
            "/health"        : "Vérifier que l'API fonctionne",
            "/info"          : "Informations sur le modèle",
            "/predict"       : "Prédire pour 1 employé (POST)",
            "/predict/batch" : "Prédire pour plusieurs employés (POST)",
            "/docs"          : "Documentation interactive Swagger",
        }
    }

@app.get("/health")
def health():
    """Vérifier que l'API est opérationnelle."""
    return {"status": " OK", "message": "L'API est opérationnelle"}

@app.get("/info")
def info_modele():
    """Informations sur le modèle utilisé."""
    return {
        "modele"    : "XGBoost (optimisé par GridSearchCV)",
        "f1_score"  : round(F1, 4),
        "auc_roc"   : round(AUC, 4),
        "seuil"     : seuil,
        "features"  : len(FEATURES),
        "dataset"   : "IBM HR Analytics (hr.csv) — 1 470 employés",
    }

@app.post("/predict")
def predict(employe: EmployeInput):
    """
    Prédire le risque de départ pour un employé.

    Envoyer les informations de l'employé et recevoir :
    - La probabilité de départ (0 à 1)
    - Le niveau de risque (Faible / Modéré / Élevé / Critique)
    - Une recommandation RH
    """
    try:
        resultat = predire(employe.dict())
        return resultat
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(employes: list[EmployeInput]):
    """
    Prédire le risque de départ pour plusieurs employés en une fois.
    Maximum 100 employés par requête.
    """
    if len(employes) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 employés par requête"
        )
    try:
        resultats = []
        for i, emp in enumerate(employes):
            res = predire(emp.dict())
            res["index"] = i
            resultats.append(res)

        # Résumé
        n_critique = sum(1 for r in resultats if "Critique" in r["niveau_risque"])
        n_eleve    = sum(1 for r in resultats if "Élevé" in r["niveau_risque"])

        return {
            "nb_employes"  : len(resultats),
            "n_critique"   : n_critique,
            "n_eleve"      : n_eleve,
            "risque_moyen" : round(np.mean([r["probabilite"] for r in resultats]) * 100, 1),
            "resultats"    : resultats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Lancement ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("6_fastapi:app", host="0.0.0.0", port=8000, reload=True)
