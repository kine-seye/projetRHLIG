# =============================================================================
#  GENERATIVE HR ANALYTICS — PRÉDICTION DU TURNOVER DES EMPLOYÉS
#  Auteurs   : Seye Kiné | Bindia Adeline Thiara
#  Encadrant : M. Aidara — UCAO 2025-2026
# =============================================================================
#  ÉTAPE 10 — API REST AVEC FASTAPI
#
#  Lancement : uvicorn etape10_fastapi:app --reload
#  Swagger   : http://localhost:8000/docs
#
#  Installation : pip install fastapi uvicorn
# =============================================================================

from fastapi           import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic          import BaseModel, Field
from typing            import List
import pickle, numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

# ── Chargement ────────────────────────────────────────────────────────────────
with open("pipeline_data.pkl", "rb") as f:
    pipeline = pickle.load(f)

modele_final  = pipeline["modele_final"]
preprocesseur = pipeline["preprocesseur"]
seuil_opt     = pipeline.get("seuil_optimal", 0.5)
FEATURES      = pipeline["FEATURES"]
nom_final     = pipeline["nom_final"]
eval_f        = pipeline.get("eval_finale", {})
taux          = pipeline["taux"]
n             = pipeline["n"]

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "HR Turnover Analytics API",
    description = """
## API de Prédiction du Turnover des Employés

**Auteurs** : Seye Kiné | Bindia Adeline Thiara — **UCAO 2025-2026**  
**Encadrant** : M. Aidara

Envoyer les données d'un employé → recevoir son score de risque de départ.
    """,
    version = "1.0.0",
)

# ── Schémas ───────────────────────────────────────────────────────────────────
class EmployeInput(BaseModel):
    YearsSinceLastPromotion : int = Field(..., ge=0, le=15, example=4)
    DistanceFromHome        : int = Field(..., ge=1, le=40, example=20)
    JobSatisfaction         : int = Field(..., ge=1, le=4,  example=2)
    WorkLifeBalance         : int = Field(..., ge=1, le=4,  example=2)
    Gender     : str = Field(..., example="Male",           description="Male ou Female")
    Department : str = Field(..., example="Sales",          description="Finance, HR, IT, Marketing, R&D, Sales")
    JobRole    : str = Field(..., example="Sales Executive", description="Analyst, Consultant, Engineer, HR Specialist, Manager, Sales Executive, Technician")
    OverTime   : str = Field(..., example="Yes",            description="Yes ou No")

class PredictionOutput(BaseModel):
    employe_id         : int
    probabilite_depart : float
    prediction         : str
    niveau_risque      : str
    seuil_utilise      : float
    modele             : str
    recommandation     : str

class BatchInput(BaseModel):
    employes : List[EmployeInput]

class BatchOutput(BaseModel):
    nb_employes : int
    nb_a_risque : int
    taux_risque : float
    predictions : List[PredictionOutput]

# ── Fonction prédiction ───────────────────────────────────────────────────────
def predire(employe_dict: dict, eid: int) -> PredictionOutput:
    df_in  = pd.DataFrame([employe_dict])
    X_proc = preprocesseur.transform(df_in)
    proba  = float(modele_final.predict_proba(X_proc)[0][1])

    if proba >= 0.75:
        niveau = "Critique ⚠️"
        recomm = "Action immédiate. Entretien de rétention sous 48h."
    elif proba >= 0.60:
        niveau = "Élevé 🔴"
        recomm = "Entretien individuel sous 2 semaines recommandé."
    elif proba >= seuil_opt:
        niveau = "Modéré 🟡"
        recomm = "Suivi mensuel. Vérifier satisfaction et WLB."
    else:
        niveau = "Faible 🟢"
        recomm = "Profil stable. Maintenir les bonnes conditions."

    return PredictionOutput(
        employe_id         = eid,
        probabilite_depart = round(proba, 4),
        prediction         = "Risque de départ" if proba >= seuil_opt else "Risque faible",
        niveau_risque      = niveau,
        seuil_utilise      = seuil_opt,
        modele             = nom_final,
        recommandation     = recomm,
    )

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["Général"])
async def accueil():
    return f"""<html><head><title>HR Turnover API</title>
    <style>body{{font-family:Segoe UI;background:#0F1923;color:#E8F0FE;
    display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;}}
    .card{{background:#1A2535;border:1px solid #3A4F6A;border-radius:16px;
    padding:40px 48px;max-width:600px;text-align:center;}}
    h1{{color:#4A9EF5;font-size:28px;}} p{{color:#8FA3BF;}}
    .btn{{display:inline-block;background:#4A9EF5;color:white;padding:10px 24px;
    border-radius:8px;text-decoration:none;font-weight:700;margin:6px;}}
    .m{{background:#243044;border-radius:8px;padding:10px 18px;margin:6px;display:inline-block;}}
    .mv{{font-size:22px;font-weight:800;color:#FFD166;}}
    .ml{{font-size:11px;color:#8FA3BF;}}
    </style></head><body><div class="card">
    <h1>👥 HR Turnover Analytics API</h1>
    <p>Modèle : <strong>{nom_final}</strong></p>
    <p style="font-size:12px;">Seye Kiné | Bindia Adeline Thiara — UCAO 2025-2026 | M. Aidara</p>
    <br>
    <div class="m"><div class="mv">{eval_f.get('f1',0):.4f}</div><div class="ml">F1-Score</div></div>
    <div class="m"><div class="mv">{eval_f.get('auc',0):.4f}</div><div class="ml">AUC-ROC</div></div>
    <div class="m"><div class="mv">{seuil_opt:.2f}</div><div class="ml">Seuil optimal</div></div>
    <br><br>
    <a class="btn" href="/docs">📖 Swagger UI</a>
    <a class="btn" href="/redoc">📋 ReDoc</a>
    <a class="btn" href="/info">ℹ️ Infos</a>
    </div></body></html>"""

@app.get("/health", tags=["Général"])
async def health():
    return {"status": "OK ✅", "modele": nom_final, "seuil": seuil_opt}

@app.get("/info", tags=["Général"])
async def info():
    return {
        "modele": nom_final, "features": FEATURES,
        "seuil_optimal": seuil_opt,
        "metriques": {
            "f1":  round(eval_f.get("f1",  0), 4),
            "auc": round(eval_f.get("auc", 0), 4),
            "acc": round(eval_f.get("acc", 0), 4),
        },
        "dataset": {"taille": int(n), "taux_attrition": round(taux, 2)},
        "auteurs": "Seye Kiné | Bindia Adeline Thiara — UCAO 2025-2026",
    }

@app.post("/predict", response_model=PredictionOutput,
          summary="Prédire le risque d'un employé", tags=["Prédiction"])
async def predict(emp: EmployeInput):
    """Retourne la probabilité de départ + niveau de risque + recommandation."""
    try:
        return predire({
            "YearsSinceLastPromotion": emp.YearsSinceLastPromotion,
            "DistanceFromHome": emp.DistanceFromHome,
            "JobSatisfaction": emp.JobSatisfaction,
            "WorkLifeBalance": emp.WorkLifeBalance,
            "Gender": emp.Gender, "Department": emp.Department,
            "JobRole": emp.JobRole, "OverTime": emp.OverTime,
        }, eid=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchOutput,
          summary="Prédire le risque de plusieurs employés", tags=["Prédiction"])
async def predict_batch(batch: BatchInput):
    """Analyse une liste d'employés et retourne les statistiques globales."""
    try:
        preds = []
        for i, emp in enumerate(batch.employes):
            preds.append(predire({
                "YearsSinceLastPromotion": emp.YearsSinceLastPromotion,
                "DistanceFromHome": emp.DistanceFromHome,
                "JobSatisfaction": emp.JobSatisfaction,
                "WorkLifeBalance": emp.WorkLifeBalance,
                "Gender": emp.Gender, "Department": emp.Department,
                "JobRole": emp.JobRole, "OverTime": emp.OverTime,
            }, eid=i+1))
        nb_risque = sum(1 for p in preds if p.probabilite_depart >= seuil_opt)
        return BatchOutput(
            nb_employes=len(preds), nb_a_risque=nb_risque,
            taux_risque=round(nb_risque/len(preds)*100, 1),
            predictions=preds,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Lancement ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "═"*55)
    print("  ▶  HR Turnover Analytics — FastAPI")
    print(f"  ▶  Modèle : {nom_final}")
    print("  ▶  URL    : http://localhost:8000")
    print("  ▶  Docs   : http://localhost:8000/docs")
    print("═"*55 + "\n")
    uvicorn.run("etape10_fastapi:app", host="0.0.0.0", port=8000, reload=True)
