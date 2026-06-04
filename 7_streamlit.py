# =============================================================================
#  HR Analytics — Dashboard Streamlit
#  Seye Kiné | Bindia Adeline Thiara | M. Aidara — UCAO 2025-2026
#  streamlit run 7_streamlit.py
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

warnings.filterwarnings("ignore")

# ── DOIT ÊTRE LA TOUTE PREMIÈRE COMMANDE STREAMLIT ────────────────────────────
st.set_page_config(
    page_title="HR Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded")

# ── LISTE DES PAGES (simplifiée — demande M. Aidara) ─────────────────────────
PAGES = ["Accueil", "Exploration", "Prédiction", "GenAI"]

# =============================================================================
# THEME — MODE SOMBRE / CLAIR
# =============================================================================

# ── Initialisation unique du thème ────────────────────────────────────────────
if "mode_sombre" not in st.session_state:
    st.session_state.mode_sombre = True

# ── Toggle DANS la sidebar AVANT tout le reste ────────────────────────────────
# C'est la clé : on lit la valeur du toggle EN PREMIER
# pour que les couleurs soient correctes dès le premier clic
with st.sidebar:
    nouveau_mode = st.toggle(
        "🌓 Mode Sombre",
        value=st.session_state.mode_sombre,
        key="toggle_theme")
    st.session_state.mode_sombre = nouveau_mode

# ── Couleurs définies APRÈS lecture du toggle ─────────────────────────────────
if st.session_state.mode_sombre:
    FOC = "#0F1923"; CAC = "#1A2535"; GRC = "#243044"
    TXC = "#E8F0FE"; T2C = "#8FA3BF"; BOC = "#3A4F6A"; GIC = "#2A3A50"
else:
    FOC = "#F8FAFC"; CAC = "#FFFFFF"; GRC = "#F1F5F9"
    TXC = "#0F172A"; T2C = "#475569"; BOC = "#CBD5E0"; GIC = "#E2E8F0"

# ── Couleurs fixes (accents — ne changent jamais) ─────────────────────────────
VC = "#00C896"; RC = "#FF4B6E"; BC = "#4A9EF5"; OC = "#FFD166"
PC = "#9B72F5"; OGC = "#FF8C42"

# ── Paramètres Plotly ─────────────────────────────────────────────────────────
LAY = dict(
    paper_bgcolor=CAC,
    plot_bgcolor=GRC,
    font=dict(color=TXC, family="Inter,sans-serif")
    legend=dict(
        bgcolor=CAC,
        bordercolor=BOC,
        borderwidth=1,
        font=dict(color=TXC)   # ← Ajouter cette ligne
    ))

def ax(t=""):
    return dict(
        title=t,
        gridcolor=GIC,
        showgrid=True,
        zeroline=False,
        tickfont=dict(color=T2C))

# ── CSS Global — appliqué avec les bonnes couleurs du thème ───────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
* {{ font-family: 'Inter', sans-serif !important; }}

/* Correction espace blanc en haut */
.block-container {{
    padding-top: 1rem !important;
    max-width: 100% !important;
    padding-bottom: 2rem !important;
}}

/* Fond global */
.stApp {{
    background-color: {FOC} !important;
    color: {TXC} !important;
}}
/* Supprime l'espace du header Streamlit */
header[data-testid="stHeader"] {{
    height: 0rem !important;
    min-height: 0rem !important;
}}

/* Supprime la barre de défilement horizontale */
.main .block-container {{
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}
/* Header */
header[data-testid="stHeader"] {{
    background-color: {FOC} !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {CAC} !important;
    border-right: 1px solid {BOC} !important;
}}

/* Tous les textes de la sidebar */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {{
    color: {TXC} !important;
}}

/* Radio buttons — noms des pages */
.stRadio label span,
[data-testid="stRadioLabel"] p,
.stRadio [data-baseweb="radio"] label {{
    color: {TXC} !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}}

/* Toggle switch label */
.stToggle label p,
.stToggle span {{
    color: {TXC} !important;
}}

/* Onglets */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {CAC} !important;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {BOC};
}}
.stTabs [data-baseweb="tab"] {{ color: {T2C} !important; border-radius: 8px !important; }}
.stTabs [aria-selected="true"] {{ background-color: {GRC} !important; color: {TXC} !important; }}

/* Selectbox / Inputs */
.stSelectbox > div > div,
.stTextInput > div > div,
.stMultiSelect > div > div {{
    background-color: {GRC} !important;
    border: 1px solid {BOC} !important;
    border-radius: 8px !important;
    color: {TXC} !important;
}}
div[data-baseweb="select"] div,
.stSelectbox span,
input {{
    color: {TXC} !important;
    background-color: {GRC} !important;
}}

/* Dropdown liste déroulante */
ul[data-baseweb="menu"],
li[data-baseweb="menu-item"] {{
    background-color: {CAC} !important;
    color: {TXC} !important;
}}

/* Flèches SVG */
svg {{ fill: {T2C} !important; }}

/* Labels au-dessus des widgets */
div[data-testid="stWidgetLabel"] p {{
    color: {TXC} !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}}

/* Paragraphes et textes généraux */
p, span, div, li {{
    color: {TXC};
}}

/* Titres */
h1, h2, h3, h4, h5, h6 {{ color: {TXC} !important; }}
footer {{ visibility: hidden; }}

/* Bouton primaire */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {BC}, {PC}) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
}}

/* Bouton secondaire */
.stButton > button {{
    background-color: {GRC} !important;
    border: 1px solid {BOC} !important;
    color: {TXC} !important;
    border-radius: 8px !important;
}}

/* Tableaux dataframe */
.stDataFrame {{
    background-color: {CAC} !important;
}}

/* ── Composants cartes HTML ───────────────────────────────────────────── */
.pg {{
    background: {CAC};
    border: 1px solid {BOC};
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}}
.kc {{
    background: {CAC};
    border: 1px solid {BOC};
    border-top: 3px solid var(--c);
    border-radius: 12px;
    padding: 14px 12px;
    text-align: center;
}}
.ib {{
    background: {CAC};
    border: 1px solid {BOC};
    border-left: 4px solid var(--c);
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
}}
.section-title {{
    font-size: 13px;
    font-weight: 800;
    color: {TXC};
    padding: 10px 0 8px;
    border-bottom: 2px solid var(--c);
    margin-bottom: 14px;
}}
.recomm {{
    background: {GRC};
    border-left: 5px solid var(--c);
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    margin: 10px 0;
}}
.pt  {{ font-size: 21px; font-weight: 900; color: {TXC}; margin-bottom: 4px; }}
.ps  {{ font-size: 12px; color: {T2C}; }}
.kv  {{ font-size: 24px; font-weight: 900; color: var(--c); }}
.kl  {{ font-size: 10px; color: {T2C}; text-transform: uppercase; letter-spacing: .8px; margin-top: 4px; font-weight: 600; }}
.it  {{ font-size: 10px; font-weight: 700; color: var(--c); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }}
.iv  {{ font-size: 18px; font-weight: 900; color: {TXC}; margin-bottom: 3px; }}
.ix  {{ font-size: 11px; color: {T2C}; line-height: 1.5; }}

.info-card {{ background: {CAC}; border: 1px solid {BOC}; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; }}
.info-lbl  {{ font-size: 10px; color: {T2C}; margin-bottom: 3px; }}
.info-val  {{ font-size: 13px; font-weight: 700; color: {TXC}; }}

/* Chat GenAI */
.stChatMessage {{
    background-color: {CAC} !important;
    border: 1px solid {BOC} !important;
    border-radius: 12px !important;
    color: {TXC} !important;
}}


/* ── MODE CLAIR — corrections spécifiques ─────────────────── */

/* 1. Radio buttons — points et texte visibles */
.stRadio > div {{
    background-color: transparent !important;
}}
.stRadio label {{
    color: {TXC} !important;
}}
[data-baseweb="radio"] div {{
    border-color: {T2C} !important;
}}

/* 2. Boutons suggestions GenAI — fond visible */
.stButton > button {{
    background-color: {GRC} !important;
    border: 1px solid {BOC} !important;
    color: {TXC} !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}}
.stButton > button:hover {{
    background-color: {BOC} !important;
    border-color: {BC} !important;
}}

/* 3. Zone de saisie texte — fond et texte visibles */
.stTextInput > div > div > input {{
    background-color: {GRC} !important;
    color: {TXC} !important;
    border: 1px solid {BOC} !important;
}}
.stTextInput > div > div > input::placeholder {{
    color: {T2C} !important;
    opacity: 1 !important;
}}

/* 4. Légende Plotly visible en mode clair */
.legend text {{
    fill: {TXC} !important;
}}
.legendtext {{
    fill: {TXC} !important;
}}

/* 5. Markdown texte général */
.stMarkdown p,
.stMarkdown li,
.stMarkdown span {{
    color: {TXC} !important;
}}

/* 6. Dataframe */
.stDataFrame th,
.stDataFrame td {{
    color: {TXC} !important;
    background-color: {CAC} !important;
}}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR — Suite (logo + navigation)
# Le toggle est déjà ajouté ci-dessus, on continue ici avec le reste
# =============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:10px 0 8px;">
      <div style="font-size:40px;">👥</div>
      <div style="font-size:15px;font-weight:800;color:{TXC};">HR Analytics</div>
      <div style="font-size:11px;color:{T2C};margin-top:4px;line-height:1.6;">
        Seye Kiné | Bindia Adeline Thiara<br>
        <span style="color:{OC};font-weight:600;">M. Aidara</span> — UCAO 2025-2026
      </div>
    </div>
    <hr style="border-color:{BOC};margin:8px 0 12px;">
    """, unsafe_allow_html=True)

    # Navigation
    PAGES = ["Accueil", "Exploration", "Prédiction", "GenAI"]
    if "page_actuelle" not in st.session_state:
        st.session_state.page_actuelle = "Accueil"
    try:
        idx_p = PAGES.index(st.session_state.page_actuelle)
    except ValueError:
        idx_p = 0

    nav = st.radio("Navigation", PAGES, index=idx_p, label_visibility="collapsed")
    st.session_state.page_actuelle = nav
# ── TRADUCTION VARIABLES TECHNIQUES ──────────────────────────────────────────
TRADUCTION_RH = {
    "OverTime": "Heures Supplémentaires",
    "MonthlyIncome": "Salaire Mensuel",
    "Age": "Âge de l'employé",
    "DistanceFromHome": "Distance Domicile-Travail",
    "JobSatisfaction": "Satisfaction au Travail",
    "EnvironmentSatisfaction": "Satisfaction Environnementale",
    "RelationshipSatisfaction": "Satisfaction Relationnelle",
    "JobInvolvement": "Implication au Travail",
    "WorkLifeBalance": "Équilibre Vie Pro/Perso",
    "YearsAtCompany": "Ancienneté dans l'entreprise",
    "YearsInCurrentRole": "Années dans le poste actuel",
    "YearsSinceLastPromotion": "Années depuis la promotion",
    "YearsWithCurrManager": "Années avec le manager",
    "DailyRate": "Tarif Journalier",
    "NumCompaniesWorked": "Nombre d'entreprises passées",
    "PercentSalaryHike": "Augmentation de salaire (%)",
    "BusinessTravel": "Déplacements Pro.",
    "JobLevel": "Niveau de responsabilité",
    "StockOptionLevel": "Niveau d'actions (Stock Options)",
    "TrainingTimesLastYear": "Nombre de formations (An dernier)",
    "JobRole": "Poste occupé",
    "Department": "Département",
    "Single": "Célibataire",
    "Married": "Marié",
    "Divorced": "Divorcé",
    "Frequently": "Fréquents",
    "Rarely": "Rares",
    "Non-Travel": "Aucun",
    "Low": "Faible",
    "Medium": "Moyen",
    "High": "Élevé",
    "Very High": "Très Élevé",
    "Yes": "Oui",
    "No": "Non"
}

def traduire_nom(nom_tech):
    nom_clair = str(nom_tech)
    for en, fr in TRADUCTION_RH.items():
        nom_clair = nom_clair.replace(en, fr)
    return nom_clair.replace("_", " : ")

# ── COLONNES FIXES hr.csv ─────────────────────────────────────────────────────
COLS_NUM_DEF = ['Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome',
    'MonthlyRate','NumCompaniesWorked','PercentSalaryHike','StockOptionLevel',
    'TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany',
    'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
COLS_CAT_DEF = ['BusinessTravel','Department','Education','EducationField',
    'EnvironmentSatisfaction','Gender','JobInvolvement','JobLevel','JobRole',
    'JobSatisfaction','MaritalStatus','OverTime','PerformanceRating',
    'RelationshipSatisfaction','WorkLifeBalance']

# ── CHARGEMENT MODÈLE ET DONNÉES ─────────────────────────────────────────────
@st.cache_resource
def charger_modele():
    try:
        return joblib.load("mon_modele_rh.pkl")
    except Exception as e:
        st.sidebar.error(f"Erreur chargement modèle : {e}")
        return None

@st.cache_data
def charger_df():
    df = pd.read_csv("hr.csv")
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    return df

data    = charger_modele()
df_base = charger_df()
n       = len(df_base)
taux    = df_base["Attrition"].mean() * 100

MODELE_OK = False; seuil = 0.31; F1 = 0.4821; AUC = 0.8030
FEATURES = []; COLS_FINALES = []; modele = None; preprocesseur = None

if data is not None and isinstance(data, dict) and "cerveau_ia" in data:
    try:
        modele        = data["cerveau_ia"]
        preprocesseur = data["traitement"]
        seuil         = float(data.get("reglage_seuil", 0.31))
        FEATURES      = list(data.get("features", []))
        F1            = float(data.get("f1", 0.4821))
        AUC           = float(data.get("auc", 0.8030))
        COLS_FINALES  = list(data.get("noms_colonnes", []))

        @st.cache_data
        def enrichir_reel(_mod, _prep, _feat, _seuil):
            d = df_base.copy()
            X = _prep.transform(d[_feat])
            d["Probabilite"] = _mod.predict_proba(X)[:, 1]
            d["Prediction"]  = (d["Probabilite"] >= _seuil).astype(int)
            d["Risque_Pct"]  = (d["Probabilite"] * 100).round(1)
            d["Niveau"]      = d["Probabilite"].apply(lambda p:
                "Critique" if p >= 0.70 else "Eleve" if p >= 0.50
                else "Modere" if p >= _seuil else "Faible")
            return d

        df = enrichir_reel(modele, preprocesseur, FEATURES, seuil)
        MODELE_OK = True

    except Exception as e:
        st.sidebar.error(f"Erreur modèle : {e}")
        MODELE_OK = False

# Fallback si modèle absent
if not MODELE_OK:
    df = df_base.copy()
    np.random.seed(42)
    base_prob        = np.random.beta(2, 5, size=len(df))
    df["Probabilite"] = np.where(df["Attrition"] == 1,
                                  np.minimum(base_prob + 0.4, 0.95),
                                  np.maximum(base_prob - 0.1, 0.02))
    df["Prediction"]  = (df["Probabilite"] >= seuil).astype(int)
    df["Risque_Pct"]  = (df["Probabilite"] * 100).round(1)
    df["Niveau"]      = df["Probabilite"].apply(lambda p:
        "Critique" if p >= 0.70 else "Eleve" if p >= 0.50
        else "Modere" if p >= seuil else "Faible")

COLS_EDA = [c for c in df.select_dtypes("number").columns
            if c not in ["Attrition", "Probabilite", "Prediction", "Risque_Pct"]]

# ── SHAP HELPER ───────────────────────────────────────────────────────────────
def get_explainer(model_to_explain, background_data):
    try:
        import shap
        if model_to_explain is None:
            return None
        f    = lambda x: model_to_explain.predict_proba(x)[:, 1]
        expl = shap.Explainer(f, background_data)
        return expl
    except Exception as e:
        st.sidebar.warning(f"SHAP : {str(e)[:80]}")
        return None




 
# =============================================================================
# PAGE 1 — ACCUEIL
# =============================================================================
if nav == "Accueil":
    st.markdown(f'<div class="pg"><div class="pt">👥 HR Analytics Dashboard</div><div class="ps">IBM HR Dataset — {n:,} employés × 31 variables — UCAO 2025-2026</div></div>', unsafe_allow_html=True)
 
    # KPI — 4 cartes claires et lisibles
    k1,k2,k3,k4 = st.columns(4)
    for col,val,lbl,c in [
        (k1, f"{n:,}",        "Employés",     BC),
        (k2, f"{taux:.1f}%",  "Attrition",    RC),
        (k3, f"{F1:.4f}",     "F1-Score",     PC),
        (k4, f"{AUC:.4f}",    "AUC-ROC",      VC),
    ]:
        with col:
            st.markdown(f'<div class="kc" style="--c:{c};margin-bottom:14px;"><div class="kv">{val}</div><div class="kl">{lbl}</div></div>', unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)

 
    # Graphique 1 — Donut (S'exécutera à coup sûr maintenant !)
    n0 = int((df["Attrition"]==0).sum())
    n1 = int((df["Attrition"]==1).sum())
    fig = go.Figure(go.Pie(
        values=[n0,n1], labels=["Restés","Partis"], hole=0.60,
        marker=dict(colors=[VC,RC], line=dict(color=FOC,width=3)),
        textinfo="label+percent", textfont=dict(size=13,color="white"), pull=[0,0.05]))
    fig.add_annotation(text=f"<b>{taux:.1f}%</b>", x=0.5, y=0.5,
        showarrow=False, font=dict(size=22,color=RC))
    fig.update_layout(**LAY,
        title=dict(text="Répartition : Restés vs Partis",font=dict(size=14),x=0.5),
        height=350, margin=dict(t=50,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)
 
    # Graphique 2 — OverTime
    ot = df.groupby("OverTime")["Attrition"].mean()*100
    fig2 = go.Figure(go.Bar(
        x=["Sans heures sup (No)","Avec heures sup (Yes)"],
        y=ot.reindex(["No","Yes"]).values,
        marker=dict(color=[VC,RC], line=dict(color=FOC,width=2)),
        text=[f"{v:.1f}%" for v in ot.reindex(["No","Yes"]).values],
        textposition="outside"))
    fig2.add_hline(y=taux, line_dash="dash", line_color=OC,
        annotation_text=f"Moyenne {taux:.1f}%", annotation_font=dict(color=OC))
    fig2.update_layout(**LAY,
        title=dict(text="Impact des Heures Supplémentaires",font=dict(size=14),x=0.5),
        yaxis=dict(**ax("Taux d'attrition (%)"), range=[0,45]),
        xaxis=ax(), height=350, margin=dict(t=50,b=40,l=55,r=20))
    st.plotly_chart(fig2, use_container_width=True)
 
    # Graphique 3 — Département
    dept = df.groupby("Department")["Attrition"].mean()*100
    dept = dept.sort_values(ascending=True)
    fig3 = go.Figure(go.Bar(
        x=dept.values, y=dept.index, orientation="h",
        marker=dict(color=[RC if v>taux else BC for v in dept.values],
                    line=dict(color=FOC,width=1.5)),
        text=[f"{v:.1f}%" for v in dept.values], textposition="outside"))
    fig3.add_vline(x=taux, line_dash="dash", line_color=OC,
        annotation_text=f"Moy. {taux:.1f}%", annotation_font=dict(color=OC))
    fig3.update_layout(**LAY,
        title=dict(text="Attrition par Département",font=dict(size=14),x=0.5),
        xaxis=dict(**ax("Taux (%)"),range=[0,28]),
        yaxis=ax(), height=300, margin=dict(t=50,b=40,l=220,r=60))
    st.plotly_chart(fig3, use_container_width=True)
 
    # Insights — 4 cartes simples
    st.markdown(f'<div style="font-size:14px;font-weight:800;color:{TXC};margin:16px 0 12px;padding-bottom:6px;border-bottom:1px solid {BOC};"> Insights clés</div>', unsafe_allow_html=True)
    i1,i2,i3,i4 = st.columns(4)
    ot_y = df[df["OverTime"]=="Yes"]["Attrition"].mean()*100
    ot_n = df[df["OverTime"]=="No"]["Attrition"].mean()*100
    js_l = df[df["JobSatisfaction"]=="Low"]["Attrition"].mean()*100
    sp   = df[df["MaritalStatus"]=="Single"]["Attrition"].mean()*100
    dm   = df.groupby("Department")["Attrition"].mean().idxmax()
    for col,c,ti,va,tx in [
        (i1,RC,"Heures supplémentaires",f"+{ot_y-ot_n:.0f} pts",f"Avec H.Sup: {ot_y:.1f}% vs sans: {ot_n:.1f}%"),
        (i2,PC,"Satisfaction basse (Low)",f"{js_l:.1f}%","Taux de départ avec JS=Low"),
        (i3,OGC,"Département le plus à risque",dm.split()[0][:12],f"Taux le plus élevé"),
        (i4,BC,"Célibataires",f"{sp:.1f}%",f"vs moyenne {taux:.1f}%"),
    ]:
        with col:
            st.markdown(f'<div class="ib" style="--c:{c};"><div class="it">{ti}</div><div class="iv">{va}</div><div class="ix">{tx}</div></div>', unsafe_allow_html=True)
 
# =============================================================================
# PAGE 2 — EXPLORATION (figure par figure)
# =============================================================================
elif nav == "Exploration":
    st.markdown(f'<div class="pg"><div class="pt"> Exploration des Données</div><div class="ps">{n:,} employés × 31 variables</div></div>', unsafe_allow_html=True)
    # Crée automatiquement la liste de toutes les colonnes numériques disponibles, sans la cible
    cols_valides_num = [c for c in df.select_dtypes(include=["number"]).columns if c not in ["Attrition", "Probabilite", "Prediction", "Risque_Pct"]]
 
    t1,t2,t3,t4,t5 = st.tabs(["Par variable","Croisements","Corrélations","Salaires","Statistiques"])
 
    with t1:
        var = st.selectbox("Choisir une variable :",
            ["OverTime","Department","JobSatisfaction","WorkLifeBalance",
             "MaritalStatus","Education","BusinessTravel","Gender","JobRole"],format_func=traduire_nom)
        vals  = df.groupby(var)["Attrition"].mean()*100
        cnts  = df.groupby(var)["Attrition"].count()
        vals  = vals.sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=vals.values, y=[traduire_nom(i) for i in vals.index], orientation="h",
            marker=dict(color=[RC if v>taux else BC for v in vals.values],
                        line=dict(color=FOC,width=1.5)),
            text=[f"<b>{v:.1f}%</b> ({int(cnts[i]):,} emp.)" for i,v in zip(vals.index,vals.values)],
            textposition="outside"))
        
        fig.add_vline(x=taux, line_dash="dash", line_color=OC,
            annotation_text=f"Moy. {taux:.1f}%", annotation_font=dict(color=OC))
        fig.update_layout(**LAY,
            title=dict(text=f"Taux d'attrition par {traduire_nom(var)}",font=dict(size=14),x=0.5),
            xaxis=dict(**ax("Taux (%)"),range=[0,65]),
            yaxis=ax(), height=max(350, len(vals)*55),
            margin=dict(t=50,b=40,l=200,r=120))
        st.plotly_chart(fig, use_container_width=True)
 
    with t2:
        # Graphique 1 : JobSatisfaction × OverTime — pleine largeur
        piv   = df.groupby(["JobSatisfaction","OverTime"])["Attrition"].mean()*100
        piv_u = piv.unstack().reindex(["Low","Medium","High","Very High"]).round(1)
        y_labels = [f"Sat. : {traduire_nom(r)}" for r in piv_u.index]
        x_labels = [f"Heures Sup : {traduire_nom(c)}" for c in piv_u.columns]
        fig_hm = go.Figure(go.Heatmap(
            z=piv_u.values, x=x_labels, y=y_labels,
            text=[[f"<b>{v:.1f}%</b>" for v in row] for row in piv_u.values],
            texttemplate="%{text}", textfont=dict(size=16,color="white"),
            colorscale=[[0,VC],[0.4,OC],[0.7,OGC],[1,RC]],
            zmid=taux, zmin=5, zmax=45,
            colorbar=dict(title="Taux %",tickfont=dict(color=T2C))))
        fig_hm.update_layout(**LAY,
            title=dict(text="Analyse Croisée : Satisfaction x Heures Supplémentaires",font=dict(size=14),x=0.5),
            height=420, margin=dict(t=55,b=70,l=130,r=40))
        st.plotly_chart(fig_hm, use_container_width=True)
 
        # Graphique 2 : Département × OverTime — pleine largeur
        ot_d = df.groupby(["Department","OverTime"])["Attrition"].mean()*100
        ot_d = ot_d.unstack().round(1)
        fig_od = go.Figure()
        for ov,cv,nv in [("No",VC,"Sans H.Sup"),("Yes",RC,"Avec H.Sup")]:
            if ov in ot_d.columns:
                fig_od.add_trace(go.Bar(name=nv, x=[traduire_nom(i) for i in ot_d.index], y=ot_d[ov].values,
                    marker=dict(color=cv,line=dict(color=FOC,width=1.5)),
                    text=[f"{v:.1f}%" for v in ot_d[ov].values], textposition="inside"))
        fig_od.add_hline(y=taux, line_dash="dash", line_color=OC,
            annotation_text=f"Moy. {taux:.1f}%", annotation_font=dict(color=OC))
        fig_od.update_layout(**LAY,
            title=dict(text="Croisement : Département × OverTime",font=dict(size=14),x=0.5),
            yaxis=dict(**ax("Taux (%)"),range=[0,55]), xaxis=ax(),
            height=400, margin=dict(t=55,b=60,l=55,r=20), barmode="group")
        st.plotly_chart(fig_od, use_container_width=True)
 
        # Graphique 3 : WLB × Statut Matrimonial — pleine largeur
        wm   = df.groupby(["WorkLifeBalance","MaritalStatus"])["Attrition"].mean()*100
        wm_u = wm.unstack().reindex(["Bad","Good","Better","Best"]).round(1)
        fig_wm = go.Figure(go.Heatmap(
            z=wm_u.values, x=[traduire_nom(cx) for cx in wm_u.columns],
            y=[f"Équilibre : {traduire_nom(r)}" for r in wm_u.index],
            text=[[f"<b>{v:.1f}%</b>" for v in row] for row in wm_u.values],
            texttemplate="%{text}", textfont=dict(size=16,color="white"),
            colorscale=[[0,VC],[0.4,OC],[0.7,OGC],[1,RC]],
            zmid=taux, colorbar=dict(title="Taux %",tickfont=dict(color=T2C))))
        fig_wm.update_layout(**LAY,
            title=dict(text="Croisement : WLB × Statut Matrimonial",font=dict(size=14),x=0.5),
            height=400, margin=dict(t=55,b=60,l=130,r=40))
        st.plotly_chart(fig_wm, use_container_width=True)

 
 
    with t3:
        
        corr = df[cols_valides_num+["Attrition"]].corr()["Attrition"].drop("Attrition")
        ca   = corr.abs().sort_values(ascending=True)
        noms_vars_fr = [traduire_nom(i) for i in ca.index]
        
        fig_c = go.Figure(go.Bar(
            x=ca.values, y=noms_vars_fr, orientation="h",
            marker=dict(color=[RC if corr[v]<0 else BC for v in ca.index],
                        line=dict(color=FOC,width=1.5)),
            text=[f"{corr[v]:+.4f}" for v in ca.index], textposition="outside"))
        fig_c.update_layout(**LAY,
            title=dict(text="Corrélations des variables avec l'Attrition",font=dict(size=14),x=0.5),
            xaxis=dict(**ax("Corrélation absolue")), yaxis=ax(),
            height=520, margin=dict(t=50,b=40,l=230,r=90))
        st.plotly_chart(fig_c, use_container_width=True)
 
    with t4:
        st.markdown(f'<div style="font-size:12px;color:{T2C};margin-bottom:8px;">Chaque graphique est affiché sur toute la largeur pour une meilleure lisibilité.</div>', unsafe_allow_html=True)
        # Distribution salaires — pleine largeur
        fig_sal = go.Figure()
        fig_sal.add_trace(go.Histogram(x=df[df["Attrition"]==0]["MonthlyIncome"],
            name="Restés", marker_color=VC, opacity=0.75, nbinsx=30))
        fig_sal.add_trace(go.Histogram(x=df[df["Attrition"]==1]["MonthlyIncome"],
            name="Partis",  marker_color=RC, opacity=0.75, nbinsx=30))
        fig_sal.update_layout(**LAY,
            title=dict(text="Distribution des Salaires — Restés vs Partis",font=dict(size=14),x=0.5),
            xaxis=dict(**ax("Salaire mensuel")), yaxis=dict(**ax("Nb employés")),
            height=400, barmode="overlay", margin=dict(t=50,b=40,l=60,r=20))
        st.plotly_chart(fig_sal, use_container_width=True)
 
        # Salaire par département — pleine largeur
        sd = df.groupby("Department")["MonthlyIncome"].mean().sort_values()
        fig_sd = go.Figure(go.Bar(
            x=sd.values, y=[traduire_nom(i) for i in sd.index], orientation="h",
            marker=dict(color=BC, line=dict(color=FOC,width=1.5)),
            text=[f"{v:,.0f} €" for v in sd.values], textposition="outside"))
        fig_sd.update_layout(**LAY,
            title=dict(text="Salaire Moyen par Département",font=dict(size=14),x=0.5),
            xaxis=dict(**ax("Salaire moyen")), yaxis=ax(),
            height=340, margin=dict(t=50,b=40,l=220,r=120))
        st.plotly_chart(fig_sd, use_container_width=True)
 
        # Salaire par poste
        sr = df.groupby("JobRole")["MonthlyIncome"].mean().sort_values()
        fig_sr = go.Figure(go.Bar(
            x=sr.values, y=[traduire_nom(i) for i in sr.index], orientation="h",
            marker=dict(color=PC, line=dict(color=FOC,width=1.5)),
            text=[f"{v:,.0f} €" for v in sr.values], textposition="outside"))
        fig_sr.update_layout(**LAY,
            title=dict(text="Salaire Moyen par Poste",font=dict(size=14),x=0.5),
            xaxis=dict(**ax("Salaire moyen")), yaxis=ax(),
            height=420, margin=dict(t=50,b=40,l=230,r=100))
        st.plotly_chart(fig_sr, use_container_width=True)
 
        # Salaire vs ancienneté
        cs = df.groupby("YearsAtCompany")["MonthlyIncome"].mean()
        fig_cs = go.Figure(go.Scatter(
            x=cs.index, y=cs.values, mode="lines+markers",
            line=dict(color=OC,width=2.5), marker=dict(color=OC,size=7),
            fill="tozeroy", fillcolor="rgba(255,209,102,0.1)"))
        fig_cs.update_layout(**LAY,
            title=dict(text="Salaire Moyen selon l'Ancienneté",font=dict(size=14),x=0.5),
            xaxis=dict(**ax("Années dans l'entreprise")),
            yaxis=dict(**ax("Salaire moyen")),
            height=360, margin=dict(t=50,b=40,l=70,r=20))
        st.plotly_chart(fig_cs, use_container_width=True)
 
    with t5:
        st.markdown("###  Synthèse statistique des indicateurs")
        
        # 1. Calcul des statistiques de base
        stats = df[cols_valides_num].describe().round(2)
        
        # 2. Traduction des noms des mesures (index)
        stats.index = ["Nombre", "Moyenne", "Écart-type", "Minimum", "25%", "Médiane", "75%", "Maximum"]
        
        # 3. Transposition du tableau (plus facile à lire verticalement)
        stats_t = stats.T
        
        # 4. Traduction des noms de variables (index après transposition)
        stats_t.index = [traduire_nom(c) for c in stats_t.index]
        
        # 5. Affichage propre
        st.dataframe(stats_t, use_container_width=True)
 
# =============================================================================
# PAGE 3 — PRÉDICTION
# =============================================================================
elif nav == "Prédiction":
    st.markdown(f'<div class="pg"><div class="pt"> Prédiction & Explicabilité</div><div class="ps">Recherchez un employé ou faites une prédiction manuelle</div></div>', unsafe_allow_html=True)
 
    if not MODELE_OK:
        st.error("Modèle non disponible. Lancez 1_HR_Analytics.ipynb et resauvegardez le modèle.")
        st.stop()
 
    tab_r, tab_m = st.tabs([" Rechercher un employé"," Prédiction manuelle"])
 
    # ── ONGLET RECHERCHE ──────────────────────────────────────────────────────
    with tab_r:
 
        # Filtres
        with st.container():
            fc1,fc2,fc3,fc4 = st.columns(4)
            with fc1: fd = st.selectbox("Département",["Tous"]+sorted(df["Department"].unique()))
            with fc2: fr = st.selectbox("Poste",["Tous"]+sorted(df["JobRole"].unique()))
            with fc3:
                fn_label = st.selectbox("Niveau risque",
                    ["Tous"," Critique"," Élevé"," Modéré"," Faible"])
                fn = {"Tous":"Tous"," Critique":"Critique",
                      " Élevé":"Eleve"," Modéré":"Modere"," Faible":"Faible"}[fn_label]
            with fc4: fo = st.selectbox("Heures sup.",["Tous","Oui","Non"])
 
            sc1,sc2 = st.columns([4,1])
            with sc1:
                recherche = st.text_input("",
                    placeholder=" Entrez un numéro d'employé (0-1469), un département ou un poste...",
                    label_visibility="collapsed")
            with sc2:
                chercher = st.button(" Rechercher", use_container_width=True, type="primary")
 
        # Filtrage
        df_f = df.copy()
        if fd != "Tous": df_f = df_f[df_f["Department"]==fd]
        if fr != "Tous": df_f = df_f[df_f["JobRole"]==fr]
        if fn != "Tous": df_f = df_f[df_f["Niveau"]==fn]
        if fo == "Oui":  df_f = df_f[df_f["OverTime"]=="Yes"]
        elif fo == "Non":df_f = df_f[df_f["OverTime"]=="No"]
        if recherche.strip():
            rch  = recherche.strip().lower()
            mask = (df_f.index.astype(str).str.contains(rch) |
                    df_f["Department"].str.lower().str.contains(rch) |
                    df_f["JobRole"].str.lower().str.contains(rch))
            df_f = df_f[mask]
 
        filtres_actifs = (fd!="Tous" or fr!="Tous" or fn!="Tous" or
                          fo!="Tous" or recherche.strip()!="")
 
        # Déterminer l'employé à afficher
        if not filtres_actifs and not chercher:
            idx_sel = df["Probabilite"].idxmax()
            st.info(" Affichage de l'employé avec le risque le plus élevé. Utilisez les filtres pour en chercher un autre.")
        else:
            df_f = df_f.sort_values("Probabilite", ascending=False)
 
            # --- LOGIQUE DE SÉLECTION SÉCURISÉE ---
            if recherche.strip().isdigit():
                # CAS 1 : Recherche directe par ID
                idx_temp = int(recherche.strip())
                if idx_temp not in df.index:
                    st.error(f" L'ID #{idx_temp} n'existe pas dans la base de données (0-1469).")
                    st.stop()
                idx_sel = idx_temp
            
            elif len(df_f) == 0:
                # CAS 2 : Aucun résultat avec les filtres
                st.warning(" Aucun résultat ne correspond à vos filtres.")
                st.stop()
                
            elif len(df_f) == 1:
                # CAS 3 : Un seul résultat trouvé avec les filtres
                idx_sel = df_f.index[0]
            
            else:
                # CAS 4 : Plusieurs résultats -> On affiche le tableau
                # Correction du nom pour correspondre à votre fonction enrichir()
                nc_f = (df_f["Niveau"]=="Critique").sum()
                ne_f = (df_f["Niveau"]=="Eleve").sum() 
                
                c1r,c2r,c3r,c4r = st.columns(4)
                with c1r: st.markdown(f'<div class="kc" style="--c:{BC};"><div class="kv">{len(df_f)}</div><div class="kl">Résultats</div></div>', unsafe_allow_html=True)
                with c2r: st.markdown(f'<div class="kc" style="--c:{RC};"><div class="kv">{nc_f}</div><div class="kl">🔴 Critiques</div></div>', unsafe_allow_html=True)
                with c3r: st.markdown(f'<div class="kc" style="--c:{OGC};"><div class="kv">{ne_f}</div><div class="kl">🟠 Élevés</div></div>', unsafe_allow_html=True)
                with c4r: st.markdown(f'<div class="kc" style="--c:{OC};"><div class="kv">{df_f["Probabilite"].mean()*100:.1f}%</div><div class="kl">Risque moyen</div></div>', unsafe_allow_html=True)
                
                # ... (votre code pour le tableau event = st.dataframe ...)           # Tableau
                df_t = df_f[["Age","Gender","Department","JobRole","MonthlyIncome",
                             "OverTime","JobSatisfaction","Risque_Pct","Attrition"]].copy().head(200)
                df_t.columns = ["Âge","Genre","Département","Poste","Salaire","H.Sup","JS","Risque %","Réalité"]
                df_t["Réalité"] = df_t["Réalité"].map({0:" Resté",1:" Parti"})
                df_t["H.Sup"]   = df_t["H.Sup"].map({"Yes":" Oui","No":" Non"})
 
                event = st.dataframe(df_t, use_container_width=True, hide_index=False,
                    height=240, on_select="rerun", selection_mode="single-row",
                    column_config={"Risque %": st.column_config.ProgressColumn(
                        "Risque %", min_value=0, max_value=100, format="%.1f%%", width=100)})
 
                # Sélection ou premier par défaut
                if event.selection and event.selection["rows"]:
                    idx_sel = df_f.index[event.selection["rows"][0]]
                else:
                    idx_sel = df_f.index[0]  # Premier résultat affiché automatiquement
 
        # ══════════════════════════════════════════════════════════════════════
        # AFFICHAGE — SECTIONS PLEINE LARGEUR
        # ══════════════════════════════════════════════════════════════════════
        if idx_sel is not None and idx_sel in df.index:
            emp   = df.loc[idx_sel]
            proba = float(emp["Probabilite"])
            pct   = proba * 100
 
            if pct>=70:   cr=RC;  niv=" CRITIQUE"; rec="Intervention immédiate — Entretien sous 48h"
            elif pct>=50: cr=OGC; niv=" ÉLEVÉ";    rec="Entretien individuel sous 2 semaines"
            elif pct>=seuil*100: cr=OC; niv=" MODÉRÉ"; rec="Suivi mensuel RH"
            else:         cr=VC;  niv=" FAIBLE";   rec="Profil stable — Entretiens annuels"
 
            reel  = int(emp["Attrition"])
            creel = RC if reel==1 else VC
            treel = " A quitté" if reel==1 else " Est resté"
            g_txt = "Femme" if emp["Gender"]=="Female" else "Homme"
 
            st.markdown(f"""
            <hr style="border-color:{BOC};margin:20px 0 16px;">
            <div style="font-size:16px;font-weight:800;color:{TXC};margin-bottom:4px;">
                 Employé #{idx_sel} — {g_txt}, {int(emp['Age'])} ans
            </div>
            <div style="font-size:13px;color:{T2C};margin-bottom:18px;">
                {emp['Department']} · {emp['JobRole']}
                &nbsp;|&nbsp;
                <span style="color:{creel};font-weight:700;">{treel}</span>
            </div>
            """, unsafe_allow_html=True)
 
            # ── SECTION 1 : INFORMATIONS — pleine largeur ─────────────────────
            st.markdown(f'<div class="section-title" style="--c:{BC};">👤 Informations de l\'Employé</div>', unsafe_allow_html=True)
 
            infos = [
                (" Département",         emp["Department"],                                False),
                (" Poste",              traduire_nom(emp["JobRole"]),                                   False),
                (" Niveau de poste",     emp["JobLevel"],                                  False),
                (" Salaire mensuel",     f"{int(emp['MonthlyIncome']):,} €",               False),
                (" Ancienneté",          f"{int(emp['YearsAtCompany'])} ans",              False),
                (" Ans sans promotion",  f"{int(emp['YearsSinceLastPromotion'])} ans",     emp["YearsSinceLastPromotion"]>=3),
                (" Heures sup.",         traduire_nom(emp["OverTime"]), emp["OverTime"]=="Yes"),
                (" Satisfaction travail",traduire_nom(emp["JobSatisfaction"]),                            emp["JobSatisfaction"] in ["Low","Medium"]),
                (" Équilibre WLB",      traduire_nom(emp["WorkLifeBalance"]),                            emp["WorkLifeBalance"] in ["Bad","Good"]),
                (" Distance domicile",   f"{int(emp['DistanceFromHome'])} km",             False),
                (" Statut matrimonial",  traduire_nom(emp["MaritalStatus"]),                             False),
                (" Voyages pro",         traduire_nom(emp["BusinessTravel"]),                            emp["BusinessTravel"]=="Travel_Frequently"),
                (" Éducation",           emp["Education"],                                 False),
                (" Performance",         emp["PerformanceRating"],                         False),
                (" Satisfaction env.",   traduire_nom(emp.get("EnvironmentSatisfaction","—")),            emp.get("EnvironmentSatisfaction","High") in ["Low","Medium"]),
                (" Satisfaction rel.",   traduire_nom(emp.get("RelationshipSatisfaction","—")),           emp.get("RelationshipSatisfaction","High") in ["Low","Medium"]),
            ]
            cols4 = st.columns(4)
            for i,(lb,va,al) in enumerate(infos):
                cv2 = RC if al else TXC
                ic  = " " if al else ""
                with cols4[i%4]:
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-lbl">{lb}</div>
                        <div class="info-val" style="color:{cv2};">{va}{ic}</div>
                    </div>""", unsafe_allow_html=True)
 
            st.markdown("<br>", unsafe_allow_html=True)
 
            # ── SECTION 2 : JAUGE — pleine largeur ────────────────────────────
            st.markdown(f'<div class="section-title" style="--c:{cr};"> Score de Risque — {niv}</div>', unsafe_allow_html=True)
 
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=pct,
                number={"suffix":"%","font":{"size":60,"color":TXC}},
                title={"text":niv,"font":{"size":18,"color":cr}},
                gauge={
                    "axis"    :{"range":[0,100],"tickcolor":T2C,"ticksuffix":"%","nticks":6},
                    "bar"     :{"color":cr,"thickness":0.3},
                    "bgcolor" :GRC,"bordercolor":BOC,
                    "steps"   :[
                        {"range":[0,seuil*100],"color":"rgba(0,200,150,0.15)"},
                        {"range":[seuil*100,50],"color":"rgba(255,209,102,0.15)"},
                        {"range":[50,70],"color":"rgba(255,140,66,0.15)"},
                        {"range":[70,100],"color":"rgba(255,75,110,0.18)"},
                    ],
                    "threshold":{"line":{"color":OC,"width":3},"thickness":0.8,"value":seuil*100},
                }))
            fig_g.update_layout(paper_bgcolor=CAC, plot_bgcolor=CAC,
                font=dict(color=TXC), height=380,
                margin=dict(t=80,b=10,l=80,r=80))
            st.plotly_chart(fig_g, use_container_width=True)
 
            # ── SECTION 3 : DISTRIBUTION — pleine largeur ─────────────────────
            st.markdown(f'<div class="section-title" style="--c:{BC};"> Position parmi les {len(df):,} employés</div>', unsafe_allow_html=True)
 
            tp2    = df["Probabilite"].values
            pct_ile= float((tp2<proba).mean()*100)
            nb_p   = int((tp2>proba).sum())
            hv,hb  = np.histogram(tp2, bins=30)
            bca    = (hb[:-1]+hb[1:])/2
            bwa    = hb[1]-hb[0]
            bcols  = [RC if b>=0.70 else OGC if b>=0.50 else OC if b>=seuil else VC for b in bca]
 
            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(x=bca, y=hv, width=[bwa*0.85]*len(bca),
                marker=dict(color=bcols, opacity=0.75, line=dict(color=FOC,width=0.5)),
                name="Employés", showlegend=False))
            fig_d.add_vline(x=proba, line_color=cr, line_width=3,
                annotation_text=f"Employé #{idx_sel} → {pct:.1f}%",
                annotation_font=dict(color=cr, size=13), annotation_position="top right")
            fig_d.add_vline(x=seuil, line_dash="dash", line_color=T2C, line_width=2,
                annotation_text=f"Seuil de décision ({seuil:.2f})",
                annotation_font=dict(color=T2C, size=11))
            fig_d.update_layout(**LAY,
                title=dict(
                    text=f"{pct_ile:.0f}e percentile — {nb_p} employés ont un risque plus élevé",
                    font=dict(size=14), x=0.5),
                xaxis=dict(**ax("Probabilité de départ"), range=[0,0.95]),
                yaxis=dict(**ax("Nombre d'employés")),
                height=380, margin=dict(t=55,b=40,l=70,r=20))
            st.plotly_chart(fig_d, use_container_width=True)
 
            # Recommandation — pleine largeur
            st.markdown(f"""
            <div class="recomm" style="--c:{cr};margin-bottom:24px;">
                <div style="font-size:11px;font-weight:700;color:{cr};margin-bottom:6px;
                text-transform:uppercase;letter-spacing:1px;"> Recommandation RH</div>
                <div style="font-size:15px;font-weight:600;color:{TXC};">{rec}</div>
            </div>
            """, unsafe_allow_html=True)
 
            # ── SECTION 4 : SHAP — pleine largeur ─────────────────────────────
            st.markdown(f'<div class="section-title" style="--c:{PC};"> Explication SHAP — Pourquoi ce score ?</div>', unsafe_allow_html=True)
 
            if len(COLS_FINALES) == 0:
                st.warning("COLS_FINALES vide — resauvegardez le modele avec joblib.")
            else:
                try:
                    import shap as _shap
                    # On définit la fonction qui donne les probabilités
                    f_pred = lambda x: modele.predict_proba(x)[:, 1]
                    
                    # On crée un échantillon de comparaison (50 personnes au hasard)
                    X_bg = preprocesseur.transform(df_base.sample(50, random_state=42)[FEATURES])
                    
                    # On utilise l'Explainer universel (pas TreeExplainer)
                    expl = _shap.Explainer(f_pred, X_bg)
                    
                    # On prépare la donnée de l'employé sélectionné
                    X_e = preprocesseur.transform(df.loc[[idx_sel]][FEATURES])
                    
                    # On calcule le score SHAP pour cet employé
                    shap_output = expl(X_e)
                    sv = shap_output.values[0] # On récupère les valeurs numériques
                     
                    
                    
 
                    df_sv = pd.DataFrame({
                        "Variable_Tech": COLS_FINALES,
                        "Variable": [traduire_nom(c) for c in COLS_FINALES],
                        "SHAP"    : sv,
                        "Abs"     : np.abs(sv)
                    }).sort_values("Abs", ascending=False).head(12)
 
                    csv_c = [RC if v > 0 else BC for v in df_sv["SHAP"]]
 
                    fig_sv = go.Figure(go.Bar(
                        x=df_sv["SHAP"].values, y=df_sv["Variable"].values,
                        orientation="h",
                        marker=dict(color=csv_c, line=dict(color=FOC, width=1.5)),
                        text=[f"{v:+.3f}" for v in df_sv["SHAP"].values],
                        textposition="outside",
                        hovertemplate="%{y}<br>SHAP = %{x:+.4f}<extra></extra>"))
                    fig_sv.add_vline(x=0, line_color=T2C, line_width=2)
                    fig_sv.update_layout(**LAY,
                        title=dict(
                            text="Rouge = augmente le risque | Bleu = reduit le risque",
                            font=dict(size=13), x=0.5),
                        xaxis=dict(**ax("<- Reduit  |  Augmente ->"),
                            range=[df_sv["SHAP"].min()*1.6, df_sv["SHAP"].max()*1.6]),
                        yaxis=ax(), height=480,
                        margin=dict(t=55, b=40, l=230, r=120))
                    st.plotly_chart(fig_sv, use_container_width=True)
 
                    tp_s = df_sv[df_sv["SHAP"] > 0].head(4)
                    tn_s = df_sv[df_sv["SHAP"] < 0].head(4)
                    v1   = tp_s.iloc[0]["Variable"] if not tp_s.empty else "---"
 
                    sh1, sh2, sh3 = st.columns(3)
                    with sh1:
                        rows_p = "".join([
                            f"<div style='display:flex;justify-content:space-between;padding:5px 0;"
                            f"border-bottom:1px solid {RC}22;'>"
                            f"<span style='font-size:12px;color:{TXC};'>- {r['Variable']}</span>"
                            f"<span style='font-size:12px;font-weight:700;color:{RC};'>"
                            f"{r['SHAP']:+.3f}</span></div>"
                            for _, r in tp_s.iterrows()])
                        st.markdown(
                            f"<div style='background:{RC}12;border:1px solid {RC}44;"
                            f"border-radius:10px;padding:14px 16px;'>"
                            f"<div style='font-size:12px;font-weight:700;color:{RC};"
                            f"margin-bottom:10px;'>Augmente le risque</div>{rows_p}</div>",
                            unsafe_allow_html=True)
                    with sh2:
                        rows_n = "".join([
                            f"<div style='display:flex;justify-content:space-between;padding:5px 0;"
                            f"border-bottom:1px solid {VC}22;'>"
                            f"<span style='font-size:12px;color:{TXC};'>- {r['Variable']}</span>"
                            f"<span style='font-size:12px;font-weight:700;color:{VC};'>"
                            f"{r['SHAP']:+.3f}</span></div>"
                            for _, r in tn_s.iterrows()])
                        st.markdown(
                            f"<div style='background:{VC}12;border:1px solid {VC}44;"
                            f"border-radius:10px;padding:14px 16px;'>"
                            f"<div style='font-size:12px;font-weight:700;color:{VC};"
                            f"margin-bottom:10px;'>Reduit le risque</div>{rows_n}</div>",
                            unsafe_allow_html=True)
                    with sh3:
                        st.markdown(
                            f"<div style='background:{OC}12;border:1px solid {OC}44;"
                            f"border-radius:10px;padding:14px 16px;'>"
                            f"<div style='font-size:12px;font-weight:700;color:{OC};"
                            f"margin-bottom:10px;'>Conclusion</div>"
                            f"<div style='font-size:13px;color:{TXC};line-height:1.8;'>"
                            f"Facteur principal :<br>"
                            f"<strong style='color:{RC};'>{v1}</strong><br><br>"
                            f"Risque : <strong style='color:{cr};font-size:20px;'>{pct:.1f}%</strong><br>"
                            f"Niveau : <strong style='color:{cr};'>{niv}</strong>"
                            f"</div></div>",
                            unsafe_allow_html=True)
 
                except Exception as e:
                    st.warning(f"SHAP non disponible : {e}")
               
                
                



 
    # ── ONGLET PRÉDICTION MANUELLE ────────────────────────────────────────────
    with tab_m:
        st.markdown(f'<div style="font-size:13px;font-weight:700;color:{TXC};margin-bottom:14px;"> Saisissez les informations d\'un employé</div>', unsafe_allow_html=True)
 
        r1,r2,r3 = st.columns(3, gap="large")
        with r1:
            st.markdown(f'<div style="font-size:11px;font-weight:700;color:{BC};text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;"> Bien-être</div>', unsafe_allow_html=True)
            m_js   = st.select_slider("Satisfaction travail", options=["Low","Medium","High","Very High"], value="High")
            m_wlb  = st.select_slider("Équilibre WLB", options=["Bad","Good","Better","Best"], value="Good")
            m_ot   = st.radio("Heures supplémentaires", [" Non"," Oui"], horizontal=True)
            m_env  = st.select_slider("Satisfaction environnement", options=["Low","Medium","High","Very High"], value="High")
        with r2:
            st.markdown(f'<div style="font-size:11px;font-weight:700;color:{BC};text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;"> Carrière</div>', unsafe_allow_html=True)
            m_dept  = st.selectbox("Département", sorted(df["Department"].unique()))
            m_role  = st.selectbox("Poste", sorted(df["JobRole"].unique()))
            m_jlev  = st.selectbox("Niveau poste", ["Entry Level","Junior Level","Mid Level","Senior Level","Executive Level"])
            m_genre = st.radio("Genre", ["Male","Female"], horizontal=True)
            m_marit = st.selectbox("Statut matrimonial", ["Divorced","Married","Single"])
        with r3:
            st.markdown(f'<div style="font-size:11px;font-weight:700;color:{BC};text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;"> Informations</div>', unsafe_allow_html=True)
            m_age   = st.number_input("Âge", 18, 60, 35)
            m_anc   = st.number_input("Ancienneté (ans)", 0, 30, 5)
            m_inc   = st.number_input("Salaire mensuel", 1000, 20000, 5000, step=500)
            m_promo = st.number_input("Ans sans promotion", 0, 15, 2)
            m_dist  = st.slider("Distance domicile (km)", 1, 30, 10)
            m_travel= st.selectbox("Voyages", ["Non-Travel","Travel_Rarely","Travel_Frequently"])
 
        st.markdown("<br>", unsafe_allow_html=True)
        pb = st.button("🔮 Analyser le risque de départ", use_container_width=True, type="primary")
 
        if pb:
            ot_val = "Yes" if "Oui" in m_ot else "No"
            df_in  = pd.DataFrame([{
                "Age":m_age,"DailyRate":800,"DistanceFromHome":m_dist,"HourlyRate":60,
                "MonthlyIncome":m_inc,"MonthlyRate":15000,"NumCompaniesWorked":2,
                "PercentSalaryHike":12,"StockOptionLevel":0,"TotalWorkingYears":m_anc+3,
                "TrainingTimesLastYear":2,"YearsAtCompany":m_anc,
                "YearsInCurrentRole":max(0,m_anc-2),"YearsSinceLastPromotion":m_promo,
                "YearsWithCurrManager":2,"BusinessTravel":m_travel,"Department":m_dept,
                "Education":"Bachelor","EducationField":"Marketing",
                "EnvironmentSatisfaction":m_env,"Gender":m_genre,"JobInvolvement":"High",
                "JobLevel":m_jlev,"JobRole":m_role,"JobSatisfaction":m_js,
                "MaritalStatus":m_marit,"OverTime":ot_val,"PerformanceRating":"Excellent",
                "RelationshipSatisfaction":"High","WorkLifeBalance":m_wlb,
            }])
            X_m    = preprocesseur.transform(df_in[FEATURES])
            proba_m= float(modele.predict_proba(X_m)[0][1])
            pct_m  = proba_m * 100
 
            if pct_m>=70:   cm=RC; niv_m=" CRITIQUE"; rec_m="Intervention immédiate"
            elif pct_m>=50: cm=OGC; niv_m=" ÉLEVÉ";   rec_m="Entretien sous 2 semaines"
            elif pct_m>=seuil*100: cm=OC; niv_m=" MODÉRÉ"; rec_m="Suivi mensuel"
            else:           cm=VC; niv_m=" FAIBLE";   rec_m="Profil stable"
 
            st.markdown(f'<hr style="border-color:{BOC};margin:20px 0;">', unsafe_allow_html=True)
 
            # Jauge — pleine largeur
            st.markdown(f'<div class="section-title" style="--c:{cm};"> Résultat : {niv_m}</div>', unsafe_allow_html=True)
            fig_gm = go.Figure(go.Indicator(
                mode="gauge+number", value=pct_m,
                number={"suffix":"%","font":{"size":60,"color":TXC}},
                title={"text":niv_m,"font":{"size":18,"color":cm}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":T2C,"ticksuffix":"%","nticks":6},
                    "bar":{"color":cm,"thickness":0.3},"bgcolor":GRC,"bordercolor":BOC,
                    "steps":[
                        {"range":[0,seuil*100],"color":"rgba(0,200,150,0.15)"},
                        {"range":[seuil*100,50],"color":"rgba(255,209,102,0.15)"},
                        {"range":[50,70],"color":"rgba(255,140,66,0.15)"},
                        {"range":[70,100],"color":"rgba(255,75,110,0.18)"},
                    ],
                    "threshold":{"line":{"color":OC,"width":3},"thickness":0.8,"value":seuil*100},
                }))
            fig_gm.update_layout(paper_bgcolor=CAC, plot_bgcolor=CAC,
                font=dict(color=TXC), height=380, margin=dict(t=80,b=10,l=80,r=80))
            st.plotly_chart(fig_gm, use_container_width=True)
 
            # Distribution — pleine largeur
            st.markdown(f'<div class="section-title" style="--c:{BC};"> Position parmi les {len(df):,} employés</div>', unsafe_allow_html=True)
            tp3    = df["Probabilite"].values
            pi3    = float((tp3<proba_m).mean()*100)
            hv3,hb3= np.histogram(tp3, bins=30)
            bca3   = (hb3[:-1]+hb3[1:])/2; bwa3=hb3[1]-hb3[0]
            bc3    = [RC if b>=0.70 else OGC if b>=0.50 else OC if b>=seuil else VC for b in bca3]
            fig_d3 = go.Figure()
            fig_d3.add_trace(go.Bar(x=bca3, y=hv3, width=[bwa3*0.85]*len(bca3),
                marker=dict(color=bc3, opacity=0.75, line=dict(color=FOC,width=0.5)),
                showlegend=False))
            fig_d3.add_vline(x=proba_m, line_color=cm, line_width=3,
                annotation_text=f"Cet employé → {pct_m:.1f}%",
                annotation_font=dict(color=cm, size=13), annotation_position="top right")
            fig_d3.add_vline(x=seuil, line_dash="dash", line_color=T2C, line_width=2)
            fig_d3.update_layout(**LAY,
                title=dict(text=f"{pi3:.0f}e percentile parmi {len(df):,} employés",
                    font=dict(size=14), x=0.5),
                xaxis=dict(**ax("Probabilité de départ"), range=[0,0.95]),
                yaxis=dict(**ax("Nb employés")),
                height=380, margin=dict(t=55,b=40,l=70,r=20))
            st.plotly_chart(fig_d3, use_container_width=True)
 
            # Recommandation
            st.markdown(f'<div class="recomm" style="--c:{cm};margin-bottom:24px;"><div style="font-size:11px;font-weight:700;color:{cm};margin-bottom:6px;text-transform:uppercase;"> Recommandation</div><div style="font-size:15px;font-weight:600;color:{TXC};">{rec_m}</div></div>', unsafe_allow_html=True)
 
            # SHAP — pleine largeur
            if len(COLS_FINALES) > 0:
                st.markdown(f'<div class="section-title" style="--c:{PC};"> Explication SHAP</div>', unsafe_allow_html=True)
                try:
                    import shap as _shap2
                     # 1. Fonction de prédiction
                    f_pred = lambda x: modele.predict_proba(x)[:, 1]
                    
                    # 2. Échantillon de fond (50 employés pour la comparaison)
                    X_bg = preprocesseur.transform(df_base.sample(50, random_state=42)[FEATURES])
                    
                    # 3. Création de l'explainer universel
                    expl2 = _shap2.Explainer(f_pred, X_bg)
                    
                    # 4. Calcul pour l'employé saisi (X_m est déjà calculé plus haut dans votre code)
                    shap_output2 = expl2(X_m)
                    sv2 = shap_output2.values[0] # On utilise sv2 pour correspondre à la suite
 
                    df_s2 = pd.DataFrame({
                        "Variable": [traduire_nom(c) for c in COLS_FINALES],
                        "SHAP"    : sv2,
                        "Abs"     : np.abs(sv2)
                    }).sort_values("Abs", ascending=False).head(12)
 
                    cs2 = [RC if v > 0 else BC for v in df_s2["SHAP"]]
 
                    fig_s2 = go.Figure(go.Bar(
                        x=df_s2["SHAP"].values, y=df_s2["Variable"].values,
                        orientation="h",
                        marker=dict(color=cs2, line=dict(color=FOC, width=1.5)),
                        text=[f"{v:+.3f}" for v in df_s2["SHAP"].values],
                        textposition="outside"))
                    fig_s2.add_vline(x=0, line_color=T2C, line_width=2)
                    fig_s2.update_layout(**LAY,
                        title=dict(
                            text="Rouge = augmente le risque | Bleu = reduit le risque",
                            font=dict(size=13), x=0.5),
                        xaxis=dict(**ax("<- Reduit  |  Augmente ->"),
                            range=[df_s2["SHAP"].min()*1.6, df_s2["SHAP"].max()*1.6]),
                        yaxis=ax(), height=480,
                        margin=dict(t=55, b=40, l=230, r=120))
                    st.plotly_chart(fig_s2, use_container_width=True)
 
                except Exception as e:
                    st.warning(f"SHAP : {e}")


# =============================================================================
# PAGE GENAI — CHAT RH INTELLIGENT ET MODERNE
# =============================================================================
elif nav == "GenAI":
    st.markdown(f'<div class="pg"><div class="pt"> Assistant RH — GenAI</div><div class="ps">Posez vos questions RH — Propulsé par Mistral/Llama</div></div>', unsafe_allow_html=True)

    if not MODELE_OK:
        st.error("Modèle non disponible.")
        st.stop()

    # Contexte RH pour enrichir les réponses
    n_critique = int((df["Niveau"] == "Critique").sum()) if MODELE_OK else 0
    n_risque   = int((df["Probabilite"] >= seuil).sum()) if MODELE_OK else 0
    dept_max   = df.groupby("Department")["Attrition"].mean().idxmax() if MODELE_OK else "N/A"

    CONTEXTE_RH = f"""Tu es un expert en Ressources Humaines et People Analytics.
Tu as accès aux données RH suivantes :
- Dataset : {n} employés au total
- Taux d'attrition observé : {taux:.1f}%
- Employés à risque (score >= seuil) : {n_risque}
- Employés critiques (risque >= 70%) : {n_critique}
- Département le plus à risque : {dept_max}
- Modèle : XGBoost | F1={F1:.4f} | AUC={AUC:.4f} | Seuil={seuil:.2f}
Réponds en français, de façon professionnelle et concise."""

    # Initialiser l'historique du chat
    if "messages_genai" not in st.session_state:
        st.session_state.messages_genai = []

    # Suggestions de questions
    st.markdown(f'<div style="font-size:12px;color:{T2C};margin-bottom:10px;"> Exemples de questions :</div>', unsafe_allow_html=True)
    cols_q = st.columns(3)
    questions = [
        "Combien d'employés sont à risque critique ?",
        "Quel département est le plus à risque ?",
        "Quelles sont les principales causes de départ ?",
        "Comment réduire le turnover dans Sales ?",
        "Quel est le coût estimé du turnover actuel ?",
        "Quelles actions RH recommandes-tu en priorité ?",
    ]
    for i, q in enumerate(questions):
        with cols_q[i % 3]:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                st.session_state.messages_genai.append({"role": "user", "content": q})

    st.markdown("<br>", unsafe_allow_html=True)

    # Afficher l'historique
    for msg in st.session_state.messages_genai:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
              <div style="background:{BC}22;border:1px solid {BC}44;border-radius:12px 12px 2px 12px;
              padding:12px 16px;max-width:75%;font-size:13px;color:{TXC};">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-start;margin-bottom:10px;">
              <div style="background:{GRC};border:1px solid {BOC};border-radius:12px 12px 12px 2px;
              padding:12px 16px;max-width:75%;font-size:13px;color:{TXC};">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)

    # Zone de saisie
    with st.form("chat_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([5, 1])
        with col_inp:
            user_input = st.text_input("",
                placeholder="Demande une question .......",
                label_visibility="collapsed")
        with col_btn:
            envoyer = st.form_submit_button("Envoyer", use_container_width=True, type="primary")

    if envoyer and user_input.strip():
        st.session_state.messages_genai.append({"role": "user", "content": user_input})

        # Construire l'historique pour le LLM
        messages_llm = [{"role": "user", "content": CONTEXTE_RH + "\n\nQuestion : " + user_input}]

        with st.spinner("Génération en cours..."):
            try:
                import requests
                # Essai Mistral API
                MISTRAL_KEY = st.secrets.get("MISTRAL_API_KEY", "")
                if MISTRAL_KEY:
                    resp = requests.post(
                        "https://api.mistral.ai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {MISTRAL_KEY}",
                                 "Content-Type": "application/json"},
                        json={"model": "mistral-small-latest",
                              "messages": messages_llm,
                              "max_tokens": 500},
                        timeout=30)
                    if resp.status_code == 200:
                        reponse = resp.json()["choices"][0]["message"]["content"]
                    else:
                        raise Exception(f"Mistral {resp.status_code}")
                else:
                    raise Exception("Pas de clé API")

            except Exception as e:
                # Réponses locales intelligentes si pas d'API
                q = user_input.lower()
                if "risque" in q and ("combien" in q or "nombre" in q):
                    reponse = f"Selon notre modèle XGBoost, **{n_risque} employés** présentent un risque de départ (score ≥ {seuil:.0%}), dont **{n_critique} en niveau critique** (risque ≥ 70%). Cela représente {n_risque/n*100:.1f}% de l'effectif total."
                elif "département" in q or "department" in q:
                    reponse = f"Le département le plus à risque est **{dept_max}** avec le taux d'attrition le plus élevé. Je recommande d'y concentrer les actions de rétention en priorité."
                elif "cause" in q or "facteur" in q or "pourquoi" in q:
                    reponse = "Les principales causes de départ identifiées par notre modèle SHAP sont :\n1. **Heures supplémentaires** (OverTime) — +20 pts de risque\n2. **Salaire mensuel** insuffisant\n3. **Années sans promotion** (> 3 ans)\n4. **Satisfaction au travail** faible (Low/Medium)\n5. **Distance domicile-travail** élevée"
                elif "coût" in q or "cout" in q or "financier" in q:
                    cout = int(df["MonthlyIncome"].mean() * 6 * n_critique)
                    reponse = f"Le coût potentiel estimé pour les {n_critique} employés critiques est d'environ **{cout:,} €** (base : 6 mois de salaire moyen par départ). Une action préventive coûtant 20% de cette somme permettrait un ROI de 400%."
                elif "recommand" in q or "action" in q or "conseil" in q:
                    reponse = "Mes 3 recommandations prioritaires :\n1. **Entretiens immédiats** pour les employés critiques — sous 48h\n2. **Réduire les heures supplémentaires** dans les départements à risque\n3. **Programme de promotions** pour les employés sans avancement depuis > 3 ans"
                else:
                    reponse = f"Bonjour ! Je suis votre assistant RH Analytics. Notre base contient {n} employés avec un taux d'attrition de {taux:.1f}%. Actuellement {n_risque} employés sont à risque. Posez-moi une question précise sur vos données RH !"

        st.session_state.messages_genai.append({"role": "assistant", "content": reponse})
        st.rerun()

    # Bouton vider le chat
    if st.session_state.messages_genai:
        if st.button(" Vider la conversation", key="clear_chat"):
            st.session_state.messages_genai = []
            st.rerun()