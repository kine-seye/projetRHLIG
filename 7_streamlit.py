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

# ── INITIALISATION DU THÈME (une seule fois) ──────────────────────────────────
if "mode_sombre" not in st.session_state:
    st.session_state.mode_sombre = True

if "page_actuelle" not in st.session_state:
    st.session_state.page_actuelle = "Accueil"

# ── PALETTE FIXE (accents, états — ne changent pas selon le thème) ────────────
VC  = "#00C896"   # Vert   — Stable / Faible
RC  = "#FF4B6E"   # Rouge  — Critique / Parti
BC  = "#4A9EF5"   # Bleu   — Info / Neutre
OC  = "#FFD166"   # Or     — Seuil / Modéré
PC  = "#9B72F5"   # Violet — SHAP / GenAI
OGC = "#FF8C42"   # Orange — Élevé

# ── COULEURS DYNAMIQUES SELON LE THÈME ───────────────────────────────────────
if st.session_state.mode_sombre:
    FOC = "#0F1923"   # Fond global
    CAC = "#1A2535"   # Cartes / Sidebar
    GRC = "#243044"   # Sous-blocs / Graphiques
    TXC = "#E8F0FE"   # Texte principal
    T2C = "#8FA3BF"   # Texte secondaire
    BOC = "#3A4F6A"   # Bordures
    GIC = "#2A3A50"   # Grille graphiques
else:
    FOC = "#F8FAFC"
    CAC = "#FFFFFF"
    GRC = "#F1F5F9"
    TXC = "#0F172A"   # Bleu-noir très sombre
    T2C = "#475569"   # Gris foncé
    BOC = "#CBD5E0"
    GIC = "#E2E8F0"

# ── PARAMÈTRES PLOTLY ─────────────────────────────────────────────────────
LAY = dict(
    paper_bgcolor=CAC,
    plot_bgcolor=GRC,
    font=dict(color=TXC, family="Inter,sans-serif"))

def ax(t=""):
    return dict(
        title=t,
        gridcolor=GIC,
        showgrid=True,
        zeroline=False,
        tickfont=dict(color=T2C))

# ── CSS GLOBAL ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
* {{ font-family: 'Inter', sans-serif !important; }}

/* Correction espace blanc en haut */
.block-container {{
    padding-top: 0.5rem !important;
    margin-top: -2.5rem !important;
    padding-bottom: 2rem !important;
}}

/* Fond global */
.stApp {{ background-color: {FOC}; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {CAC} !important;
    border-right: 1px solid {BOC};
}}

/* Onglets */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {CAC};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {BOC};
}}
.stTabs [data-baseweb="tab"] {{ color: {T2C} !important; border-radius: 8px !important; }}
.stTabs [aria-selected="true"] {{ background-color: {GRC} !important; color: {TXC} !important; }}

/* Selectbox / Inputs */
.stSelectbox>div>div,
.stTextInput>div>div,
.stMultiSelect>div>div {{
    background-color: {GRC} !important;
    border: 1px solid {BOC} !important;
    border-radius: 8px !important;
}}
div[data-baseweb="select"] div,
.stSelectbox span,
input {{ color: {TXC} !important; }}
svg {{ fill: {T2C} !important; }}

/* Labels */
div[data-testid="stWidgetLabel"] p {{
    color: {TXC} !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}}

/* Bouton primaire */
.stButton>button[kind="primary"] {{
    background: linear-gradient(135deg, {BC}, {PC}) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
}}

/* Titres */
h1, h2, h3, h4 {{ color: {TXC} !important; }}
footer {{ visibility: hidden; }}

/* Composants cartes */
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
}}
</style>
""", unsafe_allow_html=True)

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

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Toggle mode sombre/clair
    st.session_state.mode_sombre = st.toggle(
        "🌓 Mode Sombre",
        value=st.session_state.mode_sombre)

    st.markdown(f"""
    <div style="text-align:center;padding:12px 0 8px;">
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
    try:
        idx_p = PAGES.index(st.session_state.page_actuelle)
    except ValueError:
        idx_p = 0

    nav = st.radio("Menu", PAGES, index=idx_p, label_visibility="collapsed")
    st.session_state.page_actuelle = nav


 
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

    # ── En-tête ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="pg" style="display:flex;align-items:center;gap:16px;">
      <div style="font-size:40px;">🤖</div>
      <div>
        <div class="pt">Assistant RH Intelligent</div>
        <div class="ps">Propulsé par Mistral AI — Analyse vos données RH en temps réel</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Métriques rapides ─────────────────────────────────────────────────────
    n_critique = int((df["Niveau"] == "Critique").sum())
    n_risque   = int((df["Probabilite"] >= seuil).sum())
    d_max      = df.groupby("Department")["Attrition"].mean().idxmax()

   

    # ── Contexte RH envoyé à Mistral ─────────────────────────────────────────
    CONTEXTE_RH = f"""Tu es un expert Senior en Ressources Humaines et People Analytics.
Tu analyses les données RH d'une entreprise de {n} employés.

DONNÉES RÉELLES :
- Taux d'attrition : {taux:.1f}% ({int(taux*n/100)} départs sur {n})
- Employés à risque de départ : {n_risque} (score >= {seuil:.0%})
- Employés en niveau CRITIQUE : {n_critique} (risque >= 70%)
- Département le plus touché : {d_max}
- Modèle IA utilisé : XGBoost | F1={F1:.4f} | AUC={AUC:.4f}
- Facteurs de risque principaux (SHAP) : Heures supplémentaires, Salaire bas, Manque de promotion, Faible satisfaction

RÈGLES DE RÉPONSE :
- Réponds toujours en français
- Sois concis, professionnel et orienté solutions
- Utilise les données réelles ci-dessus dans tes réponses
- Donne des actions concrètes et mesurables
- Utilise des emojis pour structurer visuellement"""

    # ── Historique du chat ────────────────────────────────────────────────────
    if "messages_genai" not in st.session_state:
        st.session_state.messages_genai = [
            {
                "role": "assistant",
                "content": (
                    f"Bonjour ! 👋 Je suis votre **Assistant RH** basé sur l'IA.\n\n"
                    f"J'ai analysé les données de vos **{n:,} employés** :\n"
                    f"- 🔴 **{n_critique} employés critiques** nécessitent une action immédiate\n"
                    f"- ⚠️ **{n_risque} employés** sont à risque de départ\n"
                    f"- 📊 Taux d'attrition actuel : **{taux:.1f}%**\n\n"
                    f"Comment puis-je vous aider ?"
                )
            }
        ]

    # ── Suggestions de questions ──────────────────────────────────────────────
    st.markdown(f'<div style="font-size:12px;font-weight:600;color:{T2C};margin-bottom:8px;text-transform:uppercase;letter-spacing:1px;">Questions fréquentes</div>', unsafe_allow_html=True)

    suggestions = [
        ("🔴 Employés critiques",     "Combien d'employés sont en niveau critique et quelles actions urgentes recommandes-tu ?"),
        ("🏢 Département à risque",   "Quel département est le plus à risque et pourquoi ?"),
        ("📋 Causes de départ",       "Quelles sont les 5 principales causes de départ selon le modèle SHAP ?"),
        ("💰 Coût du turnover",        "Quel est le coût financier estimé du turnover actuel ?"),
        ("🎯 Plan de rétention",       "Propose-moi un plan de rétention sur 90 jours pour les employés critiques."),
        ("📈 Performance du modèle",   "Comment interpréter nos métriques F1 et AUC-ROC ?"),
    ]

    col_s = st.columns(3)
    for i, (label, question) in enumerate(suggestions):
        with col_s[i % 3]:
            if st.button(label, use_container_width=True, key=f"sug_{i}"):
                st.session_state.chat_trigger = question

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Affichage conversation ────────────────────────────────────────────────
    
    for msg in st.session_state.messages_genai:
        with st.chat_message(msg["role"], avatar=None):
            st.markdown(msg["content"])
    # ── Zone de saisie ────────────────────────────────────────────────────────
    prompt_input = st.chat_input("Posez votre question RH ici...")

    # Déclencher via suggestion
    if st.session_state.get("chat_trigger"):
        prompt_input = st.session_state.pop("chat_trigger")

    if prompt_input:
        # Message utilisateur
        st.session_state.messages_genai.append({"role": "user", "content": prompt_input})
        with st.chat_message("user", avatar=None):
            st.markdown(prompt_input)

        # Réponse assistant
        with st.chat_message("assistant", avatar=None):
            with st.spinner("Analyse en cours..."):
                reponse     = ""
                source_info = ""

                # ── Appel Mistral API ─────────────────────────────────────────
                try:
                    import requests
                    M_KEY = st.secrets.get("MISTRAL_API_KEY", "")
                    if not M_KEY:
                        raise Exception("Clé absente")

                    resp = requests.post(
                        "https://api.mistral.ai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {M_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model"      : "mistral-small-latest",
                            "messages"   : [
                                {"role": "system", "content": CONTEXTE_RH},
                                {"role": "user",   "content": prompt_input}
                            ],
                            "max_tokens" : 600,
                            "temperature": 0.6
                        },
                        timeout=30)

                    if resp.status_code == 200:
                        reponse     = resp.json()["choices"][0]["message"]["content"]
                        source_info = "✅ Mistral AI"
                    else:
                        raise Exception(f"Erreur {resp.status_code}")

                # ── Réponses locales intelligentes (fallback) ─────────────────
                except Exception:
                    source_info = "⚠️ Mode local (ajoutez MISTRAL_API_KEY dans les secrets)"
                    q = prompt_input.lower()

                    if any(x in q for x in ["critique", "urgent", "combien", "nombre", "risque"]):
                        reponse = (
                            f"### 🔴 Situation Critique\n\n"
                            f"Notre modèle XGBoost identifie :\n"
                            f"- **{n_critique} employés CRITIQUES** (risque ≥ 70%)\n"
                            f"- **{n_risque} employés à risque** au total (seuil {seuil:.0%})\n"
                            f"- Soit **{n_risque/n*100:.1f}%** de l'effectif\n\n"
                            f"**Actions urgentes :**\n"
                            f"1. 📅 Planifier des entretiens sous **48h** pour les {n_critique} critiques\n"
                            f"2. 📋 Préparer des plans de rétention individualisés\n"
                            f"3. 👥 Alerter les managers concernés dès aujourd'hui"
                        )
                    elif any(x in q for x in ["département", "department", "service"]):
                        reponse = (
                            f"### 🏢 Analyse par Département\n\n"
                            f"Le département **{d_max}** est le plus touché.\n\n"
                            f"**Recommandations :**\n"
                            f"1. 🔍 Audit RH spécifique au département {d_max}\n"
                            f"2. 💬 Focus group avec les employés du département\n"
                            f"3. 📊 Analyse des heures sup et satisfaction\n"
                            f"4. 🎯 Programme de rétention ciblé sur ce département"
                        )
                    elif any(x in q for x in ["cause", "facteur", "pourquoi", "raison", "shap"]):
                        reponse = (
                            f"### 📋 Top 5 Causes de Départ (SHAP)\n\n"
                            f"1. 🕐 **Heures supplémentaires** — +20 pts de risque\n"
                            f"   → Réduire la charge, limiter les HS à 10%\n\n"
                            f"2. 💰 **Salaire insuffisant** — fort impact direct\n"
                            f"   → Révision salariale pour les profils à risque\n\n"
                            f"3. 📅 **Pas de promotion depuis 3+ ans** — démotivation\n"
                            f"   → Programme promotion accéléré\n\n"
                            f"4. 😞 **Satisfaction travail faible** (Low/Medium)\n"
                            f"   → Entretiens 1-to-1 réguliers\n\n"
                            f"5. 🏠 **Distance domicile élevée**\n"
                            f"   → Politique télétravail flexible"
                        )
                    elif any(x in q for x in ["coût", "cout", "financier", "argent", "euro", "budget"]):
                        cout_est    = int(df["MonthlyIncome"].mean() * 6 * n_critique)
                        cout_action = int(cout_est * 0.15)
                        roi         = int((cout_est - cout_action) / cout_action * 100)
                        reponse = (
                            f"### 💰 Analyse Financière du Turnover\n\n"
                            f"**Coût potentiel estimé :**\n"
                            f"- {n_critique} employés critiques × 6 mois de salaire\n"
                            f"- **Total estimé : {cout_est:,} €**\n\n"
                            f"**ROI des actions RH :**\n"
                            f"- Investissement recommandé : **{cout_action:,} €**\n"
                            f"- Économie potentielle : **{cout_est - cout_action:,} €**\n"
                            f"- ROI estimé : **{roi}%**\n\n"
                            f"💡 Chaque euro investi en rétention rapporte ~{roi//100}€"
                        )
                    elif any(x in q for x in ["plan", "rétention", "retention", "stratégie", "90"]):
                        reponse = (
                            f"### 🎯 Plan de Rétention 90 Jours\n\n"
                            f"**Phase 1 — Urgence (0-30 jours)**\n"
                            f"- Entretiens immédiats : {n_critique} critiques\n"
                            f"- Suppression heures supplémentaires excessives\n"
                            f"- Revalorisation salariale ciblée\n\n"
                            f"**Phase 2 — Stabilisation (30-60 jours)**\n"
                            f"- Plans de promotion accélérés\n"
                            f"- Programme bien-être et WLB\n"
                            f"- Politique télétravail flexible\n\n"
                            f"**Phase 3 — Mesure (60-90 jours)**\n"
                            f"- Suivi mensuel scores de risque\n"
                            f"- Bilan ROI actions mises en place\n"
                            f"- Ajustement du plan selon résultats"
                        )
                    elif any(x in q for x in ["f1", "auc", "modèle", "performance", "métrique"]):
                        reponse = (
                            f"### 📈 Performance du Modèle XGBoost\n\n"
                            f"**Métriques actuelles :**\n"
                            f"- F1-Score : **{F1:.4f}** (équilibre précision/rappel)\n"
                            f"- AUC-ROC  : **{AUC:.4f}** ✅ Excellent (> 0.80)\n"
                            f"- Seuil optimal : **{seuil:.2f}**\n\n"
                            f"**Interprétation :**\n"
                            f"- AUC > 0.80 = le modèle discrimine très bien\n"
                            f"- F1 faible = déséquilibre des classes (16% attrition)\n\n"
                            f"**Pour améliorer :**\n"
                            f"1. Plus de données historiques\n"
                            f"2. Ensemble XGBoost + LightGBM\n"
                            f"3. Features engineering avancé"
                        )
                    else:
                        reponse = (
                            f"### 👋 À votre service !\n\n"
                            f"Je peux analyser vos données RH sur ces thèmes :\n\n"
                            f"- 🔴 **Employés critiques** — {n_critique} identifiés\n"
                            f"- 🏢 **Département à risque** — {d_max}\n"
                            f"- 📋 **Causes de départ** — analyse SHAP\n"
                            f"- 💰 **Coût du turnover** — impact financier\n"
                            f"- 🎯 **Plan de rétention** — 90 jours\n"
                            f"- 📈 **Performance modèle** — F1={F1:.4f}\n\n"
                            f"Posez votre question !"
                        )

            st.markdown(reponse)
            st.caption(source_info)

        st.session_state.messages_genai.append({"role": "assistant", "content": reponse})
        st.rerun()

    # ── Bouton vider ──────────────────────────────────────────────────────────
    if len(st.session_state.messages_genai) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        _, col_clear, _ = st.columns([3, 2, 3])
        with col_clear:
            if st.button("🗑️ Nouvelle conversation", use_container_width=True, key="clear_chat"):
                st.session_state.messages_genai = []
                st.rerun()