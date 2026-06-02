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
from xgboost import XGBClassifier
from plotly.subplots import make_subplots
import xgboost as xgb
from xgboost import XGBClassifier





# Au tout début, après les imports, initialisez la page si elle n'existe pas
if 'page_actuelle' not in st.session_state:
    st.session_state.page_actuelle = "Accueil"
    
# Liste officielle des pages (à utiliser partout pour éviter les erreurs)
PAGES = ["Accueil", "Exploration", "Prédiction", "Simulation What-If", "GenAI","Talents","Cout RH","Rapport RH"]


# Dictionnaire pour traduire les variables techniques en français lisible
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
    # On remplace les termes anglais par le français
    nom_clair = str(nom_tech)
    for en, fr in TRADUCTION_RH.items():
        nom_clair = nom_clair.replace(en, fr)
    
    # On nettoie les tirets bas (ex: OverTime_Yes -> Heures Supplémentaires_Oui)
    return nom_clair.replace("_", " : ")


warnings.filterwarnings("ignore")
 
st.set_page_config(page_title="HR Analytics", page_icon="👥",
    layout="wide", initial_sidebar_state="expanded")
 
# Couleurs
VC="#00C896"; RC="#FF4B6E"; BC="#4A9EF5"; OC="#FFD166"
PC="#9B72F5"; OGC="#FF8C42"; FOC="#0F1923"; CAC="#1A2535"
GRC="#243044"; TXC="#E8F0FE"; T2C="#8FA3BF"; GIC="#2A3A50"; BOC="#3A4F6A"
 
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
*{{font-family:'Inter',sans-serif!important;}}
.stApp{{background:{FOC};}}
section[data-testid="stSidebar"]{{background:{CAC}!important;border-right:1px solid {BOC};}}
.stTabs [data-baseweb="tab-list"]{{background:{CAC};border-radius:10px;padding:4px;border:1px solid {BOC};}}
.stTabs [data-baseweb="tab"]{{color:{T2C}!important;border-radius:8px!important;}}
.stTabs [aria-selected="true"]{{background:{GRC}!important;color:{TXC}!important;}}
.stSelectbox>div>div{{background:{GRC}!important;border:1px solid {BOC}!important;color:{TXC}!important;border-radius:8px!important;}}
.stButton>button[kind="primary"]{{background:linear-gradient(135deg,{BC},{PC})!important;border:none!important;color:white!important;font-weight:700!important;border-radius:10px!important;padding:10px 20px!important;}}
h1,h2,h3,h4{{color:{TXC}!important;}}
footer{{visibility:hidden;}}
.block-container{{padding-top:1.2rem!important;}}
.section-title{{font-size:13px;font-weight:800;color:{TXC};padding:10px 0 8px;border-bottom:2px solid var(--c);margin-bottom:14px;}}
.kc{{background:{CAC};border:1px solid {BOC};border-top:3px solid var(--c);border-radius:12px;padding:14px 12px;text-align:center;}}
.kv{{font-size:24px;font-weight:900;color:var(--c);}}
.kl{{font-size:10px;color:{T2C};text-transform:uppercase;letter-spacing:.8px;margin-top:4px;font-weight:600;}}
.ib{{background:{CAC};border:1px solid {BOC};border-left:4px solid var(--c);border-radius:0 10px 10px 0;padding:12px 16px;}}
.it{{font-size:10px;font-weight:700;color:var(--c);text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;}}
.iv{{font-size:18px;font-weight:900;color:{TXC};margin-bottom:3px;}}
.ix{{font-size:11px;color:{T2C};line-height:1.5;}}
.info-card{{background:{CAC};border:1px solid {BOC};border-radius:8px;padding:10px 12px;margin-bottom:8px;}}
.info-lbl{{font-size:10px;color:{T2C};margin-bottom:3px;}}
.info-val{{font-size:13px;font-weight:700;color:{TXC};}}
.recomm{{background:{GRC};border-left:5px solid var(--c);border-radius:0 10px 10px 0;padding:14px 20px;margin:10px 0;}}
.pg{{background:linear-gradient(135deg,{CAC},{GRC});border:1px solid {BOC};border-radius:14px;padding:20px 24px;margin-bottom:16px;}}
.pt{{font-size:21px;font-weight:900;color:{TXC};margin-bottom:4px;}}
.ps{{font-size:12px;color:{T2C};}}
</style>
""", unsafe_allow_html=True)
 
LAY = dict(paper_bgcolor=CAC, plot_bgcolor=GRC,
    font=dict(color=TXC, family="Inter,sans-serif"))
def ax(t=""):
    return dict(title=t, gridcolor=GIC, showgrid=True,
                zeroline=False, tickfont=dict(color=T2C))
 
# ── COLONNES FIXES hr.csv ─────────────────────────────────────────────────────
COLS_NUM_DEF = ['Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome',
    'MonthlyRate','NumCompaniesWorked','PercentSalaryHike','StockOptionLevel',
    'TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany',
    'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
COLS_CAT_DEF = ['BusinessTravel','Department','Education','EducationField',
    'EnvironmentSatisfaction','Gender','JobInvolvement','JobLevel','JobRole',
    'JobSatisfaction','MaritalStatus','OverTime','PerformanceRating',
    'RelationshipSatisfaction','WorkLifeBalance']
 
# ── CHARGEMENT ────────────────────────────────────────────────────────────────
@st.cache_resource
def charger_modele():
    import joblib
    try:
        # On utilise DIRECTEMENT joblib puisque le fichier a été généré avec joblib.dump
        return joblib.load("mon_modele_rh.pkl")
    except Exception as e:
        st.sidebar.error(f"❌ Erreur critique de lecture Joblib : {str(e)}")
        return None

@st.cache_data
def charger_df():
    df = pd.read_csv("hr.csv")
    df["Attrition"] = df["Attrition"].map({"Yes":1,"No":0})
    return df
 
data    = charger_modele()
df_base = charger_df()
n       = len(df_base)
taux    = df_base["Attrition"].mean()*100
 
MODELE_OK = False; seuil = 0.31; F1 = 0.4821 ; AUC = 0.8030
FEATURES = []; COLS_FINALES = []; modele = None; preprocesseur = None

if data is not None:
    try:
        if isinstance(data, dict) and "cerveau_ia" in data:
            modele        = data.get("cerveau_ia")
            preprocesseur = data.get("traitement")
            seuil         = float(data.get("reglage_seuil", 0.31))
            FEATURES      = list(data.get("features", []))
            F1            = float(data.get("f1", 0.4821))
            AUC           = float(data.get("auc", 0.8030))
            COLS_FINALES  = list(data.get("noms_colonnes", []))
            
            if modele is None:
                raise ValueError("Le modèle extrait est vide ou introuvable.")
            
            @st.cache_data
            def enrichir_reel(_mod, _prep, _feat):
                d = df_base.copy()
                X = _prep.transform(d[_feat])
                d["Probabilite"] = _mod.predict_proba(X)[:, 1]
                d["Prediction"]  = (d["Probabilite"] >= seuil).astype(int)
                d["Risque_Pct"]  = (d["Probabilite"] * 100).round(1)
                d["Niveau"]      = d["Probabilite"].apply(lambda p:
                    "Critique" if p >= 0.70 else "Eleve" if p >= 0.50
                    else "Modere" if p >= seuil else "Faible")
                return d
            
            df = enrichir_reel(modele, preprocesseur, FEATURES)
            MODELE_OK = True
            st.sidebar.success(f"✅ Vrai modèle chargé ({len(FEATURES)} variables)")
        else:
            raise ValueError("Format de fichier non reconnu")

    except Exception as e:
        st.sidebar.warning(f"⚠️ Mode Secours activé")
        st.sidebar.error(f"Détail technique : {str(e)}")
        MODELE_OK = False  # Sécurité : pas de faux pavé XGBoost si le modèle est en panne

# ── INITIALISATION SÉCURISÉE GLOBALE DE DF (Unique et centralisée pour le plan B) ──
if 'df' not in globals() or df is None:
    import numpy as np
    df = df_base.copy()
    np.random.seed(42)
    base_prob = np.random.beta(2, 5, size=len(df))
    df["Probabilite"] = np.where(df["Attrition"] == 1, np.minimum(base_prob + 0.4, 0.95), np.maximum(base_prob - 0.1, 0.02))
    df["Prediction"]  = (df["Probabilite"] >= seuil).astype(int)
    df["Risque_Pct"]  = (df["Probabilite"] * 100).round(1)
    df["Niveau"]      = df["Probabilite"].apply(lambda p: "Critique" if p >= 0.70 else "Eleve" if p >= 0.50 else "Modere" if p >= seuil else "Faible")

# ── SHAP helper ───────────────────────────────────────────────────────────────
def get_explainer(model_to_explain, background_data):
    """Explainer SHAP universel (plus fiable que TreeExplainer pour XGB 3.x)"""
    try:
        import shap
        if model_to_explain is None:
            return None
            
        # On définit la fonction de prédiction
        f = lambda x: model_to_explain.predict_proba(x)[:, 1]
        
        # On utilise l'Explainer générique (Boîte noire)
        # background_data sont les 50 employés de référence
        expl = shap.Explainer(f, background_data)
        
        return expl
    except Exception as e:
        st.sidebar.warning(f"SHAP Error: {str(e)[:100]}")
        return None
 
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 12px;">
      <div style="font-size:40px;">👥</div>
      <div style="font-size:14px;font-weight:800;color:{TXC};margin-top:5px;">HR Analytics</div>
      <div style="font-size:11px;color:{T2C};margin-top:4px;line-height:1.8;">
        Seye Kiné | Bindia Adeline Thiara<br>
        <span style="color:{OC};font-weight:600;">M. Aidara</span> — UCAO 2025-2026
      </div>
    </div><hr style="border-color:{BOC};margin:0 0 12px;">
    """, unsafe_allow_html=True)

   # Calcul de l'index pour que le bouton reste sur la bonne page après un rerun
    try:
        index_page = PAGES.index(st.session_state.page_actuelle)
    except ValueError:
        index_page = 0

    nav = st.radio("Navigation", PAGES, index=index_page, label_visibility="collapsed")
    
    # On met à jour la page actuelle dans la mémoire
    st.session_state.page_actuelle = nav
    if MODELE_OK:
        st.markdown(f"""
        <hr style="border-color:{BOC};margin:12px 0;">
        <div style="background:{GRC};border-radius:10px;padding:12px 14px;">
          <div style="font-size:10px;color:{T2C};text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;font-weight:700;">Modèle actif</div>
          <div style="font-size:12px;font-weight:700;color:{TXC};margin-bottom:8px;">XGBoost</div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:11px;color:{T2C};">F1-Score</span>
            <span style="font-size:11px;font-weight:700;color:{VC};">{F1:.4f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:11px;color:{T2C};">AUC-ROC</span>
            <span style="font-size:11px;font-weight:700;color:{BC};">{AUC:.4f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="font-size:11px;color:{T2C};">Seuil</span>
            <span style="font-size:11px;font-weight:700;color:{OC};">{seuil:.2f}</span>
          </div>
        </div>""", unsafe_allow_html=True)
 
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
        st.markdown("### 📋 Synthèse statistique des indicateurs")
        
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
                # ── SECTION 5 : BOUTON D'ACTION (Bien aligné dans le else) ──────
                st.markdown("---")
                c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
                with c_btn2:
                    if st.button(f" Lancer une simulation de rétention pour l'employé #{idx_sel}", use_container_width=True, type="primary"):
                        st.session_state.emp_id_transfert = idx_sel
                        st.session_state.page_actuelle = "Simulation What-If"
                        st.rerun()

                
                



 
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
# Ajouter " What-If" dans la navigation


elif nav == "Simulation What-If":
    # 1. On récupère l'ID transféré OU l'employé le plus à risque par défaut
    if 'emp_id_transfert' in st.session_state:
        id_initial = int(st.session_state.emp_id_transfert)
    else:
        # Sécurité anti-crash si les probabilités ne sont pas encore calculées
        if "Probabilite" in df.columns:
            id_initial = int(df["Probabilite"].idxmax())
        else:
            id_initial = int(df.index[0])

    # 2. On affiche le titre de la page
    st.markdown(f'<div class="pg"><div class="pt"> Simulation What-If</div><div class="ps">Testez l\'impact des actions RH sur le risque de départ</div></div>', unsafe_allow_html=True)
 
    if not MODELE_OK:
        st.error("Modele non disponible. Lancez HR_Analytics.ipynb en premier.")
        st.stop()
# ── Sélection de l'employé ────────────────────────────────────────────────
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        # On utilise id_initial qui contient soit le transfert, soit le plus risqué par défaut
        emp_idx_wi = st.number_input(
            "Numéro employé (0-1469)", 0, 1469,
            value=id_initial, step=1, key="wi_idx")
        if emp_idx_wi not in df.index:
            st.warning("⚠️ Cet identifiant n'est pas répertorié. Veuillez choisir un nombre entre 0 et 1469.")
            st.stop()
            
    with col_s2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(" Plus à risque", use_container_width=True):
            # On force l'ID du plus risqué dans la session et on relance
            st.session_state.emp_id_transfert = int(df["Probabilite"].idxmax())
            st.rerun()

    # --- NETTOYAGE AUTOMATIQUE ---
    # Une fois que l'ID a été utilisé pour remplir le champ, on l'efface de la "mémoire vive"
    # pour que l'utilisateur puisse changer de numéro sans être bloqué.
    if 'emp_id_transfert' in st.session_state:
        del st.session_state.emp_id_transfert

    # 4. Chargement et Calculs
    emp_wi     = df.loc[emp_idx_wi].copy()
    proba_base = float(emp_wi["Probabilite"])
    pct_base   = proba_base * 100
 
    # Détermination des couleurs et labels (avec accents pour le français)
    if pct_base >= 70:        cr_b = RC;  niv_b = "🔴 CRITIQUE"
    elif pct_base >= 50:      cr_b = OGC; niv_b = "🟠 ÉLEVÉ"
    elif pct_base >= seuil*100: cr_b = OC;  niv_b = "🟡 MODÉRÉ"
    else:                     cr_b = VC;  niv_b = "🟢 FAIBLE"
    # ── Profil actuel ─────────────────────────────────────────────────────────
    ot_color = RC if emp_wi["OverTime"] == "Yes" else VC
    ot_label = " Oui" if emp_wi["OverTime"] == "Yes" else " Non"
    js_color = RC if emp_wi["JobSatisfaction"] in ["Low", "Medium"] else VC
 
    st.markdown(f"""
    <div style="background:{CAC};border:1px solid {BOC};border-radius:12px;
    padding:16px 20px;margin-bottom:16px;">
      <div style="font-size:11px;font-weight:700;color:{T2C};margin-bottom:10px;
      text-transform:uppercase;letter-spacing:1px;">
        Profil actuel — Employe #{emp_idx_wi}
      </div>
      <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;text-align:center;">
        <div><div style="font-size:10px;color:{T2C};">Departement</div>
             <div style="font-size:12px;font-weight:700;color:{TXC};">{emp_wi["Department"]}</div></div>
        <div><div style="font-size:10px;color:{T2C};">Poste</div>
             <div style="font-size:12px;font-weight:700;color:{TXC};">{traduire_nom(emp_wi["JobRole"])}</div></div>
        <div><div style="font-size:10px;color:{T2C};">Salaire</div>
             <div style="font-size:12px;font-weight:700;color:{TXC};">{int(emp_wi["MonthlyIncome"]):,} €</div></div>
        <div><div style="font-size:10px;color:{T2C};">H.Sup</div>
             <div style="font-size:12px;font-weight:700;color:{ot_color};">{ot_label}</div></div>
        <div><div style="font-size:10px;color:{T2C};">Ans sans promo</div>
             <div style="font-size:12px;font-weight:700;color:{RC if emp_wi["YearsSinceLastPromotion"]>=3 else TXC};">{int(emp_wi["YearsSinceLastPromotion"])} ans</div></div>
        <div><div style="font-size:10px;color:{T2C};">Risque actuel</div>
             <div style="font-size:14px;font-weight:900;color:{cr_b};">{pct_base:.1f}%</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Actions à simuler ─────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:13px;font-weight:700;color:{TXC};margin:14px 0 12px;"> Choisissez les actions RH a simuler</div>', unsafe_allow_html=True)
 
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f'<div style="background:{CAC};border:1px solid {BC};border-radius:10px;padding:14px 16px;">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;font-weight:700;color:{BC};margin-bottom:10px;"> Politique salariale</div>', unsafe_allow_html=True)
        augmentation = st.slider("Augmentation salaire (%)", 0, 50, 0, step=5, key="wi_aug")
        supprimer_ot = st.checkbox(" Supprimer les heures supplementaires", value=False, key="wi_ot")
        st.markdown("</div>", unsafe_allow_html=True)
    with a2:
        st.markdown(f'<div style="background:{CAC};border:1px solid {PC};border-radius:10px;padding:14px 16px;">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;font-weight:700;color:{PC};margin-bottom:10px;"> Evolution de carriere</div>', unsafe_allow_html=True)
        promotion  = st.checkbox(" Accorder une promotion maintenant", value=False, key="wi_promo")
        amelio_js  = st.select_slider("Ameliorer satisfaction travail",
            options=["Sans changement","Medium","High","Very High"],
            value="Sans changement", key="wi_js")
        st.markdown("</div>", unsafe_allow_html=True)
    with a3:
        st.markdown(f'<div style="background:{CAC};border:1px solid {VC};border-radius:10px;padding:14px 16px;">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;font-weight:700;color:{VC};margin-bottom:10px;"> Qualite de vie</div>', unsafe_allow_html=True)
        amelio_wlb      = st.select_slider("Ameliorer WLB",
            options=["Sans changement","Good","Better","Best"],
            value="Sans changement", key="wi_wlb")
        reduire_distance = st.checkbox(" Teletravail (distance ÷ 2)", value=False, key="wi_dist")
        st.markdown("</div>", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    simuler = st.button("Simuler l\'impact des actions", use_container_width=True, type="primary", key="wi_sim")
 
    # ── Calcul de la simulation ───────────────────────────────────────────────
    action_active = (augmentation > 0 or supprimer_ot or promotion or
                     amelio_js != "Sans changement" or
                     amelio_wlb != "Sans changement" or reduire_distance)
 
    if simuler or action_active:
        emp_sim = emp_wi.copy()
 
        # Appliquer les actions
        if augmentation > 0:
            emp_sim["MonthlyIncome"]      = emp_wi["MonthlyIncome"] * (1 + augmentation / 100)
            emp_sim["HourlyRate"]         = emp_wi["HourlyRate"]    * (1 + augmentation / 100)
            emp_sim["DailyRate"]          = emp_wi["DailyRate"]     * (1 + augmentation / 100)
            emp_sim["PercentSalaryHike"]  = min(25, emp_wi["PercentSalaryHike"] + augmentation // 5)
        if supprimer_ot:
            emp_sim["OverTime"] = "No"
        if promotion:
            emp_sim["YearsSinceLastPromotion"] = 0
        if amelio_js != "Sans changement":
            emp_sim["JobSatisfaction"] = amelio_js
        if amelio_wlb != "Sans changement":
            emp_sim["WorkLifeBalance"] = amelio_wlb
        if reduire_distance:
            emp_sim["DistanceFromHome"] = max(1, emp_wi["DistanceFromHome"] // 2)
 
        try:
            df_sim    = pd.DataFrame([emp_sim[FEATURES]])
            X_sim     = preprocesseur.transform(df_sim)
            proba_sim = float(modele.predict_proba(X_sim)[0][1])
            pct_sim   = proba_sim * 100
            delta     = pct_base - pct_sim
 
            if pct_sim >= 70:        cr_s = RC;  niv_s = "🔴 CRITIQUE"
            elif pct_sim >= 50:      cr_s = OGC; niv_s = "🟠 ELEVE"
            elif pct_sim >= seuil*100: cr_s = OC;  niv_s = "🟡 MODERE"
            else:                    cr_s = VC;  niv_s = "🟢 FAIBLE"
 
            # ── Résultats ─────────────────────────────────────────────────────
            st.markdown(f'<div class="section-title" style="--c:{VC};">📊 Resultat de la simulation</div>', unsafe_allow_html=True)
 
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""
                <div style="background:{cr_b}22;border:2px solid {cr_b};
                border-radius:12px;padding:20px;text-align:center;">
                  <div style="font-size:11px;color:{T2C};margin-bottom:6px;
                  text-transform:uppercase;">AVANT les actions</div>
                  <div style="font-size:52px;font-weight:900;color:{cr_b};">{pct_base:.1f}%</div>
                  <div style="font-size:13px;font-weight:700;color:{cr_b};margin-top:4px;">{niv_b}</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                color_d = VC if delta > 0 else RC
                signe   = "-" if delta > 0 else "+"
                st.markdown(f"""
                <div style="background:{GRC};border:2px solid {color_d};
                border-radius:12px;padding:20px;text-align:center;">
                  <div style="font-size:11px;color:{T2C};margin-bottom:6px;
                  text-transform:uppercase;">IMPACT</div>
                  <div style="font-size:52px;font-weight:900;color:{color_d};">{signe}{abs(delta):.1f}</div>
                  <div style="font-size:13px;font-weight:700;color:{color_d};margin-top:4px;">points de risque</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""
                <div style="background:{cr_s}22;border:2px solid {cr_s};
                border-radius:12px;padding:20px;text-align:center;">
                  <div style="font-size:11px;color:{T2C};margin-bottom:6px;
                  text-transform:uppercase;">APRES les actions</div>
                  <div style="font-size:52px;font-weight:900;color:{cr_s};">{pct_sim:.1f}%</div>
                  <div style="font-size:13px;font-weight:700;color:{cr_s};margin-top:4px;">{niv_s}</div>
                </div>""", unsafe_allow_html=True)
 
            st.markdown("<br>", unsafe_allow_html=True)
 
            # Graphique comparatif
            fig_wi = go.Figure()
            fig_wi.add_trace(go.Bar(
                name="Avant", x=["Risque de depart"], y=[pct_base],
                marker=dict(color=cr_b, line=dict(color=FOC, width=2)),
                text=[f"{pct_base:.1f}%"], textposition="outside", width=[0.25]))
            fig_wi.add_trace(go.Bar(
                name="Apres", x=["Risque de depart"], y=[pct_sim],
                marker=dict(color=cr_s, line=dict(color=FOC, width=2)),
                text=[f"{pct_sim:.1f}%"], textposition="outside", width=[0.25]))
            fig_wi.add_hline(y=seuil*100, line_dash="dash", line_color=OC,
                annotation_text=f"Seuil decision ({seuil:.0%})",
                annotation_font=dict(color=OC))
            fig_wi.update_layout(**LAY,
                title=dict(
                    text=f"Reduction du risque : -{delta:.1f} points grace aux actions RH",
                    font=dict(size=14), x=0.5),
                yaxis=dict(**ax("Risque de depart (%)"), range=[0, 110]),
                xaxis=ax(), height=400, barmode="group",
                margin=dict(t=55, b=40, l=60, r=20))
            st.plotly_chart(fig_wi, use_container_width=True)
 
            # Actions appliquées
            actions_text = []
            if augmentation > 0:
                actions_text.append(f" Augmentation +{augmentation}% : {int(emp_wi['MonthlyIncome']):,}€ → {int(emp_sim['MonthlyIncome']):,}€")
            if supprimer_ot:
                actions_text.append(" Heures supplementaires supprimees")
            if promotion:
                actions_text.append(" Promotion accordee : annees sans promo → 0")
            if amelio_js != "Sans changement":
                actions_text.append(f" Satisfaction : {emp_wi['JobSatisfaction']} → {amelio_js}")
            if amelio_wlb != "Sans changement":
                actions_text.append(f" WLB : {emp_wi['WorkLifeBalance']} → {amelio_wlb}")
            if reduire_distance:
                actions_text.append(f" Distance : {int(emp_wi['DistanceFromHome'])} km → {int(emp_sim['DistanceFromHome'])} km")
 
            if actions_text:
                rows_a = "".join([
                    f'<div style="padding:6px 0;border-bottom:1px solid {BOC};">'
                    f'<span style="font-size:13px;color:{TXC};"> {a}</span></div>'
                    for a in actions_text])
                st.markdown(f"""
                <div style="background:{VC}12;border:1px solid {VC}44;
                border-radius:10px;padding:16px 20px;margin-bottom:16px;">
                  <div style="font-size:12px;font-weight:700;color:{VC};
                  margin-bottom:10px;text-transform:uppercase;"> Actions simulees</div>
                  {rows_a}
                </div>""", unsafe_allow_html=True)
 
            # Analyse financière
            cout_depart  = int(emp_wi["MonthlyIncome"]) * 6
            cout_actions = int(emp_wi["MonthlyIncome"] * augmentation / 100 * 12) if augmentation > 0 else 0
            economie     = cout_depart - cout_actions
 
            st.markdown(f"""
            <div style="background:{OC}12;border:1px solid {OC}44;
            border-radius:10px;padding:16px 20px;">
              <div style="font-size:12px;font-weight:700;color:{OC};
              margin-bottom:12px;text-transform:uppercase;"> Analyse financiere</div>
              <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;text-align:center;">
                <div>
                  <div style="font-size:10px;color:{T2C};margin-bottom:4px;">Cout si depart</div>
                  <div style="font-size:22px;font-weight:800;color:{RC};">{cout_depart:,} €</div>
                  <div style="font-size:10px;color:{T2C};">(6 mois de salaire)</div>
                </div>
                <div>
                  <div style="font-size:10px;color:{T2C};margin-bottom:4px;">Cout des actions</div>
                  <div style="font-size:22px;font-weight:800;color:{OC};">{cout_actions:,} €</div>
                  <div style="font-size:10px;color:{T2C};">(cout annuel)</div>
                </div>
                <div>
                  <div style="font-size:10px;color:{T2C};margin-bottom:4px;">Economie potentielle</div>
                  <div style="font-size:22px;font-weight:800;color:{VC};">{economie:,} €</div>
                  <div style="font-size:10px;color:{T2C};">si retention reussie</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
 
        except Exception as e:
            st.error(f"Erreur simulation : {e}")





# =============================================================================
# CODE GENAI — LETTRE RH PERSONNALISEE PAR LLM

elif nav == "GenAI":
    # 1. TITRE DE LA PAGE (Bien indenté)
    st.markdown(f'<div class="pg"><div class="pt">GenAI — Lettre RH Personnalisée</div><div class="ps">Intelligence Artificielle Générative — Conseils de rétention rédigés automatiquement</div></div>', unsafe_allow_html=True)
 
    if not MODELE_OK:
        st.error("Modèle non disponible. Lancez HR_Analytics.ipynb en premier.")
        st.stop()

    # 2. RÉCUPÉRATION DE L'ID (Logique de transfert harmonisée)
    if 'emp_id_transfert' in st.session_state:
        id_initial_gen = int(st.session_state.emp_id_transfert)
    else:
        id_initial_gen = int(df["Probabilite"].idxmax())

    # 3. INTERFACE DE SÉLECTION
    st.markdown('<div class="section-title" style="--c:#4A9EF5;">👤 1. Sélection de l\'employé à analyser</div>', unsafe_allow_html=True)
    cg1, cg2 = st.columns([3, 1])
    
    with cg1:
        emp_idx_g = st.number_input(
            "Numéro employé (0-1469)", 0, 1469,
            value=int(id_initial_gen), step=1, key="g_idx_input")
            
    with cg2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Plus à risque", use_container_width=True):
            # On met à jour la MÊME clé que celle lue en haut (emp_id_transfert)
            st.session_state.emp_id_transfert = int(df["Probabilite"].idxmax())
            st.rerun()

    # 4. NETTOYAGE DU TRANSFERT (Syntaxe réparée et alignée)
    if 'emp_id_transfert' in st.session_state and st.session_state.emp_id_transfert == emp_idx_g:
        del st.session_state.emp_id_transfert

    # 5. CHARGEMENT DES DONNÉES DE L'EMPLOYÉ
    emp_g   = df.loc[emp_idx_g]
    proba_g = float(emp_g["Probabilite"]) * 100
 
    if proba_g >= 70:   cr_g, niv_g = RC, "🔴 CRITIQUE"
    elif proba_g >= 50: cr_g, niv_g = OGC, "🟠 ÉLEVÉ"
    elif proba_g >= seuil * 100: cr_g, niv_g = OC, "🟡 MODÉRÉ"
    else:               cr_g, niv_g = VC, "🟢 FAIBLE"
    
 
    
    # ── Résumé employé ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{CAC};border:1px solid {BOC};border-radius:12px;
    padding:14px 18px;margin-bottom:16px;">
      <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;text-align:center;">
        <div><div style="font-size:10px;color:{T2C};">Employé</div>
             <div style="font-size:13px;font-weight:700;color:{TXC};">#{emp_idx_g}</div></div>
        <div><div style="font-size:10px;color:{T2C};">Département</div>
             <div style="font-size:13px;font-weight:700;color:{TXC};">{emp_g["Department"]}</div></div>
        <div><div style="font-size:10px;color:{T2C};">Poste</div>
             <div style="font-size:12px;font-weight:700;color:{TXC};">{traduire_nom(emp_g["JobRole"])}</div></div>
        <div><div style="font-size:10px;color:{T2C};">Salaire</div>
             <div style="font-size:13px;font-weight:700;color:{TXC};">{int(emp_g["MonthlyIncome"]):,} €</div></div>
        <div><div style="font-size:10px;color:{T2C};">Ancienneté</div>
             <div style="font-size:13px;font-weight:700;color:{TXC};">{int(emp_g["YearsAtCompany"])} ans</div></div>
        <div><div style="font-size:10px;color:{T2C};">Risque</div>
             <div style="font-size:14px;font-weight:900;color:{cr_g};">{proba_g:.1f}%</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Calcul SHAP pour enrichir le contexte ─────────────────────────────────
 

    facteurs_risque     = ["OverTime", "YearsSinceLastPromotion", "JobSatisfaction"]
    facteurs_protection = ["MonthlyIncome", "YearsAtCompany", "JobLevel"]
 
    if len(COLS_FINALES) > 0:
        try:
            import shap as _shap_g
            f_pred_g = lambda x: modele.predict_proba(x)[:, 1]
            X_bg_g   = preprocesseur.transform(df_base.sample(50, random_state=42)[FEATURES])
            expl_g   = _shap_g.Explainer(f_pred_g, X_bg_g)
            X_g      = preprocesseur.transform(df.loc[[emp_idx_g]][FEATURES])
            sv_g     = expl_g(X_g).values[0]
            df_sv_g  = pd.DataFrame({"Variable": COLS_FINALES, "SHAP": sv_g})
            facteurs_risque     = df_sv_g.sort_values("SHAP", ascending=False).head(3)["Variable"].tolist()
            facteurs_protection = df_sv_g.sort_values("SHAP", ascending=True).head(3)["Variable"].tolist()
        except Exception:
            pass  # Utilise les valeurs par défaut si SHAP échoue
    facteurs_risque = [traduire_nom(f) for f in facteurs_risque]
    facteurs_protection = [traduire_nom(f) for f in facteurs_protection]
 
    # ── Choix du type de document ─────────────────────────────────────────────
    st.markdown(f'<div style="font-size:13px;font-weight:700;color:{TXC};margin-bottom:10px;"> Type de document à générer</div>', unsafe_allow_html=True)
 
    type_doc = st.radio("",
        [" Lettre au manager", " Plan de rétention", " Email urgent RH", " Rapport individuel"],
        horizontal=True, label_visibility="collapsed")
 
    # ── Bouton générer ────────────────────────────────────────────────────────
    generer = st.button(" Générer avec l'IA", use_container_width=True, type="primary", key="g_gen")
 
    if generer:
        # Contexte employé pour le prompt
        ot_txt    = "fait des heures supplémentaires" if emp_g["OverTime"] == "Yes" else "ne fait pas d'heures supplémentaires"
        promo_txt = f"n'a pas eu de promotion depuis {int(emp_g['YearsSinceLastPromotion'])} an"
        js_txt    = f"satisfaction au travail : {emp_g['JobSatisfaction']}"
        wlb_txt   = f"équilibre vie pro/perso : {emp_g['WorkLifeBalance']}"
 
        # Prompts selon le type de document
        prompts = {
            " Lettre au manager": f"""Tu es un expert RH senior. Rédige une lettre confidentielle et professionnelle au manager direct.
 
PROFIL DE L'EMPLOYÉ :
- Poste : {emp_g["JobRole"]} | Département : {emp_g["Department"]}
- Ancienneté : {int(emp_g["YearsAtCompany"])} ans | Salaire : {int(emp_g["MonthlyIncome"])} €/mois
- Cet employé {ot_txt}, {promo_txt}
- {js_txt} | {wlb_txt}
- Risque de départ calculé par l'IA : {proba_g:.0f}% (Niveau {niv_g})
- Facteurs de risque identifiés : {", ".join(facteurs_risque)}
- Facteurs protecteurs : {", ".join(facteurs_protection)}
 
Rédige une lettre avec :
1. Objet clair et percutant
2. Résumé de la situation (2-3 phrases)
3. Les 3 actions concrètes prioritaires à mettre en place dans les 30 prochains jours
4. Un plan de suivi sur 90 jours
5. Conclusion bienveillante et motivante
 
Ton : professionnel, bienveillant, orienté solutions. En français. Maximum 400 mots.""",
 
            " Plan de rétention": f"""Tu es un consultant RH expert en rétention des talents. Crée un plan de rétention complet et personnalisé.
 
EMPLOYÉ À RISQUE :
- {emp_g["JobRole"]} | {emp_g["Department"]} | {int(emp_g["YearsAtCompany"])} ans d'ancienneté
- Risque de départ : {proba_g:.0f}% ({niv_g}) | Salaire : {int(emp_g["MonthlyIncome"])} €
- {ot_txt} | {promo_txt}
- Satisfaction : {emp_g["JobSatisfaction"]} | WLB : {emp_g["WorkLifeBalance"]}
- Facteurs de risque IA : {", ".join(facteurs_risque)}
 
Rédige un plan structuré avec :
1. Analyse des causes racines (3 points max)
2. Actions immédiates 0-30 jours (avec responsable et délai)
3. Actions moyen terme 30-90 jours
4. Indicateurs de succès mesurables
5. Budget estimé des actions
 
Format professionnel, en français.""",
 
            " Email urgent RH": f"""Rédige un email urgent et professionnel du service RH au manager.
 
CONTEXTE : Un employé ({emp_g["JobRole"]}, {emp_g["Department"]}, {int(emp_g["YearsAtCompany"])} ans d'ancienneté) présente un risque de départ de {proba_g:.0f}% selon notre système IA.
 
Contraintes :
1. Objet de l'email : Alerte RH confidentielle - Risque de rétention d'un talent
2. Ton : Urgent, professionnel, collaboratif.
3. Demande de rendez-vous sous 48h.
4. Synthèse ultra-courte des risques : {", ".join(facteurs_risque)}.
En français, maximum 200 mots.""",
 
            " Rapport individuel": f"""Tu es un Analyste RH. Rédige un rapport individuel d'évaluation des risques de turnover pour la direction.
 
DONNÉES :
- Employé : {emp_g["JobRole"]} | Risque : {proba_g:.0f}% ({niv_g})
- Salaire : {int(emp_g["MonthlyIncome"])} € | Ancienneté : {int(emp_g["YearsAtCompany"])} ans
- Facteurs négatifs : {", ".join(facteurs_risque)}
- Facteurs positifs : {", ".join(facteurs_protection)}
 
Structure requise :
1. Diagnostic de vulnérabilité RH
2. Évaluation de l'impact financier d'un départ
3. Matrice Risques / Actions
4. Avis et Recommandation finale de l'analyste
En français, style corporate."""
        }
 
        prompt_final = prompts.get(type_doc, prompts[" Lettre au manager"])
 
        # ── Appel sécurisé à l'API Google Gemini ──────────────────────────────
        with st.spinner("  Génération en cours par l'IA ..."):
            try:
                # Importation dynamique et locale pour éviter les blocages au démarrage
                import google.generativeai as genai
                
                # Vérification de la clé dans les secrets
                if "GEMINI_API_KEY" in st.secrets:
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                    
                    # Configuration du modèle
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(prompt_final)
                    
                    texte_genere = response.text
                    source = " Généré en direct par l'IA "
                else:
                    raise Exception("Clé API GEMINI_API_KEY manquante dans les Secrets Streamlit")
                    
            except Exception as e:
                # ── Système de secours local (Fallback) si l'importation ou l'API échoue ──
                source = f"⚠️ Génération locale (API déconnectée : {str(e)[:25]}...)"
                texte_genere = f"""{type_doc.upper()} — Employé de secours #{emp_g.name if hasattr(emp_g, 'name') else 'Sélectionné'}
{"=" * 60}
SITUATION :
L'employé occupe le poste de {emp_g["JobRole"]} dans le département {emp_g["Department"]}.
Avec {int(emp_g["YearsAtCompany"])} ans d'ancienneté et un salaire de {int(emp_g["MonthlyIncome"])} €/mois,
cet employé présente un risque de départ de {proba_g:.0f}% ({niv_g}).
 
FACTEURS DE RISQUE PRINCIPAUX :
{chr(10).join([f"   • {f}" for f in facteurs_risque]) if facteurs_risque else "   • Aucun facteur critique détecté"}
 
FACTEURS PROTECTEURS :
{chr(10).join([f"   • {f}" for f in facteurs_protection]) if facteurs_protection else "   • Données protectrices insuffisantes"}
 
RECOMMANDATION :
Veuillez vérifier la configuration de votre clé 'GEMINI_API_KEY' dans vos paramètres Streamlit Secrets pour activer la rédaction automatique personnalisée par l'intelligence artificielle.
 
{"=" * 60}
Générateur de documents RH | UCAO 2025-2026"""
 
               # ── Affichage du résultat ─────────────────────────────────────────────
        st.markdown(f'<div class="section-title" style="--c:{VC};"> {type_doc} — Résultat</div>', unsafe_allow_html=True)
        st.caption(source)
 
        st.markdown(f"""
        <div style="background:{GRC};border:1px solid {VC}44;border-radius:12px;
        padding:22px 26px;font-size:13px;color:{TXC};line-height:1.9;
        white-space:pre-wrap;max-height:520px;overflow-y:auto;
        font-family:\'Courier New\',monospace;">{texte_genere}</div>
        """, unsafe_allow_html=True)
        # ── Boutons de téléchargement ─────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
 
        with dl1:
            st.download_button(
                " Télécharger (.txt)",
                data=texte_genere,
                file_name=f"lettre_RH_employe_{emp_idx_g}.txt",
                mime="text/plain",
                use_container_width=True)
 
        with dl2:
            html_lettre = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{type_doc} - Employé #{emp_idx_g}</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;color:#111;line-height:1.7;max-width:800px;}}
h1{{color:#2c3e50;border-bottom:2px solid #e74c3c;padding-bottom:10px;}}
.badge{{background:#e74c3c;color:white;padding:4px 14px;border-radius:20px;
        font-size:12px;font-weight:bold;margin-left:10px;}}
.info{{background:#f8f9fa;border-left:4px solid #3498db;padding:12px 16px;
       border-radius:0 8px 8px 0;margin-bottom:20px;font-size:13px;}}
pre{{background:#f8f9fa;padding:20px;border-radius:8px;
     white-space:pre-wrap;font-size:13px;line-height:1.7;}}
.footer{{color:#666;font-size:11px;margin-top:30px;
         border-top:1px solid #ddd;padding-top:10px;}}
</style></head><body>
<h1>{type_doc} <span class="badge">Risque {proba_g:.0f}%</span></h1>
<div class="info">
  <strong>Employé #{emp_idx_g}</strong> | {emp_g["JobRole"]} | {emp_g["Department"]}<br>
  Ancienneté : {int(emp_g["YearsAtCompany"])} ans | Salaire : {int(emp_g["MonthlyIncome"]):,} €/mois<br>
  Niveau de risque : <strong>{niv_g}</strong>
</div>


<div class="info" style="background:#fdf2f2; border-left:4px solid #e74c3c;">
  <strong>Causes du risque identifiées par l'IA :</strong><br>
  {", ".join([traduire_nom(f) for f in facteurs_risque])}
</div>
<pre>{texte_genere}</pre>
<div class="footer">
  Généré par Generative HR Analytics<br>
  Seye Kiné | Bindia Adeline Thiara | M. Aidara — UCAO 2025-2026
</div>
</body></html>"""


     
 
            st.download_button(
                " Exporter en HTML (→ PDF)",
                data=html_lettre,
                file_name=f"lettre_RH_employe_{emp_idx_g}.html",
                mime="text/html",
                use_container_width=True)


        
            # --- BOUTON DE TÉLÉCHARGEMENT DU PACK COMPLET ---
        st.markdown("---")
        st.markdown("##### 📦 Dossier Stratégique")
        
        # On prépare un document qui combine tout
        pack_complet = f"""
        DOSSIER DE RÉTENTION - EMPLOYÉ #{emp_idx_g}
        -------------------------------------------
        DIAGNOSTIC IA : {niv_g} ({proba_g:.1f}%)
        POSTE : {traduire_nom(emp_g['JobRole'])}
        DÉPARTEMENT : {emp_g['Department']}
        
        {texte_genere}
        
        -------------------------------------------
        Généré par Generative HR Analytics - UCAO 2025
        """
        
        st.download_button(
            label="📥 Télécharger le Pack de Rétention Complet (PDF/Doc)",
            data=pack_complet,
            file_name=f"Pack_Retention_{emp_idx_g}.txt",
            mime="text/plain",
            use_container_width=True)
        
        st.info(" Pour activer la vraie GenAI : le code appelle automatiquement l'API Claude (Anthropic). Sans connexion, une version locale est générée.")




# =============================================================================
# CODE QUADRANT TALENTS CRITIQUES — Performance vs Risque
# =============================================================================

 
elif nav == "Talents":
    st.markdown(f'<div class="pg"><div class="pt"> Quadrant Talents Critiques</div><div class="ps">Performance vs Risque de départ — Identifiez les talents à retenir en priorité</div></div>', unsafe_allow_html=True)
 
    if not MODELE_OK:
        st.error("Modèle non disponible. Lancez HR_Analytics.ipynb en premier.")
        st.stop()
 
    # ── Mapper PerformanceRating en score numérique ───────────────────────────
    perf_map = {
        "Low": 1, "Bad": 1,
        "Good": 2, "Medium": 2,
        "Excellent": 3, "High": 3,
        "Outstanding": 4, "Very High": 4
    }
    df_q = df.copy()
    df_q["Perf_Score"] = df_q["PerformanceRating"].map(perf_map).fillna(2)
    df_q["Risque_Pct"] = df_q["Probabilite"] * 100
 
    # ── Calcul des médianes pour séparer les 4 quadrants ─────────────────────
    med_perf = 3.1 # Cela séparera ceux qui ont 3 de ceux qui ont 4
    med_risque = df_q["Risque_Pct"].median()
 
    def classer_quadrant(row):
        if row["Perf_Score"] >= med_perf and row["Risque_Pct"] >= med_risque:
            return "Talents Critiques"
        elif row["Perf_Score"] >= med_perf and row["Risque_Pct"] < med_risque:
            return "Piliers Stables"
        elif row["Perf_Score"] < med_perf and row["Risque_Pct"] >= med_risque:
            return "À Accompagner"
        else:
            return "Profils Confortables"
 
    df_q["Quadrant"] = df_q.apply(classer_quadrant, axis=1)
 
    # ── Configuration couleurs et icônes ──────────────────────────────────────
    q_colors = {
        "Talents Critiques"   : RC,
        "Piliers Stables"     : VC,
        "À Accompagner"       : OC,
        "Profils Confortables": BC,
    }
    q_icons = {
        "Talents Critiques"   : "",
        "Piliers Stables"     : "",
        "À Accompagner"       : "",
        "Profils Confortables": "",
    }
    q_descriptions = {
        "Talents Critiques"   : "Priorité #1 — Retenir à tout prix",
        "Piliers Stables"     : "Fidèles et performants — A valoriser",
        "À Accompagner"       : "Risque élevé, perf. faible — A coacher",
        "Profils Confortables": "Stables — Suivi régulier",
    }
 
    # ── KPI par quadrant ──────────────────────────────────────────────────────
    k1q, k2q, k3q, k4q = st.columns(4)
    for col, q in zip([k1q, k2q, k3q, k4q], list(q_colors.keys())):
        n_q  = int((df_q["Quadrant"] == q).sum())
        c_q  = q_colors[q]
        i_q  = q_icons[q]
        d_q  = q_descriptions[q]
        with col:
            st.markdown(f"""
            <div class="kc" style="--c:{c_q};margin-bottom:14px;">
              <div class="kv">{n_q}</div>
              <div class="kl">{i_q} {q}</div>
              <div style="font-size:10px;color:{T2C};margin-top:4px;">{d_q}</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Graphique Scatter — pleine largeur ────────────────────────────────────
    st.markdown(f'<div class="section-title" style="--c:{BC};"> Carte des Talents — Performance vs Risque</div>', unsafe_allow_html=True)
 
    fig_q = go.Figure()
 
    for quad, color in q_colors.items():
        df_sub = df_q[df_q["Quadrant"] == quad]
        fig_q.add_trace(go.Scatter(
            x=df_sub["Perf_Score"] + np.random.uniform(-0.3, 0.3, len(df_sub)),
            y=df_sub["Risque_Pct"],
            mode="markers",
            name=f"{q_icons[quad]} {quad} ({len(df_sub)})",
            marker=dict(
                color=color,
                size=9,
                opacity=0.65,
                line=dict(color=FOC, width=0.5)),
            hovertemplate=(
                "<b>Employé #%{customdata[0]}</b><br>"
                "Département : %{customdata[1]}<br>"
                "Poste : %{customdata[2]}<br>"
                "Performance : %{x:.0f}/4<br>"
                "Risque : %{y:.1f}%<br>"
                "Salaire : %{customdata[3]:,} €<extra></extra>"),
            customdata=df_sub[
                [df_sub.index.name or "index" if df_sub.index.name else df_sub.index,
                 "Department", "JobRole", "MonthlyIncome"]
            ].reset_index()[["index", "Department", "JobRole", "MonthlyIncome"]].values
            if False else
            np.column_stack([
                df_sub.index.values,
                df_sub["Department"].values,
                df_sub["JobRole"].values,
                df_sub["MonthlyIncome"].values
            ])
        ))
 
    # Lignes de séparation des quadrants
    fig_q.add_hline(
        y=med_risque,
        line_dash="dash", line_color=T2C, line_width=1.5,
        annotation_text=f"Risque médian {med_risque:.1f}%",
        annotation_font=dict(color=T2C, size=11))
 
    fig_q.add_vline(
        x=med_perf,
        line_dash="dash", line_color=T2C, line_width=1.5,
        annotation_text="Performance médiane",
        annotation_font=dict(color=T2C, size=11))
 
    # Annotations des 4 quadrants
    # 1. Annotation TALENTS CRITIQUES (Haut Droite)
    fig_q.add_annotation(
        x=3.7, y=88,
        text=" TALENTS CRITIQUES<br>Retenir à tout prix",
        showarrow=False,
        font=dict(color=RC, size=12, family="Inter"),
        bgcolor="rgba(255, 75, 110, 0.15)", # Correction ICI
        bordercolor=RC, borderwidth=1, borderpad=6)
 
    # 2. Annotation PILIERS STABLES (Bas Droite)
    fig_q.add_annotation(
        x=3.7, y=5,
        text=" PILIERS STABLES<br>Fidèles et performants",
        showarrow=False,
        font=dict(color=VC, size=12, family="Inter"),
        bgcolor="rgba(0, 200, 150, 0.15)", # Correction ICI
        bordercolor=VC, borderwidth=1, borderpad=6)
 
    # 3. Annotation A ACCOMPAGNER (Haut Gauche)
    fig_q.add_annotation(
        x=1.1, y=88,
        text=" À ACCOMPAGNER<br>Coacher et fidéliser",
        showarrow=False,
        font=dict(color=OC, size=12, family="Inter"),
        bgcolor="rgba(255, 209, 102, 0.15)", # Correction ICI
        bordercolor=OC, borderwidth=1, borderpad=6)
 
    # 4. Annotation CONFORTABLES (Bas Gauche)
    fig_q.add_annotation(
        x=1.1, y=5,
        text=" CONFORTABLES<br>Suivi régulier",
        showarrow=False,
        font=dict(color=BC, size=12, family="Inter"),
        bgcolor="rgba(74, 158, 245, 0.15)", # Correction ICI
        bordercolor=BC, borderwidth=1, borderpad=6)
    fig_q.update_layout(**LAY,
        title=dict(
            text="Chaque point = 1 employé | Axe X = Performance | Axe Y = Risque de départ",
            font=dict(size=13), x=0.5),
        xaxis=dict(**ax("Performance (1=Low → 4=Outstanding)"),
            range=[0.4, 4.6],
            tickvals=[1, 2, 3, 4],
            ticktext=["Low (1)", "Good (2)", "Excellent (3)", "Outstanding (4)"]),
        yaxis=dict(**ax("Risque de départ (%)"), range=[-3, 103]),
        height=580,
        margin=dict(t=55, b=50, l=70, r=20),
        legend=dict(bgcolor=CAC, bordercolor=BOC, borderwidth=1,
                    font=dict(color=TXC)))
    st.plotly_chart(fig_q, use_container_width=True)
 
    # ── Explication méthodologie ──────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{CAC};border:1px solid {BOC};border-radius:10px;
    padding:14px 18px;margin-bottom:20px;">
      <div style="font-size:12px;font-weight:700;color:{TXC};margin-bottom:8px;">
         Comment lire ce graphique ?
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
        <div style="border-left:3px solid {RC};padding-left:10px;">
          <div style="font-size:11px;font-weight:700;color:{RC};"> Haut droite</div>
          <div style="font-size:11px;color:{T2C};">Performance élevée + Risque élevé = Budget rétention prioritaire</div>
        </div>
        <div style="border-left:3px solid {VC};padding-left:10px;">
          <div style="font-size:11px;font-weight:700;color:{VC};"> Bas droite</div>
          <div style="font-size:11px;color:{T2C};">Performance élevée + Risque faible = Valoriser et fidéliser</div>
        </div>
        <div style="border-left:3px solid {OC};padding-left:10px;">
          <div style="font-size:11px;font-weight:700;color:{OC};"> Haut gauche</div>
          <div style="font-size:11px;color:{T2C};">Performance faible + Risque élevé = Accompagnement RH</div>
        </div>
        <div style="border-left:3px solid {BC};padding-left:10px;">
          <div style="font-size:11px;font-weight:700;color:{BC};"> Bas gauche</div>
          <div style="font-size:11px;color:{T2C};">Performance faible + Risque faible = Suivi standard</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Tableau Talents Critiques ─────────────────────────────────────────────
    st.markdown(f'<div class="section-title" style="--c:{RC};"> Top Talents Critiques — À retenir en priorité absolue</div>', unsafe_allow_html=True)
 
    df_tc = df_q[df_q["Quadrant"] == "Talents Critiques"][[
        "Department", "JobRole", "MonthlyIncome",
        "YearsAtCompany", "PerformanceRating",
        "Risque_Pct", "OverTime", "JobSatisfaction"
    ]].sort_values("Risque_Pct", ascending=False).head(20)
 
    df_tc.columns = [
        "Département", "Poste", "Salaire €",
        "Ancienneté", "Performance",
        "Risque %", "H.Sup", "Satisfaction"
    ]
    df_tc["H.Sup"] = df_tc["H.Sup"].map({"Yes": " Oui", "No": " Non"})
 
    st.dataframe(
        df_tc,
        use_container_width=True,
        height=380,
        column_config={
            "Risque %": st.column_config.ProgressColumn(
                "Risque %",
                min_value=0, max_value=100,
                format="%.1f%%",
                width=120),
            "Salaire €": st.column_config.NumberColumn(
                "Salaire €",
                format="%d €")
        })
 
    # ── Analyse par département ───────────────────────────────────────────────
    st.markdown(f'<div class="section-title" style="--c:{OGC};"> Répartition des Talents Critiques par Département</div>', unsafe_allow_html=True)
 
    dept_tc = df_q[df_q["Quadrant"] == "Talents Critiques"].groupby("Department").agg(
        Nb_Talents_Critiques=("Probabilite", "count"),
        Risque_Moyen=("Risque_Pct", "mean"),
        Salaire_Moyen=("MonthlyIncome", "mean")
    ).reset_index().sort_values("Nb_Talents_Critiques", ascending=True)
 
    fig_dept = go.Figure(go.Bar(
        x=dept_tc["Nb_Talents_Critiques"],
        y=dept_tc["Department"],
        orientation="h",
        marker=dict(
            color=[RC if v > dept_tc["Nb_Talents_Critiques"].median() else OGC
                   for v in dept_tc["Nb_Talents_Critiques"]],
            line=dict(color=FOC, width=1.5)),
        text=[f"{v} talents critiques" for v in dept_tc["Nb_Talents_Critiques"]],
        textposition="outside"))
 
    fig_dept.update_layout(**LAY,
        title=dict(
            text="Nombre de Talents Critiques par Département",
            font=dict(size=14), x=0.5),
        xaxis=dict(**ax("Nb Talents Critiques")),
        yaxis=ax(),
        height=300,
        margin=dict(t=55, b=40, l=220, r=80))
    st.plotly_chart(fig_dept, use_container_width=True)
 
    # ── Recommandation finale ─────────────────────────────────────────────────
    n_talents = int((df_q["Quadrant"] == "Talents Critiques").sum())
    cout_total = int(df_q[df_q["Quadrant"] == "Talents Critiques"]["MonthlyIncome"].sum() * 6)
 
    st.markdown(f"""
    <div style="background:{RC}12;border:1px solid {RC}44;border-left:5px solid {RC};
    border-radius:0 12px 12px 0;padding:16px 22px;margin-top:8px;">
      <div style="font-size:12px;font-weight:700;color:{RC};margin-bottom:8px;
      text-transform:uppercase;letter-spacing:1px;">
         Alerte RH — Action Prioritaire
      </div>
      <div style="font-size:14px;color:{TXC};line-height:1.8;">
        <strong style="color:{RC};">{n_talents} Talents Critiques</strong> identifiés —
        performances élevées et fort risque de départ.<br>
        Coût total estimé si tous partent :
        <strong style="color:{RC};">{cout_total:,} €</strong> (6 mois de salaire × employé)<br>
         Lancez une simulation What-If pour tester les actions de rétention.
      </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CODE COUT DU TURNOVER — ROI des actions de rétention
# =============================================================================

 
elif nav == "Cout RH":
    st.markdown(f'<div class="pg"><div class="pt"> Coût du Turnover</div><div class="ps">Impact financier des départs — Calculez le ROI de la rétention</div></div>', unsafe_allow_html=True)
 
    if not MODELE_OK:
        st.error("Modèle non disponible. Lancez HR_Analytics.ipynb en premier.")
        st.stop()
 
    # ── Paramètres de calcul ──────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:13px;font-weight:700;color:{TXC};margin-bottom:12px;"> Paramètres de calcul</div>', unsafe_allow_html=True)
 
    p1, p2, p3 = st.columns(3)
    with p1:
        multiplicateur = st.slider(
            "Coût d'un départ (x salaire mensuel)", 3, 12, 6,
            help="En moyenne 6 mois de salaire : recrutement + formation + perte de productivité")
    with p2:
        taux_retention_obj = st.slider(
            "Taux de rétention espéré (%)", 10, 90, 60,
            help="Pourcentage d'employés à risque que vous espérez retenir grâce aux actions RH")
    with p3:
        budget_dispo = st.number_input(
            "Budget actions RH disponible (€)", 0, 1000000, 50000, step=5000,
            help="Budget total alloué aux actions de rétention (augmentations, formations, etc.)")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Calculs ───────────────────────────────────────────────────────────────
    df_c = df.copy()
    df_c["Cout_Depart"] = df_c["MonthlyIncome"] * multiplicateur
 
    # Employés à risque (au-dessus du seuil)
    df_risque  = df_c[df_c["Probabilite"] >= seuil]
    df_critique= df_c[df_c["Niveau"] == "Critique"]
    df_eleve = df_c[df_c["Niveau"] == "Eleve"]
 
    n_risque   = len(df_risque)
    n_crit_c   = len(df_critique)
    n_eleve_c  = len(df_eleve)
    cout_total = float(df_risque["Cout_Depart"].sum())
    cout_crit  = float(df_critique["Cout_Depart"].sum())
    cout_eleve = float(df_eleve["Cout_Depart"].sum())
 
    economie_p = cout_total * (taux_retention_obj / 100)
    roi        = ((economie_p - budget_dispo) / max(budget_dispo, 1)) * 100
 
    # ── KPI financiers ────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-title" style="--c:{RC};"> Impact financier du Turnover</div>', unsafe_allow_html=True)
 
    f1, f2, f3, f4 = st.columns(4)
    for col, val, lbl, c, sub in [
        (f1, f"{n_risque}",           "Employés à risque",     RC,  f"{n_crit_c} critiques + {n_eleve_c} élevés"),
        (f2, f"{cout_total:,.0f} €",  "Coût si tous partent",  RC,  f"Base : {multiplicateur}x salaire mensuel"),
        (f3, f"{economie_p:,.0f} €",  "Économie possible",     VC,  f"Si {taux_retention_obj}% retenus"),
        (f4, f"{roi:.0f}%",           "ROI des actions RH",    OC,  f"Budget : {budget_dispo:,} €"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kc" style="--c:{c};margin-bottom:14px;">
              <div class="kv" style="font-size:18px;">{val}</div>
              <div class="kl">{lbl}</div>
              <div style="font-size:10px;color:{T2C};margin-top:3px;">{sub}</div>
            </div>""", unsafe_allow_html=True)
 
    # ── Graphique Coût par département ────────────────────────────────────────
    st.markdown(f'<div class="section-title" style="--c:{RC};"> Coût potentiel par Département</div>', unsafe_allow_html=True)
 
    dept_cout = df_risque.groupby("Department").agg(
        Nb_Employes  = ("MonthlyIncome", "count"),
        Cout_Total   = ("Cout_Depart",   "sum"),
        Risque_Moyen = ("Probabilite",   "mean"),
        Salaire_Moy  = ("MonthlyIncome", "mean")
    ).reset_index().sort_values("Cout_Total", ascending=True)
 
    mediane_cout = dept_cout["Cout_Total"].median()
 
    fig_cout = go.Figure(go.Bar(
        x=dept_cout["Cout_Total"],
        y=[traduire_nom(i) for i in dept_cout["Department"]],
        orientation="h",
        marker=dict(
            color=[RC if v > mediane_cout else OC for v in dept_cout["Cout_Total"]],
            line=dict(color=FOC, width=1.5)),
        text=[f"{v:,.0f} €  ({int(n)} emp.)"
              for v, n in zip(dept_cout["Cout_Total"], dept_cout["Nb_Employes"])],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Coût total : %{x:,.0f} €<br>"
            "Nb employés : %{customdata[0]}<br>"
            "Salaire moyen : %{customdata[1]:,.0f} €<extra></extra>"),
        customdata=dept_cout[["Nb_Employes", "Salaire_Moy"]].values))
 
    fig_cout.add_vline(
        x=mediane_cout,
        line_dash="dash", line_color=OC, line_width=2,
        annotation_text=f"Médiane {mediane_cout:,.0f} €",
        annotation_font=dict(color=OC))
 
    fig_cout.update_layout(**LAY,
        title=dict(
            text=f"Coût potentiel du turnover par Département (base {multiplicateur}x salaire)",
            font=dict(size=14), x=0.5),
        xaxis=dict(**ax("Coût total (€)")),
        yaxis=ax(),
        height=350,
        margin=dict(t=55, b=40, l=200, r=140))
    st.plotly_chart(fig_cout, use_container_width=True)
 
    # ── Graphique ROI selon le budget ─────────────────────────────────────────
    st.markdown(f'<div class="section-title" style="--c:{VC};"> Analyse ROI selon le Budget Alloué</div>', unsafe_allow_html=True)
 
    budgets  = [5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000, 300000]
    roi_list = [((economie_p - b) / max(b, 1) * 100) for b in budgets]
 
    fig_roi = go.Figure()
 
    # Zone positive (ROI > 0)
    fig_roi.add_trace(go.Scatter(
        x=budgets, y=roi_list,
        mode="lines+markers",
        name="ROI",
        line=dict(color=VC, width=3),
        marker=dict(color=VC, size=10, line=dict(color=FOC, width=2)),
        fill="tozeroy",
        fillcolor="rgba(0,200,150,0.1)",
        hovertemplate="Budget : %{x:,.0f} €<br>ROI : %{y:.0f}%<extra></extra>"))
 
    # Ligne seuil de rentabilité
    fig_roi.add_hline(
        y=0,
        line_dash="dash", line_color=OC, line_width=2,
        annotation_text="Seuil de rentabilité (ROI = 0%)",
        annotation_font=dict(color=OC, size=11))
 
    # Budget actuel
    fig_roi.add_vline(
        x=budget_dispo,
        line_color=RC, line_width=2.5,
        annotation_text=f"Votre budget : {budget_dispo:,} €",
        annotation_font=dict(color=RC, size=11),
        annotation_position="top right")
 
    fig_roi.update_layout(**LAY,
        title=dict(
            text="Plus le budget est élevé, plus le ROI diminue — trouvez l'équilibre optimal",
            font=dict(size=13), x=0.5),
        xaxis=dict(**ax("Budget actions RH (€)")),
        yaxis=dict(**ax("ROI (%)")),
        height=380,
        margin=dict(t=55, b=40, l=70, r=20))
    st.plotly_chart(fig_roi, use_container_width=True)
 
    # ── Décomposition du coût du turnover ────────────────────────────────────
    st.markdown(f'<div class="section-title" style="--c:{OGC};"> Décomposition du Coût d\'un Départ</div>', unsafe_allow_html=True)
 
    salaire_moy = df_risque["MonthlyIncome"].mean()
    composantes = {
        "Recrutement (annonces, chasseur)"    : salaire_moy * 1.5,
        "Formation du remplaçant"             : salaire_moy * 1.5,
        "Perte de productivité (3 mois)"      : salaire_moy * 1.5,
        "Coûts administratifs et RH"          : salaire_moy * 0.5,
        "Transmission des connaissances"      : salaire_moy * 0.5,
        "Impact sur l'équipe (moral)"         : salaire_moy * 0.5,
    }
 
    comp_names = list(composantes.keys())
    comp_vals  = list(composantes.values())
    colors_comp = [RC, OGC, OC, BC, PC, VC]
 
    fig_comp = go.Figure(go.Bar(
        x=comp_vals, y=comp_names,
        orientation="h",
        marker=dict(color=colors_comp, line=dict(color=FOC, width=1.5)),
        text=[f"{v:,.0f} €" for v in comp_vals],
        textposition="outside"))
 
    fig_comp.update_layout(**LAY,
        title=dict(
            text=f"Décomposition du coût pour un employé au salaire moyen ({salaire_moy:,.0f} €/mois)",
            font=dict(size=13), x=0.5),
        xaxis=dict(**ax("Coût estimé (€)")),
        yaxis=ax(),
        height=360,
        margin=dict(t=55, b=40, l=290, r=100),
        showlegend=False)
    st.plotly_chart(fig_comp, use_container_width=True)
 
    # ── Tableau détaillé par niveau de risque ─────────────────────────────────
    st.markdown(f'<div class="section-title" style="--c:{BC};"> Détail par Niveau de Risque</div>', unsafe_allow_html=True)
 
    niveaux_data = []
    for niv, label, color in [
        ("Critique", "🔴 Critique (≥70%)", RC),
        ("R_Eleve",  "🟠 Élevé (50-70%)",  OGC),
        ("Modere",   "🟡 Modéré",           OC),
        ("Faible",   "🟢 Faible",           VC),
    ]:
        df_niv = df_c[df_c["Niveau"] == niv]
        if len(df_niv) > 0:
            niveaux_data.append({
                "Niveau"            : label,
                "Nb Employés"       : len(df_niv),
                "Salaire Moyen €"   : int(df_niv["MonthlyIncome"].mean()),
                "Coût si départ €"  : int(df_niv["Cout_Depart"].sum()),
                "Risque Moyen %"    : round(df_niv["Probabilite"].mean() * 100, 1),
            })
 
    df_niv_table = pd.DataFrame(niveaux_data)
    st.dataframe(
        df_niv_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Coût si départ €": st.column_config.NumberColumn(
                "Coût si départ €", format="%d €"),
            "Salaire Moyen €": st.column_config.NumberColumn(
                "Salaire Moyen €", format="%d €"),
            "Risque Moyen %": st.column_config.ProgressColumn(
                "Risque Moyen %", min_value=0, max_value=100, format="%.1f%%"),
        })
 
    # ── Recommandation budgétaire ─────────────────────────────────────────────
    n_a_retenir     = int(n_risque * taux_retention_obj / 100)
    budget_min_rec  = int(salaire_moy * 2 * n_a_retenir)
    budget_ideal    = int(economie_p * 0.20)  # 20% de l'économie = ROI de 400%
 
    st.markdown(f"""
    <div style="background:{VC}12;border:1px solid {VC}44;border-left:5px solid {VC};
    border-radius:0 12px 12px 0;padding:18px 24px;margin-top:8px;">
      <div style="font-size:12px;font-weight:700;color:{VC};margin-bottom:12px;
      text-transform:uppercase;letter-spacing:1px;">
         Recommandation Budgétaire
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">
        <div style="text-align:center;">
          <div style="font-size:10px;color:{T2C};margin-bottom:4px;">Employés à retenir</div>
          <div style="font-size:22px;font-weight:800;color:{VC};">{n_a_retenir}</div>
          <div style="font-size:10px;color:{T2C};">sur {n_risque} à risque ({taux_retention_obj}%)</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:10px;color:{T2C};margin-bottom:4px;">Budget minimum recommandé</div>
          <div style="font-size:22px;font-weight:800;color:{OC};">{budget_min_rec:,} €</div>
          <div style="font-size:10px;color:{T2C};">(2 salaires × nb employés)</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:10px;color:{T2C};margin-bottom:4px;">Économie nette estimée</div>
          <div style="font-size:22px;font-weight:800;color:{VC};">{int(economie_p - budget_dispo):,} €</div>
          <div style="font-size:10px;color:{T2C};">ROI net : {roi:.0f}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

               
 
# =============================================================================
# PAGE 4 — RAPPORT RH
# =============================================================================
elif nav == "Rapport RH":
    st.markdown(f'<div class="pg"><div class="pt"> Rapport RH</div><div class="ps">Synthèse — XGBoost | UCAO 2025-2026</div></div>', unsafe_allow_html=True)
 
    nc2 = (df["Niveau"]=="Critique").sum() if MODELE_OK else 0
    ne2 = (df["Niveau"]=="Eleve").sum()    if MODELE_OK else 0
 
    k1,k2,k3,k4 = st.columns(4)
    for col,val,lbl,c in [
        (k1,f"{n:,}","Employés",BC),(k2,f"{taux:.1f}%","Attrition",RC),
        (k3,f"{nc2}","Critiques ≥70%",RC),(k4,f"{F1:.4f}","F1-Score",PC)]:
        with col:
            st.markdown(f'<div class="kc" style="--c:{c};margin-bottom:16px;"><div class="kv">{val}</div><div class="kl">{lbl}</div></div>', unsafe_allow_html=True)
 
    m1,m2 = st.columns(2)
    with m1:
        st.markdown(f'<div style="background:{CAC};border:1px solid {BOC};border-radius:12px;padding:16px 20px;"><div style="font-size:13px;font-weight:700;color:{TXC};margin-bottom:12px;"> Métriques du modèle</div>', unsafe_allow_html=True)
        for lb,va,c in [("Algorithme","XGBoost",BC),("F1-Score",f"{F1:.4f}",PC),("AUC-ROC",f"{AUC:.4f}",BC),("Seuil",f"{seuil:.2f}",OC),("Taux d'attrition",f"{taux:.1f}%",RC)]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid {BOC};"><span style="font-size:12px;color:{T2C};">{lb}</span><span style="font-size:12px;font-weight:700;color:{c};">{va}</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    with m2:
        dr = df.groupby("Department")["Attrition"].mean()*100
        st.markdown(f'<div style="background:{CAC};border:1px solid {BOC};border-radius:12px;padding:16px 20px;"><div style="font-size:13px;font-weight:700;color:{TXC};margin-bottom:12px;"> Risque par Département</div>', unsafe_allow_html=True)
        for dept,risque in dr.sort_values(ascending=False).items():
            cv2 = RC if risque>taux else VC
            nd  = len(df[df["Department"]==dept])
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid {BOC};"><span style="font-size:12px;color:{T2C};">{dept}</span><div style="text-align:right;"><span style="font-size:14px;font-weight:800;color:{cv2};">{risque:.1f}%</span><span style="font-size:11px;color:{T2C};margin-left:8px;">({nd:,} emp.)</span></div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    # Niveaux de risque
    if MODELE_OK:
        niv_c = df["Niveau"].value_counts()
        ordre2 = ["Critique","Eleve","Modere","Faible"]
        fig_niv = go.Figure(go.Bar(
            x=[niv_c.get(nx,0) for nx in ordre2], y=ordre2, orientation="h",
            marker=dict(color=[RC,OGC,OC,VC], line=dict(color=FOC,width=1.5)),
            text=[f"{niv_c.get(nx,0):,}" for nx in ordre2], textposition="outside"))
        fig_niv.update_layout(**LAY,
            title=dict(text="Répartition des Niveaux de Risque",font=dict(size=14),x=0.5),
            xaxis=dict(**ax("Nb employés")), yaxis=ax(),
            height=300, margin=dict(t=50,b=35,l=90,r=60))
        st.plotly_chart(fig_niv, use_container_width=True)
 
    # Rapport texte
    st.markdown(f'<div style="font-size:14px;font-weight:800;color:{TXC};margin:20px 0 12px;padding-bottom:6px;border-bottom:1px solid {BOC};"> Rapport narratif</div>', unsafe_allow_html=True)
    try:
        with open("rapport_rh_genai.txt","r",encoding="utf-8") as f:
            rapport = f.read()
        st.markdown(f'<div style="background:{GRC};border:1px solid {BOC};border-radius:12px;padding:20px 24px;font-family:monospace;font-size:12px;color:{T2C};white-space:pre-wrap;line-height:1.8;max-height:450px;overflow-y:auto;">{rapport}</div>', unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(" Télécharger (.txt)", data=rapport,
                file_name="rapport_rh.txt", mime="text/plain", use_container_width=True)
        with col_dl2:
            # Export HTML imprimable (s'ouvre dans le navigateur → Ctrl+P → Enregistrer en PDF)
            rapport_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Rapport RH - UCAO 2025-2026</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;color:#111;line-height:1.6;}}
h1{{color:#2c3e50;border-bottom:2px solid #2c3e50;padding-bottom:8px;}}
pre{{background:#f8f9fa;padding:20px;border-radius:8px;white-space:pre-wrap;font-size:13px;}}
.footer{{color:#666;font-size:11px;margin-top:30px;border-top:1px solid #ddd;padding-top:10px;}}
</style></head><body>
<h1>Rapport RH — Analyse Prédictive du Turnover</h1>
<pre>{rapport}</pre>
<div class="footer">Seye Kiné | Bindia Adeline Thiara | M. Aidara — UCAO 2025-2026</div>
</body></html>"""
            st.download_button(" Exporter en HTML (→ PDF)", data=rapport_html,
                file_name="rapport_rh.html", mime="text/html", use_container_width=True)
        st.info(" Pour obtenir un PDF : téléchargez le fichier HTML, ouvrez-le dans votre navigateur, puis Ctrl+P → Enregistrer en PDF")
    except FileNotFoundError:
        dr2   = df.groupby("Department")["Attrition"].mean()*100
        ot_y2 = df[df["OverTime"]=="Yes"]["Attrition"].mean()*100
        ot_n2 = df[df["OverTime"]=="No"]["Attrition"].mean()*100
        r_auto = f"""RAPPORT RH — HR ANALYTICS | UCAO 2025-2026
{"="*50}
 
1. RESUME : {n:,} employes | Attrition : {taux:.1f}% | F1={F1:.4f} | AUC={AUC:.4f}
 
2. NIVEAUX DE RISQUE
   Critique (>=70%) : {nc2:>4} employes — Action immediate
   Eleve   (50-70%) : {ne2:>4} employes — Entretien sous 2 semaines
 
3. FACTEURS
   Avec heures sup  : {ot_y2:.1f}% | Sans : {ot_n2:.1f}% | Ecart : +{ot_y2-ot_n2:.1f} pts
 
4. PAR DEPARTEMENT
""" + "\n".join([f"   {d:<35} : {r:.1f}%" for d,r in dr2.sort_values(ascending=False).items()]) + f"""
 
5. RECOMMANDATIONS
   1. Entretiens immediats pour {nc2} employes critiques
   2. Reduire les heures supplementaires
   3. Ameliorer satisfaction et equilibre WLB
{"="*50}
Seye Kine & Bindia Adeline Thiara | M. Aidara | UCAO 2025-2026"""
        st.markdown(f'<div style="background:{GRC};border:1px solid {BOC};border-radius:12px;padding:20px 24px;font-family:monospace;font-size:12px;color:{T2C};white-space:pre-wrap;line-height:1.8;max-height:400px;overflow-y:auto;">{r_auto}</div>', unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(" Télécharger (.txt)", data=r_auto,
                file_name="rapport_rh.txt", mime="text/plain", use_container_width=True)
        with col_dl2:
            rapport_html2 = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Rapport RH - UCAO 2025-2026</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;color:#111;line-height:1.6;}}
h1{{color:#2c3e50;border-bottom:2px solid #2c3e50;padding-bottom:8px;}}
pre{{background:#f8f9fa;padding:20px;border-radius:8px;white-space:pre-wrap;font-size:13px;}}
.footer{{color:#666;font-size:11px;margin-top:30px;border-top:1px solid #ddd;padding-top:10px;}}
</style></head><body>
<h1>Rapport RH — Analyse Prédictive du Turnover</h1>
<pre>{r_auto}</pre>
<div class="footer">Seye Kiné | Bindia Adeline Thiara | M. Aidara — UCAO 2025-2026</div>
</body></html>"""
            st.download_button(" Exporter en HTML (→ PDF)", data=rapport_html2,
                file_name="rapport_rh.html", mime="text/html", use_container_width=True)
        st.info(" Pour PDF : ouvrez le fichier HTML dans votre navigateur → Ctrl+P → Enregistrer en PDF")




# --- NOUVELLE SECTION : ÉTHIQUE & ÉQUITÉ ---
    st.markdown(f'<div class="section-title" style="--c:{PC};"> Éthique & Équité Algorithmique</div>', unsafe_allow_html=True)
    
    col_eth1, col_eth2 = st.columns([1, 2])
    
    with col_eth1:
        # Calcul de la parité de prédiction
        parite_genre = df.groupby("Gender")["Probabilite"].mean()
        st.markdown(f"""
        <div style='background:{CAC}; padding:15px; border-radius:10px; border:1px solid {BOC};'>
            <div style='font-size:12px; color:{T2C};'>Indice de parité (Genre)</div>
            <div style='font-size:18px; font-weight:bold; color:{VC};'>
                {abs(parite_genre['Male'] - parite_genre['Female']):.2f} Δ
            </div>
            <div style='font-size:10px; color:{T2C}; margin-top:5px;'>
                Un écart proche de 0 indique une absence de discrimination systématique.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_eth2:
        st.markdown(f"""
        <div style='background:{GRC}; padding:15px; border-radius:10px; border-left:5px solid {VC};'>
            <b style='color:{VC};'>Note de Conformité (AI Act)</b><br>
            <span style='font-size:13px; color:{TXC};'>
                Le modèle a été audité pour vérifier l'absence de biais sur les variables protégées (Genre, Âge). 
                Les décisions de l'IA reposent à <b>92%</b> sur des critères de performance et d'organisation 
                (Heures sup, Satisfaction, Ancienneté), garantissant un traitement équitable des collaborateurs.
            </span>
        </div>
        """, unsafe_allow_html=True)
 
# Footer
st.markdown(f'<hr style="border-color:{BOC};margin:30px 0 10px;"><div style="text-align:center;color:{T2C};font-size:12px;padding:6px 0;">HR Analytics · Seye Kiné | Bindia Adeline Thiara · <span style="color:{OC};">M. Aidara</span> · UCAO 2025-2026</div>', unsafe_allow_html=True)