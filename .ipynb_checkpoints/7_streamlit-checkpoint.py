# =============================================================================
#  HR Analytics — Dashboard Streamlit
#  Auteurs   : Seye Kiné | Bindia Adeline Thiara
#  Encadrant : M. Aidara — UCAO 2025-2026
#  Lancement : streamlit run 7_streamlit.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import json
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Couleurs ───────────────────────────────────────────────────────────────────
VC="#00C896"; RC="#FF4B6E"; BC="#4A9EF5"; OC="#FFD166"
PC="#9B72F5"; OGC="#FF8C42"; FOC="#0F1923"; CAC="#1A2535"
GRC="#243044"; TXC="#E8F0FE"; T2C="#8FA3BF"; GIC="#2A3A50"; BOC="#3A4F6A"

# ── Chargement ─────────────────────────────────────────────────────────────────
@st.cache_resource
def charger():
    with open("mon_modele_rh.pkl", "rb") as f:
        return pickle.load(f)

try:
    data = charger()
except FileNotFoundError:
    st.error("mon_modele_rh.pkl introuvable. Lancez d'abord 1_HR_Analytics.ipynb")
    st.stop()

modele        = data["cerveau_ia"]
preprocesseur = data["traitement"]
seuil         = data["reglage_seuil"]
FEATURES      = data["features"]
F1            = data["f1"]
AUC           = data["auc"]

@st.cache_data
def charger_df():
    df = pd.read_csv("hr.csv")
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    X = preprocesseur.transform(df[FEATURES])
    df["Probabilite"] = modele.predict_proba(X)[:, 1]
    df["Prediction"]  = (df["Probabilite"] >= seuil).astype(int)
    df["Risque_Pct"]  = (df["Probabilite"] * 100).round(1)
    def niv(p):
        if p >= 0.70: return "Critique"
        elif p >= 0.50: return "Eleve"
        elif p >= seuil: return "Modere"
        else: return "Faible"
    df["Niveau"] = df["Probabilite"].apply(niv)
    return df

df = charger_df()
n  = len(df)
taux = df["Attrition"].mean() * 100

LAY = dict(
    paper_bgcolor=CAC, plot_bgcolor=GRC,
    font=dict(color=TXC, family="Segoe UI, sans-serif"),
)
def ax(t=""):
    return dict(title=t, gridcolor=GIC, showgrid=True,
                zeroline=False, tickfont=dict(color=T2C))

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
* {{ font-family: 'Segoe UI', sans-serif !important; }}
.stApp {{ background: {FOC}; }}
section[data-testid="stSidebar"] {{ background: {CAC} !important; }}
.stTabs [data-baseweb="tab-list"] {{ background: {CAC}; border-radius:10px; padding:4px; }}
.stTabs [data-baseweb="tab"] {{ color: {T2C} !important; border-radius:8px !important; }}
.stTabs [aria-selected="true"] {{ background: {GRC} !important; color: {TXC} !important; }}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {BC}, {PC}) !important;
    border: none !important; color: white !important;
    font-weight: 700 !important; border-radius: 10px !important;
}}
.stSelectbox > div > div {{ background: {GRC} !important; border: 1px solid {BOC} !important; color: {TXC} !important; border-radius: 8px !important; }}
h1,h2,h3,h4 {{ color: {TXC} !important; }}
footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem !important; }}
.kpi {{ background: {CAC}; border: 1px solid {BOC}; border-top: 3px solid var(--c);
        border-radius: 14px; padding: 16px; text-align: center; }}
.kv  {{ font-size: 26px; font-weight: 900; color: var(--c); }}
.kl  {{ font-size: 11px; color: {T2C}; text-transform: uppercase; letter-spacing: .8px; margin-top: 4px; font-weight: 600; }}
.pg  {{ background: linear-gradient(135deg, {CAC}, {GRC}); border: 1px solid {BOC};
        border-radius: 16px; padding: 20px 24px; margin-bottom: 16px; }}
.pt  {{ font-size: 22px; font-weight: 900; color: {TXC}; margin-bottom: 4px; }}
.ps  {{ font-size: 13px; color: {T2C}; }}
.dr  {{ display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid {BOC}; }}
.dl  {{ font-size: 12px; color: {T2C}; }}
.dv  {{ font-size: 12px; font-weight: 700; color: {TXC}; }}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 12px;">
      <div style="font-size:40px;">👥</div>
      <div style="font-size:14px;font-weight:800;color:{TXC};margin-top:6px;">HR Analytics</div>
      <div style="font-size:11px;color:{T2C};margin-top:4px;line-height:1.8;">
        Seye Kiné | Bindia Adeline Thiara<br>
        <span style="color:{OC};font-weight:600;">M. Aidara</span> — UCAO 2025-2026
      </div>
    </div>
    <hr style="border-color:{BOC};margin:0 0 12px;">
    """, unsafe_allow_html=True)

    nav = st.radio("", [
        "Accueil",
        "Exploration",
        "Prediction",
        "Rapport IA",
    ], label_visibility="collapsed")

    st.markdown(f"""
    <hr style="border-color:{BOC};margin:12px 0;">
    <div style="background:{GRC};border-radius:10px;padding:12px 14px;">
      <div style="font-size:10px;color:{T2C};text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;font-weight:700;">Modele actif</div>
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
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE 1 — ACCUEIL
# =============================================================================
if nav == "Accueil":
    st.markdown(f'<div class="pg"><div class="pt">HR Analytics Dashboard</div><div class="ps">IBM HR Dataset — {n:,} employes — UCAO 2025-2026</div></div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,val,lbl,c in [
        (c1, f"{n:,}",            "Employes",    BC),
        (c2, f"{taux:.1f}%",      "Attrition",   RC),
        (c3, f"{F1:.4f}",         "F1-Score",    PC),
        (c4, f"{AUC:.4f}",        "AUC-ROC",     BC),
    ]:
        with col:
            st.markdown(f'<div class="kpi" style="--c:{c};margin-bottom:14px;"><div class="kv">{val}</div><div class="kl">{lbl}</div></div>', unsafe_allow_html=True)

    g1,g2 = st.columns(2)
    with g1:
        n0=int((df["Attrition"]==0).sum()); n1=int((df["Attrition"]==1).sum())
        fig=go.Figure(go.Pie(values=[n0,n1],labels=["Restes","Partis"],hole=0.62,
            marker=dict(colors=[VC,RC],line=dict(color=FOC,width=3)),
            textinfo="label+percent",textfont=dict(size=13,color="white"),pull=[0,0.06]))
        fig.add_annotation(text=f"<b>{taux:.1f}%</b>",x=0.5,y=0.5,showarrow=False,font=dict(size=22,color=RC))
        fig.update_layout(**LAY,title=dict(text="Distribution Attrition",font=dict(size=13),x=0.5),height=300,margin=dict(t=45,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        ot=df.groupby("OverTime")["Attrition"].mean()*100
        dept=df.groupby("Department")["Attrition"].mean()*100
        fig2=make_subplots(rows=1,cols=2,subplot_titles=["OverTime","Par Departement"],horizontal_spacing=0.14)
        fig2.add_trace(go.Bar(x=["Non","Oui"],y=ot.values,marker=dict(color=[VC,RC],line=dict(color=FOC,width=2)),text=[f"{v:.1f}%" for v in ot.values],textposition="outside",showlegend=False),row=1,col=1)
        dept_s=dept.sort_values()
        coul_d=[RC if v>taux else BC for v in dept_s.values]
        fig2.add_trace(go.Bar(x=dept_s.values,y=dept_s.index,orientation="h",marker=dict(color=coul_d,line=dict(color=FOC,width=1.5)),showlegend=False),row=1,col=2)
        fig2.update_layout(**LAY,title=dict(text="Facteurs principaux",font=dict(size=13),x=0.5),height=300,margin=dict(t=45,b=30,l=10,r=20))
        for i in [1,2]:
            fig2.update_yaxes(gridcolor=GIC,row=1,col=i)
            fig2.update_xaxes(gridcolor=GIC,row=1,col=i)
        st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# PAGE 2 — EXPLORATION
# =============================================================================
elif nav == "Exploration":
    st.markdown(f'<div class="pg"><div class="pt">Exploration des Donnees</div><div class="ps">{n:,} employes × 31 variables</div></div>', unsafe_allow_html=True)

    tab1,tab2 = st.tabs(["Par variable","Correlations"])

    with tab1:
        var = st.selectbox("Variable :", ["OverTime","Department","JobSatisfaction","WorkLifeBalance","MaritalStatus","Education"])
        vals = df.groupby(var)["Attrition"].mean()*100
        vals = vals.sort_values(ascending=True)
        coul = [RC if v>taux else BC for v in vals.values]
        fig = go.Figure(go.Bar(x=vals.values,y=vals.index,orientation="h",
            marker=dict(color=coul,line=dict(color=FOC,width=1.5)),
            text=[f"{v:.1f}%" for v in vals.values],textposition="outside"))
        fig.add_vline(x=taux,line_dash="dash",line_color=OC,
            annotation_text=f"Moy. {taux:.1f}%",annotation_font=dict(color=OC))
        fig.update_layout(**LAY,title=dict(text=f"Attrition par {var}",font=dict(size=13),x=0.5),
            xaxis=dict(**ax("Taux (%)"),range=[0,55]),yaxis=ax(),
            height=420,showlegend=False,margin=dict(t=45,b=40,l=200,r=70))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        COLS_NUM = ["Age","DailyRate","DistanceFromHome","HourlyRate","MonthlyIncome",
                    "MonthlyRate","NumCompaniesWorked","PercentSalaryHike","StockOptionLevel",
                    "TotalWorkingYears","TrainingTimesLastYear","YearsAtCompany",
                    "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
        corr = df[COLS_NUM+["Attrition"]].corr()["Attrition"].drop("Attrition")
        corr_abs = corr.abs().sort_values(ascending=True)
        coul_c = [RC if corr[v]<0 else BC for v in corr_abs.index]
        fig_c = go.Figure(go.Bar(x=corr_abs.values,y=corr_abs.index,orientation="h",
            marker=dict(color=coul_c,line=dict(color=FOC,width=1.5)),
            text=[f"{corr[v]:+.4f}" for v in corr_abs.index],textposition="outside"))
        fig_c.update_layout(**LAY,title=dict(text="Correlations avec l'Attrition",font=dict(size=13),x=0.5),
            xaxis=dict(**ax("Correlation absolue")),yaxis=ax(),
            height=500,showlegend=False,margin=dict(t=45,b=40,l=210,r=80))
        st.plotly_chart(fig_c, use_container_width=True)

# =============================================================================
# PAGE 3 — PREDICTION
# =============================================================================
elif nav == "Prediction":
    st.markdown(f'<div class="pg"><div class="pt">Prediction & Explicabilite</div><div class="ps">Recherchez un employe ou faites une prediction manuelle</div></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Rechercher un employe", "Prediction manuelle"])

    with tab1:
        fc1,fc2,fc3,fc4 = st.columns(4)
        with fc1: f_dept = st.selectbox("Departement", ["Tous"]+sorted(df["Department"].unique()))
        with fc2: f_ot   = st.selectbox("Heures sup.", ["Tous","Oui (Yes)","Non (No)"])
        with fc3: f_risq = st.selectbox("Niveau risque", ["Tous","Critique","Eleve","Modere","Faible"])
        with fc4: f_gene = st.selectbox("Genre", ["Tous","Male","Female"])

        sc1,sc2 = st.columns([4,1])
        with sc1: recherche = st.text_input("", placeholder="Tapez un indice (ex: 5), departement ou poste...", label_visibility="collapsed")
        with sc2: chercher  = st.button("Rechercher", use_container_width=True, type="primary")

        filtres_actifs = (f_dept!="Tous" or f_ot!="Tous" or f_risq!="Tous" or f_gene!="Tous" or recherche.strip()!="")

        if not chercher and not filtres_actifs:
            n_crit = (df["Niveau"]=="Critique").sum()
            n_elev = (df["Niveau"]=="Eleve").sum()
            n_mod  = (df["Niveau"]=="Modere").sum()
            n_fai  = (df["Niveau"]=="Faible").sum()
            st.markdown(f"""
            <div style="background:{GRC};border:2px dashed {BOC};border-radius:14px;padding:36px 24px;text-align:center;margin:8px 0 14px;">
              <div style="font-size:40px;margin-bottom:10px;">🔎</div>
              <div style="font-size:15px;font-weight:700;color:{TXC};">Utilisez les filtres ou la barre de recherche</div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
              <div style="background:{RC}15;border:1px solid {RC}44;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:22px;font-weight:800;color:{RC};">{n_crit}</div>
                <div style="font-size:11px;color:{T2C};">Critique</div>
              </div>
              <div style="background:{OGC}15;border:1px solid {OGC}44;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:22px;font-weight:800;color:{OGC};">{n_elev}</div>
                <div style="font-size:11px;color:{T2C};">Eleve</div>
              </div>
              <div style="background:{OC}15;border:1px solid {OC}44;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:22px;font-weight:800;color:{OC};">{n_mod}</div>
                <div style="font-size:11px;color:{T2C};">Modere</div>
              </div>
              <div style="background:{VC}15;border:1px solid {VC}44;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:22px;font-weight:800;color:{VC};">{n_fai}</div>
                <div style="font-size:11px;color:{T2C};">Faible</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            df_f = df.copy()
            if f_dept != "Tous":  df_f = df_f[df_f["Department"]==f_dept]
            if f_gene != "Tous":  df_f = df_f[df_f["Gender"]==f_gene]
            if "Oui" in f_ot:     df_f = df_f[df_f["OverTime"]=="Yes"]
            elif "Non" in f_ot:   df_f = df_f[df_f["OverTime"]=="No"]
            if f_risq != "Tous":  df_f = df_f[df_f["Niveau"]==f_risq]
            if recherche.strip():
                rch  = recherche.strip().lower()
                mask = (df_f.index.astype(str).str.contains(rch) |
                        df_f["Department"].str.lower().str.contains(rch) |
                        df_f["JobRole"].str.lower().str.contains(rch))
                df_f = df_f[mask]
            df_f = df_f.sort_values("Probabilite", ascending=False)

            c_tab, c_det = st.columns([1.4,1], gap="large")

            with c_tab:
                st.markdown(f'<div style="font-size:12px;font-weight:700;color:{TXC};margin-bottom:6px;">{len(df_f):,} employes — cliquez sur une ligne</div>', unsafe_allow_html=True)
                df_tab = df_f[["Age","Gender","Department","JobRole","MonthlyIncome","OverTime","JobSatisfaction","Risque_Pct","Attrition"]].head(300).copy()
                df_tab.columns = ["Age","Genre","Dept","Poste","Salaire","H.Sup","JS","Risque %","Reel"]
                df_tab["Reel"] = df_tab["Reel"].map({0:"Reste",1:"Parti"})
                df_tab["H.Sup"] = df_tab["H.Sup"].map({"Yes":"Oui","No":"Non"})
                event = st.dataframe(df_tab, use_container_width=True, hide_index=False, height=400,
                    on_select="rerun", selection_mode="single-row",
                    column_config={
                        "Risque %" : st.column_config.ProgressColumn("Risque %", min_value=0, max_value=100, format="%.1f%%", width=110),
                    })
                if event.selection and event.selection["rows"]:
                    st.session_state["emp_idx"] = df_f.index[event.selection["rows"][0]]

            with c_det:
                if "emp_idx" not in st.session_state:
                    st.markdown(f'<div style="background:{CAC};border:2px dashed {BOC};border-radius:14px;padding:48px 24px;text-align:center;"><div style="font-size:40px;">👆</div><div style="font-size:14px;font-weight:700;color:{TXC};margin-top:10px;">Cliquez sur une ligne</div></div>', unsafe_allow_html=True)
                else:
                    idx = st.session_state["emp_idx"]
                    emp = df.loc[idx]
                    proba = float(emp["Probabilite"])
                    pct   = proba * 100

                    if pct>=70:   c_r=RC; niv="Critique"; rec="Intervention immediate. Entretien sous 48h."
                    elif pct>=50: c_r=OGC; niv="Eleve"; rec="Entretien sous 2 semaines."
                    elif pct>=seuil*100: c_r=OC; niv="Modere"; rec="Suivi mensuel RH."
                    else:         c_r=VC; niv="Faible"; rec="Profil stable."

                    reel = int(emp["Attrition"])
                    cr   = RC if reel==1 else VC
                    tr   = "A quitte" if reel==1 else "Est reste"

                    # Jauge
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=pct,
                        number={"suffix":"%","font":{"size":30,"color":TXC}},
                        title={"text":f"{emp['Gender']} — {emp['Age']} ans","font":{"size":12,"color":T2C}},
                        gauge={"axis":{"range":[0,100],"tickcolor":T2C,"ticksuffix":"%"},
                               "bar":{"color":c_r,"thickness":0.28},
                               "bgcolor":GRC,"bordercolor":BOC,
                               "steps":[{"range":[0,seuil*100],"color":"rgba(0,200,150,0.12)"},
                                         {"range":[seuil*100,70],"color":"rgba(255,209,102,0.12)"},
                                         {"range":[70,100],"color":"rgba(255,75,110,0.15)"}],
                               "threshold":{"line":{"color":OC,"width":3},"thickness":0.78,"value":seuil*100}},
                    ))
                    fig_g.update_layout(paper_bgcolor=CAC,plot_bgcolor=CAC,font=dict(color=TXC),height=220,margin=dict(t=45,b=5,l=20,r=20))
                    st.plotly_chart(fig_g, use_container_width=True)

                    st.markdown(f"""
                    <div style="text-align:center;font-size:14px;font-weight:700;color:{c_r};margin-bottom:6px;">{niv}</div>
                    <div style="background:{GRC};border-left:4px solid {BC};border-radius:0 8px 8px 0;padding:8px 12px;margin-bottom:8px;font-size:12px;color:{T2C};">💡 {rec}</div>
                    <div style="background:{cr}22;border:1px solid {cr};border-radius:8px;padding:7px 12px;font-size:12px;font-weight:600;color:{cr};margin-bottom:10px;">Realite : {tr}</div>
                    """, unsafe_allow_html=True)

                    for lbl,val,alert in [
                        ("Departement",  emp["Department"],      False),
                        ("Poste",        emp["JobRole"],         False),
                        ("Salaire",      f"{emp['MonthlyIncome']:,} EUR", False),
                        ("Heures sup.",  emp["OverTime"],        emp["OverTime"]=="Yes"),
                        ("Satisfaction", emp["JobSatisfaction"], emp["JobSatisfaction"] in ["Low","Medium"]),
                        ("WLB",          emp["WorkLifeBalance"],emp["WorkLifeBalance"] in ["Bad","Good"]),
                        ("Ans sans promo", f"{emp['YearsSinceLastPromotion']} ans", emp["YearsSinceLastPromotion"]>=3),
                    ]:
                        cv = RC if alert else TXC
                        ic = " ⚠️" if alert else ""
                        st.markdown(f'<div class="dr"><span class="dl">{lbl}</span><span style="font-size:12px;font-weight:700;color:{cv};">{val}{ic}</span></div>', unsafe_allow_html=True)

    with tab2:
        st.markdown(f'<div style="font-size:13px;font-weight:700;color:{TXC};margin-bottom:12px;">Renseignez les informations d\'un employe</div>', unsafe_allow_html=True)
        r1,r2,r3 = st.columns(3, gap="large")
        with r1:
            m_age   = st.number_input("Age", 18, 60, 35)
            m_inc   = st.number_input("Salaire mensuel", 1000, 20000, 5000, step=500)
            m_ot    = st.radio("Heures sup.", ["Non (No)","Oui (Yes)"], horizontal=True)
            m_js    = st.selectbox("Satisfaction travail", ["Low","Medium","High","Very High"], index=2)
            m_wlb   = st.selectbox("Equilibre WLB", ["Bad","Good","Better","Best"], index=2)
        with r2:
            m_dept  = st.selectbox("Departement", ["Human Resources","Research & Development","Sales"])
            m_role  = st.selectbox("Poste", ["Healthcare Representative","Human Resources","Laboratory Technician","Manager","Manufacturing Director","Research Director","Research Scientist","Sales Executive","Sales Representative"])
            m_genre = st.radio("Genre", ["Male","Female"], horizontal=True)
            m_marit = st.selectbox("Statut matrimonial", ["Divorced","Married","Single"])
            m_edu   = st.selectbox("Education", ["Below College","College","Bachelor","Master","Doctor"])
        with r3:
            m_dist  = st.slider("Distance domicile (km)", 1, 30, 10)
            m_anc   = st.number_input("Anciennete (ans)", 0, 30, 5)
            m_promo = st.number_input("Ans sans promotion", 0, 15, 2)
            m_jlev  = st.selectbox("Niveau poste", ["Entry Level","Junior Level","Mid Level","Senior Level","Executive Level"])
            m_travel= st.selectbox("Voyages", ["Non-Travel","Travel_Rarely","Travel_Frequently"])

        st.markdown("<br>", unsafe_allow_html=True)
        pred_btn = st.button("Analyser le risque", use_container_width=True, type="primary")

        if pred_btn:
            ot_val = "Yes" if "Oui" in m_ot else "No"
            df_in  = pd.DataFrame([{
                "Age":m_age,"DailyRate":800,"DistanceFromHome":m_dist,
                "HourlyRate":60,"MonthlyIncome":m_inc,"MonthlyRate":15000,
                "NumCompaniesWorked":2,"PercentSalaryHike":12,"StockOptionLevel":0,
                "TotalWorkingYears":m_anc+3,"TrainingTimesLastYear":2,
                "YearsAtCompany":m_anc,"YearsInCurrentRole":max(0,m_anc-2),
                "YearsSinceLastPromotion":m_promo,"YearsWithCurrManager":2,
                "BusinessTravel":m_travel,"Department":m_dept,"Education":m_edu,
                "EducationField":"Marketing","EnvironmentSatisfaction":"High",
                "Gender":m_genre,"JobInvolvement":"High","JobLevel":m_jlev,
                "JobRole":m_role,"JobSatisfaction":m_js,"MaritalStatus":m_marit,
                "OverTime":ot_val,"PerformanceRating":"Excellent",
                "RelationshipSatisfaction":"High","WorkLifeBalance":m_wlb,
            }])
            X_m    = preprocesseur.transform(df_in[FEATURES])
            proba_m= float(modele.predict_proba(X_m)[0][1])
            pct_m  = proba_m * 100

            if pct_m>=70:   c_m=RC; niv_m="Critique"; rec_m="Intervention immediate."
            elif pct_m>=50: c_m=OGC; niv_m="Eleve"; rec_m="Entretien sous 2 semaines."
            elif pct_m>=seuil*100: c_m=OC; niv_m="Modere"; rec_m="Suivi mensuel."
            else:           c_m=VC; niv_m="Faible"; rec_m="Profil stable."

            r_a, r_b = st.columns(2)
            with r_a:
                fig_gm = go.Figure(go.Indicator(
                    mode="gauge+number",value=pct_m,
                    number={"suffix":"%","font":{"size":30,"color":TXC}},
                    title={"text":"Probabilite de depart","font":{"size":12,"color":T2C}},
                    gauge={"axis":{"range":[0,100],"tickcolor":T2C,"ticksuffix":"%"},
                           "bar":{"color":c_m,"thickness":0.28},"bgcolor":GRC,"bordercolor":BOC,
                           "steps":[{"range":[0,seuil*100],"color":"rgba(0,200,150,0.12)"},
                                     {"range":[seuil*100,70],"color":"rgba(255,209,102,0.12)"},
                                     {"range":[70,100],"color":"rgba(255,75,110,0.15)"}],
                           "threshold":{"line":{"color":OC,"width":3},"thickness":0.78,"value":seuil*100}},
                ))
                fig_gm.update_layout(paper_bgcolor=CAC,plot_bgcolor=CAC,font=dict(color=TXC),height=250,margin=dict(t=50,b=10,l=20,r=20))
                st.plotly_chart(fig_gm, use_container_width=True)
                st.markdown(f'<div style="text-align:center;font-size:15px;font-weight:700;color:{c_m};margin-bottom:6px;">{niv_m}</div><div style="background:{GRC};border-left:4px solid {BC};padding:8px 12px;border-radius:0 8px 8px 0;font-size:12px;color:{T2C};">💡 {rec_m}</div>', unsafe_allow_html=True)

            with r_b:
                # Distribution
                tp = df["Probabilite"].values
                pct_ile = float((tp<proba_m).mean()*100)
                hv,hb   = np.histogram(tp,bins=30)
                bca     = (hb[:-1]+hb[1:])/2
                bwa     = hb[1]-hb[0]
                bcols   = [RC if b>=0.70 else OGC if b>=0.50 else OC if b>=seuil else VC for b in bca]
                fig_d   = go.Figure()
                fig_d.add_trace(go.Bar(x=bca,y=hv,width=[bwa*0.85]*len(bca),
                    marker=dict(color=bcols,opacity=0.75,line=dict(color=FOC,width=0.5)),showlegend=False))
                fig_d.add_vline(x=proba_m,line_color=c_m,line_width=2.5,
                    annotation_text=f"{pct_m:.1f}%",annotation_font=dict(color=c_m,size=11))
                fig_d.add_vline(x=seuil,line_dash="dash",line_color=T2C,line_width=1.5)
                fig_d.update_layout(**LAY,title=dict(text=f"Position parmi {n:,} employes — {pct_ile:.0f}e percentile",font=dict(size=11),x=0.5),
                    xaxis=dict(**ax("Probabilite"),range=[0.0,0.85]),yaxis=dict(**ax("Nb employes")),
                    height=250,margin=dict(t=40,b=35,l=50,r=10),showlegend=False)
                st.plotly_chart(fig_d, use_container_width=True)

# =============================================================================
# PAGE 4 — RAPPORT IA
# =============================================================================
elif nav == "Rapport IA":
    st.markdown(f'<div class="pg"><div class="pt">Rapport RH Automatique</div><div class="ps">Genere par le module IA Generative</div></div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    n_crit = (df["Niveau"]=="Critique").sum()
    n_elev = (df["Niveau"]=="Eleve").sum()
    for col,val,lbl,c in [
        (c1, f"{n:,}",      "Employes",   BC),
        (c2, f"{taux:.1f}%","Attrition",  RC),
        (c3, f"{n_crit}",   "Critiques",  RC),
        (c4, f"{F1:.4f}",   "F1-Score",   PC),
    ]:
        with col:
            st.markdown(f'<div class="kpi" style="--c:{c};margin-bottom:14px;"><div class="kv">{val}</div><div class="kl">{lbl}</div></div>', unsafe_allow_html=True)

    # Lecture du rapport si disponible
    try:
        with open("rapport_rh_genai.txt","r",encoding="utf-8") as f:
            rapport = f.read()
        st.markdown(f'<div style="background:{GRC};border:1px solid {BOC};border-radius:10px;padding:18px 22px;font-family:monospace;font-size:12px;color:{T2C};white-space:pre-wrap;line-height:1.8;max-height:500px;overflow-y:auto;">{rapport}</div>', unsafe_allow_html=True)
        st.download_button("Telecharger le rapport (.txt)", data=rapport,
            file_name="rapport_rh_genai.txt", mime="text/plain", use_container_width=True)
    except FileNotFoundError:
        st.info("Lancez d'abord le Notebook 5 (5_GenAI.ipynb) pour generer le rapport.")
        # Rapport simplifié automatique
        dept_risque = df.groupby("Department")["Probabilite"].mean()*100
        st.markdown(f"""
        <div style="background:{GRC};border-radius:10px;padding:18px 22px;margin-top:12px;">
          <div style="font-size:14px;font-weight:700;color:{TXC};margin-bottom:12px;">Rapport simplifie</div>
          <div style="font-size:12px;color:{T2C};line-height:1.9;">
            Analyse sur <strong style="color:{TXC};">{n:,} employes</strong>.<br>
            Taux d'attrition : <strong style="color:{RC};">{taux:.1f}%</strong><br>
            Employes critiques : <strong style="color:{RC};">{n_crit}</strong><br>
            Employes a risque eleve : <strong style="color:{OGC};">{n_elev}</strong><br>
            Departement le plus a risque : <strong style="color:{TXC};">{dept_risque.idxmax()} ({dept_risque.max():.1f}%)</strong><br>
            Modele XGBoost : F1={F1:.4f} | AUC={AUC:.4f}
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<hr style="border-color:{BOC};margin:30px 0 10px;">
<div style="text-align:center;color:{T2C};font-size:12px;padding:6px 0;">
  HR Analytics · Seye Kiné | Bindia Adeline Thiara ·
  <span style="color:{OC};">M. Aidara</span> · UCAO 2025-2026
</div>
""", unsafe_allow_html=True)
