# =============================================================================
#  GENERATIVE HR ANALYTICS — PRÉDICTION DU TURNOVER
#  Auteurs   : Seye Kiné | Bindia Adeline Thiara
#  Encadrant : M. Aidara — UCAO 2025-2026
#  Lancement : streamlit run etape11_streamlit.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "HR Turnover Analytics",
    page_icon  = "👥",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Palette de couleurs (UNIQUEMENT hex valides — pas de hex+alpha) ────────────
V  = "#00C896"   # Vert  (reste)
R  = "#FF4B6E"   # Rouge (part)
B  = "#4A9EF5"   # Bleu
O  = "#FFD166"   # Or
P  = "#9B72F5"   # Violet
OG = "#FF8C42"   # Orange
FO = "#0F1923"   # Fond
CA = "#1A2535"   # Carte
GR = "#243044"   # Graphe
TX = "#E8F0FE"   # Texte
T2 = "#8FA3BF"   # Texte 2
GI = "#2A3A50"   # Grille
BO = "#3A4F6A"   # Bord

# rgba sûrs pour Plotly
VA = "rgba(0,200,150,0.12)"
RA = "rgba(255,75,110,0.12)"
BA = "rgba(74,158,245,0.12)"
PA = "rgba(155,114,245,0.12)"

# ── Chargement pipeline ────────────────────────────────────────────────────────
@st.cache_resource
def charger():
    with open("pipeline_data.pkl", "rb") as f:
        return pickle.load(f)

try:
    pip = charger()
except FileNotFoundError:
    st.error("❌ pipeline_data.pkl introuvable.")
    st.stop()

df        = pip["df"]
modele    = pip["modele_final"]
prep      = pip["preprocesseur"]
nom       = pip["nom_final"]
ev        = pip.get("eval_finale", {})
seuil     = pip.get("seuil_optimal", 0.5)
taux      = pip["taux"]
n         = pip["n"]
FEAT      = pip["FEATURES"]
shap_imp  = pip.get("shap_importance", [])
rapport   = pip.get("rapport_genai", "")
CIBLE     = pip["CIBLE"]
t_crit    = pip.get("taux_critique", 0)
t_sain    = pip.get("taux_sain", 0)
res_opt   = pip.get("resultats_opt", {})

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* {{ font-family: 'Inter', 'Segoe UI', sans-serif !important; }}
.stApp {{ background: {FO}; }}
section[data-testid="stSidebar"] {{
    background: {CA} !important;
    border-right: 1px solid {BO};
}}
.stRadio > div {{ gap: 6px; }}
.stRadio > div > label {{
    background: {GR}; border: 1px solid {BO};
    border-radius: 8px; padding: 8px 14px;
    color: {T2} !important; cursor: pointer;
    transition: all 0.2s;
}}
.stRadio > div > label:has(input:checked) {{
    background: {B}22; border-color: {B};
    color: {TX} !important;
}}
div[data-testid="stMetric"] {{
    background: {CA}; border: 1px solid {BO};
    border-radius: 12px; padding: 16px;
}}
div[data-testid="stMetric"] label {{ color: {T2} !important; font-size:11px !important; }}
div[data-testid="stMetricValue"] {{ color: {TX} !important; font-weight:800 !important; }}
.stSelectbox > div > div {{
    background: {GR} !important; border: 1px solid {BO} !important;
    color: {TX} !important; border-radius: 8px !important;
}}
.stSlider [data-baseweb="slider"] {{
    padding: 0 4px;
}}
.stButton > button {{
    border-radius: 10px !important; font-weight: 700 !important;
    transition: all 0.2s !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {B}, {P}) !important;
    border: none !important; color: white !important;
    font-size: 15px !important; padding: 12px 24px !important;
}}
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(74,158,245,0.35) !important;
}}
.stDownloadButton > button {{
    background: {GR} !important; border: 1px solid {BO} !important;
    color: {TX} !important; border-radius: 8px !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    background: {CA}; border-radius: 10px; padding: 4px;
    border: 1px solid {BO};
}}
.stTabs [data-baseweb="tab"] {{
    color: {T2} !important; border-radius: 8px !important;
}}
.stTabs [aria-selected="true"] {{
    background: {GR} !important; color: {TX} !important;
}}
.stDataFrame {{ background: {CA}; }}
h1,h2,h3,h4 {{ color: {TX} !important; }}
p {{ color: {T2}; font-size: 13px; }}
label {{ color: {T2} !important; }}
footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem !important; padding-bottom: 2rem !important; }}

/* Composants custom */
.pg-header {{
    background: linear-gradient(135deg, {CA} 0%, {GR} 100%);
    border: 1px solid {BO}; border-radius: 16px;
    padding: 24px 28px; margin-bottom: 20px;
}}
.pg-title {{ font-size: 26px; font-weight: 900; color: {TX}; margin-bottom: 4px; }}
.pg-sub {{ font-size: 13px; color: {T2}; }}
.kpi-card {{
    background: {CA}; border: 1px solid {BO};
    border-top: 3px solid var(--c);
    border-radius: 14px; padding: 18px 20px;
    text-align: center; transition: transform 0.2s, box-shadow 0.2s;
}}
.kpi-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }}
.kpi-val {{ font-size: 30px; font-weight: 900; color: var(--c); line-height: 1.1; }}
.kpi-lbl {{ font-size: 11px; color: {T2}; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.8px; margin-top: 5px; }}
.kpi-sub {{ font-size: 11px; color: {T2}; opacity: 0.7; margin-top: 3px; }}
.card {{
    background: {CA}; border: 1px solid {BO};
    border-radius: 14px; padding: 20px 24px; margin-bottom: 14px;
}}
.card-title {{ font-size: 14px; font-weight: 700; color: {TX}; margin-bottom: 12px; }}
.insight-box {{
    background: {CA}; border: 1px solid {BO};
    border-left: 4px solid var(--c); border-radius: 0 12px 12px 0;
    padding: 16px 20px;
}}
.insight-title {{ font-size: 10px; font-weight: 700; color: var(--c);
                  text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }}
.insight-val {{ font-size: 24px; font-weight: 900; color: {TX}; margin-bottom: 4px; }}
.insight-txt {{ font-size: 12px; color: {T2}; line-height: 1.6; }}
.pred-box {{
    background: {CA}; border: 2px solid var(--c);
    border-radius: 18px; padding: 28px 24px; text-align: center;
}}
.pred-pct  {{ font-size: 72px; font-weight: 900; color: var(--c); line-height: 1; }}
.pred-lbl  {{ font-size: 18px; font-weight: 700; color: {TX}; margin: 10px 0 4px; }}
.pred-niv  {{ font-size: 15px; color: var(--c); font-weight: 600; }}
.recomm-box {{
    background: {GR}; border-left: 4px solid {B};
    border-radius: 0 12px 12px 0; padding: 14px 18px; margin-top: 14px;
}}
.factor-row {{
    display: flex; align-items: center; gap: 12px;
    background: {GR}; border-radius: 10px;
    padding: 10px 14px; margin-bottom: 6px;
}}
.factor-icon {{ font-size: 20px; min-width: 24px; }}
.factor-label {{ font-size: 13px; color: {T2}; flex: 1; }}
.factor-val {{ font-size: 13px; font-weight: 700; color: var(--vc); }}
.guide-row {{
    display: flex; align-items: center; gap: 14px;
    background: {GR}; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 8px;
}}
.guide-pct {{ font-size: 15px; font-weight: 800; color: var(--c); min-width: 70px; }}
.guide-txt {{ font-size: 13px; color: {T2}; }}
.stat-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid {BO};
}}
.stat-row:last-child {{ border-bottom: none; }}
.stat-label {{ font-size: 13px; color: {T2}; }}
.stat-val {{ font-size: 13px; font-weight: 700; color: {TX}; }}
.rapport-pre {{
    background: {GR}; border: 1px solid {BO}; border-radius: 12px;
    padding: 22px 26px; font-family: 'Courier New', monospace;
    font-size: 12.5px; color: {T2}; white-space: pre-wrap; line-height: 1.8;
    max-height: 500px; overflow-y: auto;
}}
.badge {{
    display: inline-block; font-size: 10px; font-weight: 700;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: 1px; text-transform: uppercase;
    background: var(--c); color: {FO};
}}
</style>
""", unsafe_allow_html=True)

# ── Layout Plotly ─────────────────────────────────────────────────────────────
LAY = dict(
    paper_bgcolor=CA, plot_bgcolor=GR,
    font=dict(color=TX, family="Inter, Segoe UI, sans-serif"),
    legend=dict(bgcolor=CA, bordercolor=BO, font=dict(color=TX)),
)
def ax(t=""):
    return dict(title=t, gridcolor=GI, showgrid=True,
                zeroline=False, tickfont=dict(color=T2))

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:20px 0 16px;">
      <div style="font-size:48px; margin-bottom:8px;">👥</div>
      <div style="font-size:15px; font-weight:800; color:{TX};">HR Turnover Analytics</div>
      <div style="font-size:11px; color:{T2}; margin-top:6px; line-height:1.8;">
        Seye Kiné &nbsp;|&nbsp; Bindia Adeline Thiara<br>
        <span style="color:{O}; font-weight:600;">M. Aidara</span><br>
        UCAO 2025-2026
      </div>
    </div>
    <hr style="border-color:{BO}; margin:0 0 16px;">
    """, unsafe_allow_html=True)

    nav = st.radio("Navigation", [
        "🏠  Accueil",
        "📊  Exploration",
        "🤖  Prédiction",
        "🔍  Explicabilité",
        "📋  Rapport IA",
    ], label_visibility="collapsed")

    st.markdown(f"""
    <hr style="border-color:{BO}; margin:16px 0 12px;">
    <div style="background:{GR}; border-radius:12px; padding:14px 16px;">
      <div style="font-size:10px; color:{T2}; text-transform:uppercase;
           letter-spacing:1px; margin-bottom:10px; font-weight:700;">
        Modèle actif
      </div>
      <div style="font-size:13px; font-weight:700; color:{TX}; margin-bottom:8px;">
        {nom}
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
        <span style="font-size:12px; color:{T2};">F1-Score</span>
        <span style="font-size:12px; font-weight:700; color:{V};">{ev.get('f1',0):.4f}</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
        <span style="font-size:12px; color:{T2};">AUC-ROC</span>
        <span style="font-size:12px; font-weight:700; color:{B};">{ev.get('auc',0):.4f}</span>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span style="font-size:12px; color:{T2};">Seuil optimal</span>
        <span style="font-size:12px; font-weight:700; color:{O};">{seuil:.2f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE 1 — ACCUEIL
# =============================================================================
if nav == "🏠  Accueil":

    # Header
    st.markdown(f"""
    <div class="pg-header">
      <div class="pg-title">👥 HR Turnover Analytics Dashboard</div>
      <div class="pg-sub">
        Generative HR Analytics &amp; Explicabilité des Modèles &nbsp;|&nbsp;
        Prédiction du Turnover des Employés &nbsp;|&nbsp; UCAO 2025-2026
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kdata = [
        (k1, f"{n:,}", "Employés analysés",  B,  "Dataset complet"),
        (k2, f"{taux:.1f}%","Taux d'attrition", R,  f"{int(taux/100*n):,} départs"),
        (k3, f"{ev.get('f1',0):.4f}","F1-Score", P, "Métrique principale"),
        (k4, f"{ev.get('auc',0):.4f}","AUC-ROC",  B, "Pouvoir discriminant"),
        (k5, f"{ev.get('tp',0):,}", "Partis détectés",V, f"Sur {ev.get('tp',0)+ev.get('fn',0):,} réels"),
        (k6, f"{seuil:.2f}", "Seuil optimal",   O,  "Décision de risque"),
    ]
    for col,val,lbl,c,sub in kdata:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--c:{c};">
              <div class="kpi-val">{val}</div>
              <div class="kpi-lbl">{lbl}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphiques ligne 1
    g1, g2 = st.columns(2)
    with g1:
        n0 = int((df[CIBLE]==0).sum())
        n1 = int((df[CIBLE]==1).sum())
        fig = go.Figure(go.Pie(
            values=[n0,n1], labels=["Resté","Parti"], hole=0.62,
            marker=dict(colors=[V,R], line=dict(color=FO,width=3)),
            textinfo="label+percent", textfont=dict(size=13,color="white"),
            pull=[0,0.06],
        ))
        fig.add_annotation(text=f"<b>{taux:.1f}%</b><br>Départs",
            x=0.5,y=0.5,showarrow=False,font=dict(size=20,color=R))
        fig.update_layout(**LAY, title=dict(text="Distribution de l'Attrition",
            font=dict(size=14),x=0.5), height=320,
            margin=dict(t=50,b=20,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        ot = df.groupby("OverTime")[CIBLE].mean()*100
        js = df.groupby("JobSatisfaction")[CIBLE].mean()*100
        fig2 = make_subplots(rows=1,cols=2,
            subplot_titles=["Heures supplémentaires","Satisfaction au travail"],
            horizontal_spacing=0.12)
        fig2.add_trace(go.Bar(
            x=["Sans","Avec"], y=ot.values,
            marker=dict(color=[V,R],line=dict(color=FO,width=2)),
            text=[f"{v:.1f}%" for v in ot.values], textposition="outside",
            showlegend=False,
        ),row=1,col=1)
        fig2.add_trace(go.Bar(
            x=["1","2","3","4"], y=js.values,
            marker=dict(color=[R,OG,O,V],line=dict(color=FO,width=2)),
            text=[f"{v:.1f}%" for v in js.values], textposition="outside",
            showlegend=False,
        ),row=1,col=2)
        fig2.update_layout(**LAY,
            title=dict(text="Principaux facteurs d'attrition",
                       font=dict(size=14),x=0.5),
            height=320,margin=dict(t=55,b=30,l=40,r=20))
        for i in [1,2]:
            fig2.update_yaxes(gridcolor=GI,range=[0,65],row=1,col=i)
            fig2.update_xaxes(gridcolor=GI,row=1,col=i)
        st.plotly_chart(fig2, use_container_width=True)

    # Graphiques ligne 2
    g3, g4 = st.columns(2)
    with g3:
        dept = df.groupby("Department")[CIBLE].mean()*100
        dept = dept.sort_values()
        coul_d = [R if v>taux else B for v in dept.values]
        fig3 = go.Figure(go.Bar(
            x=dept.values, y=dept.index, orientation="h",
            marker=dict(color=coul_d,line=dict(color=FO,width=1.5)),
            text=[f"{v:.1f}%" for v in dept.values], textposition="outside",
        ))
        fig3.add_vline(x=taux,line_dash="dash",line_color=O,
            annotation_text=f"Moy. {taux:.1f}%",
            annotation_font=dict(color=O))
        fig3.update_layout(**LAY,
            title=dict(text="Attrition par Département",font=dict(size=14),x=0.5),
            xaxis=dict(**ax("Taux (%)"),range=[0,58]),
            yaxis=ax(), height=300, showlegend=False,
            margin=dict(t=50,b=30,l=100,r=60))
        st.plotly_chart(fig3, use_container_width=True)

    with g4:
        # Tableau comparaison modèles
        st.markdown(f"""
        <div class="card">
          <div class="card-title">📊 Comparaison des 4 modèles (après optimisation)</div>
        """, unsafe_allow_html=True)
        if res_opt:
            rows = []
            rangs = ["🥇","🥈","🥉","4️⃣"]
            tri = sorted(res_opt.items(),key=lambda x:x[1]["f1"],reverse=True)
            for i,(nm,re) in enumerate(tri):
                rows.append({
                    "":"" + rangs[i],
                    "Modèle": nm,
                    "F1":  f"{re['f1']:.4f}",
                    "AUC": f"{re['auc']:.4f}",
                    "Acc": f"{re['acc']:.4f}",
                })
            st.dataframe(pd.DataFrame(rows),use_container_width=True,
                         hide_index=True, height=260)
        st.markdown("</div>", unsafe_allow_html=True)

    # Insights ligne 3
    st.markdown(f"""
    <div style="font-size:16px; font-weight:800; color:{TX};
         margin:20px 0 14px; padding-bottom:8px; border-bottom:1px solid {BO};">
      💡 Insights clés du projet
    </div>
    """, unsafe_allow_html=True)
    i1,i2,i3,i4 = st.columns(4)
    ins = [
        (i1, R, "Facteur #1 — OverTime",
         "+14.9 pts",
         f"Avec heures sup → {df[df['OverTime']=='Yes'][CIBLE].mean()*100:.1f}% vs "
         f"sans → {df[df['OverTime']=='No'][CIBLE].mean()*100:.1f}%"),
        (i2, P, "Facteur #2 — JobSatisfaction",
         "r = −0.14",
         "Satisfaction 1 → 52.6% de départs. Satisfaction 4 → 36.2%."),
        (i3, R, "Profil critique",
         f"{t_crit:.1f}%",
         "OverTime=Oui + JS≤2 + WLB≤2 → risque très élevé de départ."),
        (i4, V, "Profil sain",
         f"{t_sain:.1f}%",
         "OverTime=Non + JS≥3 + WLB≥3 → profil stable, faible risque."),
    ]
    for col,c,titre,val,txt in ins:
        with col:
            st.markdown(f"""
            <div class="insight-box" style="--c:{c};">
              <div class="insight-title">{titre}</div>
              <div class="insight-val">{val}</div>
              <div class="insight-txt">{txt}</div>
            </div>""", unsafe_allow_html=True)

# =============================================================================
# PAGE 2 — EXPLORATION
# =============================================================================
elif nav == "📊  Exploration":

    st.markdown(f"""
    <div class="pg-header">
      <div class="pg-title">📊 Exploration des Données</div>
      <div class="pg-sub">
        {n:,} employés × {df.shape[1]} variables &nbsp;|&nbsp;
        Taux d'attrition global : {taux:.1f}%
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs([
        "📈 Par variable", "🔀 Croisements", "🔗 Corrélations", "📋 Statistiques"
    ])

    with tab1:
        col_sel, _ = st.columns([1,2])
        with col_sel:
            var_cat = st.selectbox("Variable à analyser :",
                ["OverTime","Department","JobRole","Gender"],
                format_func=lambda x: {
                    "OverTime":"⏰ Heures supplémentaires",
                    "Department":"🏢 Département",
                    "JobRole":"💼 Poste occupé",
                    "Gender":"👤 Genre"
                }[x])

        vals = df.groupby(var_cat)[CIBLE].mean()*100
        counts = df.groupby(var_cat)[CIBLE].count()
        vals = vals.sort_values(ascending=True)
        coul = [R if v>taux else B for v in vals.values]

        c_left, c_right = st.columns([2,1])
        with c_left:
            fig = go.Figure(go.Bar(
                x=vals.values, y=vals.index, orientation="h",
                marker=dict(color=coul,line=dict(color=FO,width=1.5)),
                text=[f"<b>{v:.1f}%</b>" for v in vals.values],
                textposition="outside",
                customdata=[[counts[i]] for i in vals.index],
                hovertemplate="%{y}<br>Taux : %{x:.1f}%<br>Effectif : %{customdata[0]:,}<extra></extra>",
            ))
            fig.add_vline(x=taux, line_dash="dash", line_color=O,
                annotation_text=f"Moy. {taux:.1f}%",
                annotation_font=dict(color=O,size=12))
            fig.update_layout(**LAY,
                title=dict(text=f"Taux d'attrition par {var_cat}",
                           font=dict(size=14),x=0.5),
                xaxis=dict(**ax("Taux d'attrition (%)"),range=[0,65]),
                yaxis=ax(), height=380, showlegend=False,
                margin=dict(t=50,b=40,l=160,r=70))
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">📋 Détail par modalité</div>
            """, unsafe_allow_html=True)
            for v_idx in vals.sort_values(ascending=False).index:
                pct_v = vals[v_idx]
                eff   = int(counts[v_idx])
                col_v = R if pct_v > taux else V
                delta = pct_v - taux
                st.markdown(f"""
                <div class="stat-row">
                  <span class="stat-label">{v_idx}</span>
                  <div style="text-align:right;">
                    <span style="font-size:14px; font-weight:800; color:{col_v};">{pct_v:.1f}%</span><br>
                    <span style="font-size:11px; color:{T2};">{eff:,} emp. ({delta:+.1f})</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        c2a, c2b = st.columns(2)
        with c2a:
            piv = df.groupby(["JobSatisfaction","WorkLifeBalance"])[CIBLE].mean()*100
            piv = piv.unstack().round(1)
            fig_hm = go.Figure(go.Heatmap(
                z=piv.values,
                x=[f"WLB={c}" for c in piv.columns],
                y=[f"JS={r}" for r in piv.index],
                text=[[f"<b>{v:.1f}%</b>" for v in row] for row in piv.values],
                texttemplate="%{text}", textfont=dict(size=14,color="white"),
                colorscale=[[0,V],[0.4,O],[0.7,OG],[1,R]],
                zmid=taux, zmin=25, zmax=65,
                colorbar=dict(title="Taux %",tickfont=dict(color=T2)),
                hovertemplate="JS=%{y} × %{x}<br>Taux = %{z:.1f}%<extra></extra>",
            ))
            fig_hm.update_layout(**LAY,
                title=dict(text="JS × WorkLifeBalance — Taux d'attrition (%)",
                           font=dict(size=13),x=0.5),
                height=360, margin=dict(t=50,b=60,l=90,r=30))
            st.plotly_chart(fig_hm, use_container_width=True)

        with c2b:
            # OverTime × Department
            ot_dept = df.groupby(["Department","OverTime"])[CIBLE].mean()*100
            ot_dept = ot_dept.unstack().round(1)
            fig_od = go.Figure()
            for ot_v, col_v, nom_v in [("No",V,"Sans heures sup"),("Yes",R,"Avec heures sup")]:
                if ot_v in ot_dept.columns:
                    fig_od.add_trace(go.Bar(
                        name=nom_v, x=ot_dept.index,
                        y=ot_dept[ot_v].values,
                        marker=dict(color=col_v,line=dict(color=FO,width=1.5)),
                        text=[f"{v:.1f}%" for v in ot_dept[ot_v].values],
                        textposition="inside",
                    ))
            fig_od.add_hline(y=taux,line_dash="dash",line_color=O,
                annotation_text=f"Moy. {taux:.1f}%",
                annotation_font=dict(color=O))
            fig_od.update_layout(**LAY,
                title=dict(text="Département × OverTime",font=dict(size=13),x=0.5),
                yaxis=dict(**ax("Taux (%)"),range=[0,70]),
                xaxis=ax(), height=360,
                margin=dict(t=50,b=50,l=55,r=20),
                barmode="group")
            st.plotly_chart(fig_od, use_container_width=True)

        # Explication
        st.markdown(f"""
        <div class="card">
          <div class="card-title">💡 Comment lire ces graphiques ?</div>
          <p>
            <span style="color:{R}; font-weight:700;">■ Rouge/Chaud</span> = taux d'attrition élevé (risque fort)
            &nbsp;&nbsp;
            <span style="color:{V}; font-weight:700;">■ Vert/Froid</span> = taux d'attrition faible (risque faible)<br><br>
            La cellule <strong style="color:{TX};">JS=1 + WLB=1</strong> est la plus critique : combinaison
            de faible satisfaction + mauvais équilibre vie pro/perso = risque maximum.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        cols_num = pip["COLS_NUM"]
        corr = df[cols_num+[CIBLE]].corr()[CIBLE].drop(CIBLE)
        corr_abs = corr.abs().sort_values(ascending=True)
        coul_c = [R if corr[v]<0 else B for v in corr_abs.index]

        fig_c = go.Figure(go.Bar(
            x=corr_abs.values, y=corr_abs.index, orientation="h",
            marker=dict(color=coul_c, line=dict(color=FO,width=1.5)),
            text=[f"{corr[v]:+.4f}" for v in corr_abs.index],
            textposition="outside",
            hovertemplate="%{y}<br>Corrélation = %{text}<extra></extra>",
        ))
        fig_c.update_layout(**LAY,
            title=dict(text="Corrélations des variables numériques avec l'Attrition",
                       font=dict(size=14),x=0.5),
            xaxis=dict(**ax("Corrélation absolue")),
            yaxis=ax(), height=460,
            margin=dict(t=50,b=40,l=210,r=80), showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)

        st.markdown(f"""
        <div class="card">
          <div class="card-title">📖 Interprétation</div>
          <p>
            <span style="color:{R}; font-weight:700;">■ Rouge</span> = corrélation négative :
            quand la variable augmente, l'attrition diminue (ex: JobSatisfaction ↑ → départs ↓)<br>
            <span style="color:{B}; font-weight:700;">■ Bleu</span> = corrélation positive :
            quand la variable augmente, l'attrition augmente (ex: YearsSinceLastPromotion ↑ → départs ↑)<br><br>
            Plus la barre est longue, plus le lien avec l'attrition est fort.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown(f"""
        <div class="card"><div class="card-title">📋 Statistiques descriptives complètes</div>
        """, unsafe_allow_html=True)
        stats = df[cols_num].describe().round(2)
        st.dataframe(stats, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">🎯 Variables catégorielles</div>
            """, unsafe_allow_html=True)
            for col in pip["COLS_CAT"]:
                vc = df[col].value_counts()
                st.markdown(f"**{col}** :")
                for v_name, cnt in vc.items():
                    pct_v = cnt/n*100
                    st.markdown(f"""
                    <div class="stat-row">
                      <span class="stat-label">{v_name}</span>
                      <span class="stat-val">{cnt:,} ({pct_v:.1f}%)</span>
                    </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_s2:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">📊 Qualité des données</div>
              <div class="stat-row">
                <span class="stat-label">Valeurs manquantes</span>
                <span style="color:{V}; font-weight:700;">0 ✅</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Doublons</span>
                <span style="color:{V}; font-weight:700;">0 ✅</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Outliers (IQR)</span>
                <span style="color:{V}; font-weight:700;">0 ✅</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Employés total</span>
                <span class="stat-val">{n:,}</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Variables total</span>
                <span class="stat-val">{df.shape[1]}</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Variables numériques</span>
                <span class="stat-val">{len(pip['COLS_NUM'])}</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Variables catégorielles</span>
                <span class="stat-val">{len(pip['COLS_CAT'])}</span>
              </div>
              <div class="stat-row">
                <span class="stat-label">Taux d'attrition</span>
                <span style="color:{R}; font-weight:700;">{taux:.1f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PAGE 3 — PRÉDICTION
# =============================================================================
elif nav == "🤖  Prédiction":

    st.markdown(f"""
    <div class="pg-header">
      <div class="pg-title">🤖 Prédiction du Risque de Départ</div>
      <div class="pg-sub">
        Renseignez les informations d'un employé pour obtenir son score de risque instantané
      </div>
    </div>
    """, unsafe_allow_html=True)

    c_form, c_res = st.columns([1,1], gap="large")

    with c_form:

        # ── Bloc 1 : Satisfaction ──────────────────────────────────────────────
        st.markdown(f"""
        <div style="font-size:11px; font-weight:700; color:{B};
             text-transform:uppercase; letter-spacing:1.2px; margin-bottom:12px;">
          😊 Bien-être au travail
        </div>
        """, unsafe_allow_html=True)

        job_sat = st.select_slider(
            "Satisfaction au travail",
            options=[1,2,3,4],
            value=3,
            format_func=lambda x:{
                1:"😞 1 — Très insatisfait",
                2:"😐 2 — Insatisfait",
                3:"🙂 3 — Satisfait",
                4:"😄 4 — Très satisfait",
            }[x],
        )
        wlb = st.select_slider(
            "Équilibre vie professionnelle / personnelle",
            options=[1,2,3,4],
            value=3,
            format_func=lambda x:{
                1:"😰 1 — Très mauvais",
                2:"😕 2 — Mauvais",
                3:"😊 3 — Bon",
                4:"🌟 4 — Excellent",
            }[x],
        )

        # Alerte visuelle
        if job_sat <= 2 and wlb <= 2:
            st.markdown(f"""
            <div style="background:rgba(255,75,110,0.1); border:1px solid {R};
                 border-radius:8px; padding:8px 14px; font-size:12px; color:{R}; margin-top:4px;">
              ⚠️ Combinaison à risque élevé détectée
            </div>""", unsafe_allow_html=True)
        elif job_sat >= 3 and wlb >= 3:
            st.markdown(f"""
            <div style="background:rgba(0,200,150,0.1); border:1px solid {V};
                 border-radius:8px; padding:8px 14px; font-size:12px; color:{V}; margin-top:4px;">
              ✅ Profil satisfaisant
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bloc 2 : Conditions ───────────────────────────────────────────────
        st.markdown(f"""
        <div style="font-size:11px; font-weight:700; color:{B};
             text-transform:uppercase; letter-spacing:1.2px; margin-bottom:12px;">
          ⚡ Conditions de travail
        </div>
        """, unsafe_allow_html=True)

        overtime = st.radio(
            "Heures supplémentaires régulières",
            options=["✅  Non — travail normal", "⚠️  Oui — heures supplémentaires"],
            index=0,
        )
        ot_val = "Yes" if "Oui" in overtime else "No"

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bloc 3 : Carrière ─────────────────────────────────────────────────
        st.markdown(f"""
        <div style="font-size:11px; font-weight:700; color:{B};
             text-transform:uppercase; letter-spacing:1.2px; margin-bottom:12px;">
          📈 Carrière & Mobilité
        </div>
        """, unsafe_allow_html=True)

        c3a, c3b = st.columns(2)
        with c3a:
            years_promo = st.number_input(
                "📅 Années sans promotion",
                min_value=0, max_value=15, value=3, step=1,
                help="Nombre d'années depuis la dernière promotion",
            )
        with c3b:
            distance = st.number_input(
                "🏠 Distance domicile-travail (km)",
                min_value=1, max_value=40, value=15, step=1,
                help="Distance en km domicile → bureau",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bloc 4 : Profil ───────────────────────────────────────────────────
        st.markdown(f"""
        <div style="font-size:11px; font-weight:700; color:{B};
             text-transform:uppercase; letter-spacing:1.2px; margin-bottom:12px;">
          👤 Profil de l'employé
        </div>
        """, unsafe_allow_html=True)

        c4a, c4b = st.columns(2)
        with c4a:
            gender = st.selectbox("Genre",
                ["Male","Female"],
                format_func=lambda x:"👨 Homme" if x=="Male" else "👩 Femme")
            dept = st.selectbox("Département",
                ["Finance","HR","IT","Marketing","R&D","Sales"],
                format_func=lambda x:{
                    "Finance":"💰 Finance","HR":"👥 RH / HR",
                    "IT":"💻 IT","Marketing":"📣 Marketing",
                    "R&D":"🔬 R&D","Sales":"🛒 Sales"
                }[x])
        with c4b:
            role = st.selectbox("Poste occupé", [
                "Analyst","Consultant","Engineer",
                "HR Specialist","Manager",
                "Sales Executive","Technician",
            ], format_func=lambda x:{
                "Analyst":"📊 Analyst","Consultant":"💼 Consultant",
                "Engineer":"⚙️ Engineer","HR Specialist":"👥 HR Specialist",
                "Manager":"🎯 Manager","Sales Executive":"🛒 Sales Executive",
                "Technician":"🔧 Technician",
            }[x])

        st.markdown("<br>", unsafe_allow_html=True)
        analyser = st.button("🔮  Analyser le risque de départ",
                              use_container_width=True, type="primary")

    # ── Résultat ───────────────────────────────────────────────────────────────
    with c_res:
        if analyser:
            df_in = pd.DataFrame([{
                "YearsSinceLastPromotion": years_promo,
                "DistanceFromHome"       : distance,
                "JobSatisfaction"        : job_sat,
                "WorkLifeBalance"        : wlb,
                "Gender"                 : gender,
                "Department"             : dept,
                "JobRole"                : role,
                "OverTime"               : ot_val,
            }])
            X_p   = prep.transform(df_in)
            proba = float(modele.predict_proba(X_p)[0][1])
            pct   = proba * 100

            if pct >= 75:
                c_r=R; niv="🔴 Risque Critique"; ico="🚨"
                rec=("Intervention RH immédiate. Planifiez un entretien "
                     "de rétention sous 48h. Revoir la charge de travail "
                     "et les conditions d'emploi en urgence.")
            elif pct >= 60:
                c_r=OG; niv="🟠 Risque Élevé"; ico="⚠️"
                rec=("Entretien individuel sous 2 semaines recommandé. "
                     "Identifier les sources d'insatisfaction et proposer "
                     "des solutions concrètes.")
            elif pct >= seuil*100:
                c_r=O; niv="🟡 Risque Modéré"; ico="👁️"
                rec=("Inclure dans le suivi mensuel RH. Vérifier satisfaction "
                     "et équilibre vie pro/perso. Envisager une discussion "
                     "sur les perspectives d'évolution.")
            else:
                c_r=V; niv="🟢 Risque Faible"; ico="✅"
                rec=("Profil stable. Maintenir les bonnes conditions. "
                     "Continuer les entretiens annuels habituels.")

            # Score principal
            st.markdown(f"""
            <div class="pred-box" style="--c:{c_r};">
              <div style="font-size:42px; margin-bottom:6px;">{ico}</div>
              <div class="pred-pct">{pct:.1f}%</div>
              <div class="pred-lbl">Probabilité de départ</div>
              <div class="pred-niv">{niv}</div>
              <div style="font-size:11px; color:{T2}; margin-top:10px;">
                Seuil de décision : {seuil*100:.0f}% &nbsp;|&nbsp; Modèle : {nom}
              </div>
            </div>
            <div class="recomm-box">
              <div style="font-size:12px; font-weight:700; color:{B}; margin-bottom:6px;">
                💡 Recommandation RH
              </div>
              <div style="font-size:13px; color:{T2};">{rec}</div>
            </div>
            """, unsafe_allow_html=True)

            # Jauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix":"%","font":{"size":28,"color":TX}},
                gauge={
                    "axis"      : {"range":[0,100],"tickcolor":T2,
                                    "ticksuffix":"%"},
                    "bar"       : {"color":c_r,"thickness":0.22},
                    "bgcolor"   : GR,
                    "bordercolor": BO,
                    "steps"     : [
                        {"range":[0, seuil*100],  "color":VA},
                        {"range":[seuil*100,100],  "color":RA},
                    ],
                    "threshold" : {
                        "line"     : {"color":O,"width":3},
                        "thickness": 0.75,
                        "value"    : seuil*100,
                    },
                },
            ))
            fig_g.update_layout(
                paper_bgcolor=CA, plot_bgcolor=CA,
                font=dict(color=TX),
                height=210,
                margin=dict(t=15,b=10,l=25,r=25),
            )
            st.plotly_chart(fig_g, use_container_width=True)

            # Facteurs saisis
            st.markdown(f"""
            <div style="font-size:11px; font-weight:700; color:{T2};
                 text-transform:uppercase; letter-spacing:1px; margin:8px 0 6px;">
              Facteurs analysés
            </div>
            """, unsafe_allow_html=True)

            facts = [
                ("😊","Satisfaction travail",f"{job_sat}/4",
                 R if job_sat<=2 else V),
                ("⚖️","Équilibre WLB",       f"{wlb}/4",
                 R if wlb<=2 else V),
                ("⏰","Heures supp.",         "Oui" if ot_val=="Yes" else "Non",
                 R if ot_val=="Yes" else V),
                ("📅","Ans sans promotion",   f"{years_promo} ans",
                 R if years_promo>=5 else V),
                ("🏠","Distance",             f"{distance} km",
                 OG if distance>=25 else V),
            ]
            for ic,lb,vl,col_f in facts:
                st.markdown(f"""
                <div class="factor-row">
                  <span class="factor-icon">{ic}</span>
                  <span class="factor-label">{lb}</span>
                  <span class="factor-val" style="--vc:{col_f};">{vl}</span>
                </div>
                """, unsafe_allow_html=True)

        else:
            # État initial avec guide
            st.markdown(f"""
            <div style="background:{CA}; border:2px dashed {BO};
                 border-radius:18px; padding:40px 28px; text-align:center;
                 margin-bottom:16px;">
              <div style="font-size:52px; margin-bottom:12px;">🔮</div>
              <div style="font-size:16px; font-weight:700; color:{TX}; margin-bottom:6px;">
                Prêt pour l'analyse
              </div>
              <div style="font-size:13px; color:{T2};">
                Remplissez le formulaire à gauche<br>
                puis cliquez sur <strong style="color:{B};">Analyser le risque de départ</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card">
              <div class="card-title">📖 Échelle de risque</div>
              <div class="guide-row" style="--c:{R};">
                <span class="guide-pct">≥ 75%</span>
                <span class="guide-txt">🚨 Risque Critique — Action immédiate</span>
              </div>
              <div class="guide-row" style="--c:{OG};">
                <span class="guide-pct">60–75%</span>
                <span class="guide-txt">⚠️ Risque Élevé — Entretien sous 2 sem.</span>
              </div>
              <div class="guide-row" style="--c:{O};">
                <span class="guide-pct">{seuil*100:.0f}–60%</span>
                <span class="guide-txt">👁️ Risque Modéré — Suivi mensuel</span>
              </div>
              <div class="guide-row" style="--c:{V};">
                <span class="guide-pct">&lt; {seuil*100:.0f}%</span>
                <span class="guide-txt">✅ Risque Faible — Profil stable</span>
              </div>
            </div>
            <div class="card">
              <div class="card-title">🧠 Comment fonctionne la prédiction ?</div>
              <p>Le modèle <strong style="color:{TX};">{nom}</strong>
              analyse 8 variables pour calculer la probabilité qu'un employé
              quitte l'entreprise.<br><br>
              Les variables les plus importantes sont :<br>
              <span style="color:{R};">■</span> Heures supplémentaires (+14.9 pts de risque)<br>
              <span style="color:{R};">■</span> Satisfaction au travail (corr = -0.14)<br>
              <span style="color:{OG};">■</span> Équilibre vie pro/perso (corr = -0.08)
              </p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PAGE 4 — EXPLICABILITÉ SHAP
# =============================================================================
elif nav == "🔍  Explicabilité":

    st.markdown(f"""
    <div class="pg-header">
      <div class="pg-title">🔍 Explicabilité — SHAP</div>
      <div class="pg-sub">
        Comprendre POURQUOI le modèle prédit un départ pour chaque employé
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
      <div class="card-title">🧠 Qu'est-ce que SHAP ?</div>
      <p>
        SHAP <em>(SHapley Additive exPlanations)</em> est une méthode mathématique
        qui mesure la contribution de chaque variable à chaque prédiction.<br><br>
        ▶ <strong style="color:{R};">Valeur SHAP élevée</strong> = cette variable
        <strong>augmente</strong> le risque de départ<br>
        ▶ <strong style="color:{V};">Valeur SHAP faible</strong> = cette variable
        <strong>réduit</strong> le risque de départ<br><br>
        C'est ce qui rend le modèle <strong style="color:{TX};">"boîte blanche"</strong>
        : le DRH comprend exactement pourquoi une alerte est déclenchée.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not shap_imp:
        st.warning("⚠️ Lancez etape7_shap_lime.py pour calculer les valeurs SHAP.")
    else:
        df_sh = pd.DataFrame(shap_imp).sort_values("Importance",ascending=True)
        med_sh = df_sh["Importance"].median()
        coul_sh = [R if v>med_sh else B for v in df_sh["Importance"]]

        tab_sh1, tab_sh2 = st.tabs(["📊 Importance globale", "🏆 Top variables"])

        with tab_sh1:
            fig_sh = go.Figure(go.Bar(
                x=df_sh["Importance"], y=df_sh["Variable"],
                orientation="h",
                marker=dict(color=coul_sh,line=dict(color=FO,width=1.5)),
                text=[f"<b>{v:.4f}</b>" for v in df_sh["Importance"]],
                textposition="outside",
                hovertemplate="%{y}<br>SHAP moyen = %{x:.4f}<extra></extra>",
            ))
            fig_sh.add_vline(x=med_sh,line_dash="dash",line_color=O,
                annotation_text="Médiane",
                annotation_font=dict(color=O))
            fig_sh.update_layout(**LAY,
                title=dict(text="Importance globale SHAP — Contribution de chaque variable",
                           font=dict(size=14),x=0.5),
                xaxis=dict(**ax("Importance SHAP (|valeur| moyenne)")),
                yaxis=ax(), height=520,
                margin=dict(t=55,b=40,l=220,r=90),
                showlegend=False)
            st.plotly_chart(fig_sh, use_container_width=True)

        with tab_sh2:
            top5 = pd.DataFrame(shap_imp).sort_values("Importance",ascending=False).head(5)
            cols_top = st.columns(len(top5))
            colors_top = [R,P,OG,B,V]
            for i,(col_t,(_, row)) in enumerate(zip(cols_top, top5.iterrows())):
                with col_t:
                    st.markdown(f"""
                    <div class="insight-box" style="--c:{colors_top[i]};">
                      <div class="insight-title">#{i+1} Variable SHAP</div>
                      <div class="insight-val" style="font-size:15px; word-break:break-word;">
                        {row['Variable']}
                      </div>
                      <div class="insight-txt">
                        SHAP = {row['Importance']:.4f}<br>
                        {"⬆️ Augmente le risque" if i < 2 else "Influence modérée"}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card">
              <div class="card-title">💡 Comment utiliser SHAP en pratique ?</div>
              <p>
                Lors d'une alerte sur un employé, le DRH peut consulter ses valeurs SHAP
                pour savoir <strong style="color:{TX};">exactement quels facteurs</strong>
                contribuent à son risque de départ.<br><br>
                Exemple : Si un employé a une probabilité de 68%, SHAP peut révéler que
                c'est principalement dû à <strong style="color:{R};">OverTime=Oui (+0.15)</strong>
                et à <strong style="color:{R};">JobSatisfaction=1 (+0.12)</strong>.
                Le DRH sait alors sur quoi agir en priorité.
              </p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PAGE 5 — RAPPORT IA
# =============================================================================
elif nav == "📋  Rapport IA":

    st.markdown(f"""
    <div class="pg-header">
      <div class="pg-title">📋 Rapport RH Automatique</div>
      <div class="pg-sub">
        Généré automatiquement par le module IA Générative — Étape 9
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs rapport
    r1,r2,r3,r4 = st.columns(4)
    for col,val,lbl,c,sub in [
        (r1,f"{ev.get('tp',0):,}","Partis détectés",V,f"TP sur {ev.get('tp',0)+ev.get('fn',0):,}"),
        (r2,f"{ev.get('fn',0):,}","Partis manqués", R,"Faux négatifs"),
        (r3,f"{ev.get('f1',0):.4f}","F1-Score",      P,"Métrique finale"),
        (r4,f"{ev.get('auc',0):.4f}","AUC-ROC",       B,"Pouvoir discrim."),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--c:{c}; margin-bottom:14px;">
              <div class="kpi-val">{val}</div>
              <div class="kpi-lbl">{lbl}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Métriques détaillées en 2 colonnes
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">📊 Métriques complètes du modèle final</div>
          <div class="stat-row"><span class="stat-label">Modèle sélectionné</span>
            <span style="color:{B}; font-weight:700;">{nom}</span></div>
          <div class="stat-row"><span class="stat-label">Accuracy</span>
            <span class="stat-val">{ev.get('acc',0):.4f}</span></div>
          <div class="stat-row"><span class="stat-label">Précision</span>
            <span class="stat-val">{ev.get('prec',0):.4f}</span></div>
          <div class="stat-row"><span class="stat-label">Rappel (Sensibilité)</span>
            <span class="stat-val">{ev.get('rec',0):.4f}</span></div>
          <div class="stat-row"><span class="stat-label">F1-Score</span>
            <span style="color:{P}; font-weight:700;">{ev.get('f1',0):.4f}</span></div>
          <div class="stat-row"><span class="stat-label">AUC-ROC</span>
            <span style="color:{B}; font-weight:700;">{ev.get('auc',0):.4f}</span></div>
          <div class="stat-row"><span class="stat-label">Balanced Accuracy</span>
            <span class="stat-val">{ev.get('bal_acc',0):.4f}</span></div>
          <div class="stat-row"><span class="stat-label">MCC</span>
            <span class="stat-val">{ev.get('mcc',0):.4f}</span></div>
          <div class="stat-row"><span class="stat-label">Seuil optimal</span>
            <span style="color:{O}; font-weight:700;">{seuil:.2f}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">🎯 Matrice de confusion</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:8px;">
            <div style="background:{V}22; border:1px solid {V}; border-radius:10px;
                 padding:16px; text-align:center;">
              <div style="font-size:28px; font-weight:900; color:{V};">{ev.get('tp',0):,}</div>
              <div style="font-size:11px; color:{T2}; margin-top:4px;">Vrais Positifs (TP)<br>
                <span style="color:{V};">Partis correctement détectés</span></div>
            </div>
            <div style="background:{R}22; border:1px solid {R}; border-radius:10px;
                 padding:16px; text-align:center;">
              <div style="font-size:28px; font-weight:900; color:{R};">{ev.get('fn',0):,}</div>
              <div style="font-size:11px; color:{T2}; margin-top:4px;">Faux Négatifs (FN)<br>
                <span style="color:{R};">Départs non anticipés</span></div>
            </div>
            <div style="background:{OG}22; border:1px solid {OG}; border-radius:10px;
                 padding:16px; text-align:center;">
              <div style="font-size:28px; font-weight:900; color:{OG};">{ev.get('fp',0):,}</div>
              <div style="font-size:11px; color:{T2}; margin-top:4px;">Faux Positifs (FP)<br>
                <span style="color:{OG};">Fausses alertes RH</span></div>
            </div>
            <div style="background:{B}22; border:1px solid {B}; border-radius:10px;
                 padding:16px; text-align:center;">
              <div style="font-size:28px; font-weight:900; color:{B};">{ev.get('tn',0):,}</div>
              <div style="font-size:11px; color:{T2}; margin-top:4px;">Vrais Négatifs (TN)<br>
                <span style="color:{B};">Restés correctement prédits</span></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Rapport narratif
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:16px; font-weight:800; color:{TX};
         margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid {BO};">
      📄 Rapport narratif complet
    </div>
    """, unsafe_allow_html=True)

    if rapport:
        st.markdown(f"""
        <div class="rapport-pre">{rapport}</div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Télécharger le rapport complet (.txt)",
            data=rapport,
            file_name="rapport_rh_turnover_analytics.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.info("⚠️ Lancez etape9_genai.py pour générer le rapport narratif.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<hr style="border-color:{BO}; margin:40px 0 10px;">
<div style="text-align:center; color:{T2}; font-size:12px; padding:8px 0;">
  HR Turnover Analytics &nbsp;·&nbsp;
  Seye Kiné &nbsp;|&nbsp; Bindia Adeline Thiara &nbsp;·&nbsp;
  <span style="color:{O};">M. Aidara</span> &nbsp;·&nbsp; UCAO 2025-2026
</div>
""", unsafe_allow_html=True)
