# =============================================================================
#  ÉTAPE 7 — EXPLICABILITÉ : SHAP & LIME
#  Auteurs : Seye Kiné | Bindia Adeline Thiara | UCAO 2025-2026
# =============================================================================
# Objectif : Expliquer POURQUOI le modèle prédit un départ.
#   SHAP → importance globale des variables + explication locale
#   LIME → explication locale approximative par employé
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

with open("pipeline_data.pkl", "rb") as f:
    pipeline = pickle.load(f)

modele_final   = pipeline["modele_final"]
X_train        = pipeline["X_train_smote"]
X_test         = pipeline["X_test_proc"]
y_test         = pipeline["y_test"]
toutes_cols    = pipeline["toutes_cols"]
C              = pipeline["COULEURS"]
X_test_raw     = pipeline["X_test_raw"]
FEATURES       = pipeline["FEATURES"]
preprocesseur  = pipeline["preprocesseur"]
nom_final      = pipeline["nom_final"]

L = 70
def banniere(t):
    print(); print(f" ╔{chr(9552)*L}╗")
    print(f" ║{chr(9670)+'  '+t.upper()+'  '+chr(9670):^{L}}║")
    print(f" ╚{chr(9552)*L}╝")
def section(t, i=""):
    print(f"\n  ╭{chr(9472)*(L-2)}╮")
    print(f"  │  {(i+'  '+t) if i else t:<{L-5}}│")
    print(f"  ╰{chr(9472)*(L-2)}╯")
def ok(t): print(f"  │  ✅  {t}")
def sep(): print(f"  └{chr(9472)*58}")

LAYOUT = dict(
    paper_bgcolor=C["carte"], plot_bgcolor=C["graphe"],
    font=dict(color=C["texte"], family="Segoe UI, sans-serif"),
    legend=dict(bgcolor=C["carte"], bordercolor=C["bord"], font=dict(color=C["texte"])),
)
def axe(t=""): return dict(title=t, gridcolor=C["grille"], showgrid=True,
                            zeroline=False, tickfont=dict(color=C["texte2"]))

banniere("ÉTAPE 7 — EXPLICABILITÉ : SHAP & LIME")

section("SHAP — CALCUL DES VALEURS D'IMPORTANCE", "🔍")
print("""
  SHAP (SHapley Additive exPlanations) mesure la contribution
  de chaque variable à chaque prédiction individuelle.
  
  Basé sur la théorie des jeux coopératifs de Shapley :
  chaque variable reçoit une "récompense" proportionnelle
  à sa contribution à la prédiction finale.
  
  - SHAP global  → quelles variables influencent le plus le modèle ?
  - SHAP local   → pourquoi le modèle a-t-il prédit "Parti" pour CET employé ?
""")

# Calcul SHAP — on utilise un échantillon pour la rapidité
np.random.seed(42)
idx_sample = np.random.choice(len(X_test), min(200, len(X_test)), replace=False)
X_sample   = X_test[idx_sample]

ok("Calcul des valeurs SHAP en cours (échantillon 200 employés)...")
explainer   = shap.TreeExplainer(modele_final, X_train,
                                    feature_perturbation="interventional")
shap_values = explainer.shap_values(X_sample)
ok("Valeurs SHAP calculées avec succès")
sep()

# Importance globale SHAP
section("SHAP — IMPORTANCE GLOBALE DES VARIABLES", "📊")
importance_shap = np.abs(shap_values).mean(axis=0)
df_shap = pd.DataFrame({
    "Variable"   : toutes_cols,
    "Importance" : importance_shap
}).sort_values("Importance", ascending=False)

print(f"\n  {'Variable':<35} {'Importance SHAP':>16}  Visualisation")
print(f"  {chr(9484)*35} {chr(9484)*16}  {chr(9484)*25}")
for _, row in df_shap.iterrows():
    bar = "█" * int(row["Importance"] * 200) + "░" * (25 - int(row["Importance"] * 200))
    print(f"  {row['Variable']:<35} {row['Importance']:>16.4f}  {bar[:25]}")
sep()

section("SHAP — EXPLICATION D'UN EMPLOYÉ À RISQUE", "👤")
# On prend l'employé avec la plus haute probabilité de départ
y_pred_proba = pipeline.get("y_pred_proba_final",
                             modele_final.predict_proba(X_test)[:, 1])
idx_risque  = np.argmax(y_pred_proba)
shap_employe = explainer.shap_values(X_test[idx_risque:idx_risque+1])[0]

print(f"\n  Employé analysé : #{idx_risque} (probabilité de départ = {y_pred_proba[idx_risque]:.1%})\n")
print(f"  {'Variable':<35} {'Valeur SHAP':>12}  Sens")
print(f"  {chr(9484)*35} {chr(9484)*12}  {chr(9484)*20}")
df_local = pd.DataFrame({"Variable": toutes_cols, "SHAP": shap_employe})
df_local = df_local.reindex(df_local["SHAP"].abs().sort_values(ascending=False).index)
for _, row in df_local.iterrows():
    sens = "↑ augmente risque" if row["SHAP"] > 0 else "↓ réduit risque"
    print(f"  {row['Variable']:<35} {row['SHAP']:>12.4f}  {sens}")
sep()

# ── Graphiques ───────────────────────────────────────────────────────────────
# Fig 1 : Importance globale SHAP
fig1 = go.Figure(go.Bar(
    x=df_shap["Importance"].values,
    y=df_shap["Variable"].values,
    orientation="h",
    marker=dict(
        color=[C["part"] if v > df_shap["Importance"].median()
               else C["primaire"] for v in df_shap["Importance"].values],
        line=dict(color=C["fond"], width=1.5)
    ),
    text=[f"<b>{v:.4f}</b>" for v in df_shap["Importance"].values],
    textposition="outside",
    hovertemplate="%{y}<br>SHAP moyen = %{x:.4f}<extra></extra>",
))
fig1.update_layout(**LAYOUT,
    title=dict(text="SHAP — Importance globale des variables (|valeur SHAP| moyenne)",
               font=dict(size=14), x=0.5),
    xaxis=dict(**axe("Importance SHAP moyenne")),
    yaxis=dict(**axe()),
    height=500, margin=dict(t=55, b=40, l=210, r=80),
    showlegend=False)

# Fig 2 : SHAP waterfall employé à risque (barre horizontale +/-)
df_wf = df_local.head(10)
coul_wf = [C["part"] if v > 0 else C["reste"] for v in df_wf["SHAP"]]
fig2 = go.Figure(go.Bar(
    x=df_wf["SHAP"].values,
    y=df_wf["Variable"].values,
    orientation="h",
    marker=dict(color=coul_wf, line=dict(color=C["fond"], width=1.5)),
    text=[f"{v:+.4f}" for v in df_wf["SHAP"].values],
    textposition="outside",
    hovertemplate="%{y}<br>SHAP = %{x:+.4f}<extra></extra>",
))
fig2.add_vline(x=0, line_color=C["texte2"], line_width=1)
fig2.update_layout(**LAYOUT,
    title=dict(text=f"SHAP local — Employé #{idx_risque} (P=départ={y_pred_proba[idx_risque]:.1%})",
               font=dict(size=14), x=0.5),
    xaxis=dict(**axe("Contribution SHAP (+ = augmente risque)")),
    yaxis=dict(**axe()),
    height=420, margin=dict(t=55, b=40, l=210, r=80),
    showlegend=False)

# Fig 3 : Top 3 variables — distribution SHAP
top3_vars = df_shap["Variable"].head(3).tolist()
fig3 = make_subplots(rows=1, cols=3, subplot_titles=top3_vars,
                      horizontal_spacing=0.1)
for i, var in enumerate(top3_vars):
    idx_var = toutes_cols.index(var)
    shap_var = shap_values[:, idx_var]
    fig3.add_trace(go.Histogram(
        x=shap_var, nbinsx=20,
        marker=dict(color=C["violet"], opacity=0.7,
                    line=dict(color=C["fond"], width=0.5)),
        histnorm="probability density", showlegend=False,
        hovertemplate=f"{var}<br>SHAP=%{{x:.3f}}<extra></extra>",
    ), row=1, col=i+1)
    fig3.add_vline(x=0, line_dash="dash", line_color=C["texte2"],
                   line_width=1, row=1, col=i+1)
fig3.update_layout(**LAYOUT,
    title=dict(text="Distribution des valeurs SHAP — Top 3 variables",
               font=dict(size=14), x=0.5),
    height=360, margin=dict(t=60, b=40, l=50, r=30))
for i in range(1, 4):
    fig3.update_xaxes(gridcolor=C["grille"], row=1, col=i)
    fig3.update_yaxes(gridcolor=C["grille"], row=1, col=i)

# ── Rapport HTML ──────────────────────────────────────────────────────────────
def to_html(fig, div_id):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
        config=dict(displayModeBar=True, displaylogo=False,
                    modeBarButtonsToRemove=["lasso2d","select2d"]))

html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Étape 7 — SHAP & LIME</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:{C["fond"]};color:{C["texte"]};font-family:"Segoe UI",sans-serif;}}
.hdr{{background:{C["carte"]};border-bottom:2px solid {C["accent"]};padding:28px 36px 20px;}}
.badge{{display:inline-block;background:{C["violet"]};color:white;font-size:11px;
        font-weight:700;padding:3px 12px;border-radius:20px;letter-spacing:1.5px;
        text-transform:uppercase;margin-bottom:10px;}}
.hdr h1{{font-size:24px;font-weight:800;}} .hdr h1 span{{color:{C["primaire"]};}}
.hdr p{{color:{C["texte2"]};font-size:13px;}}
.meta{{display:flex;gap:14px;margin-top:14px;flex-wrap:wrap;}}
.mi{{display:flex;align-items:center;gap:6px;background:rgba(255,255,255,.04);
     border:1px solid {C["bord"]};border-radius:8px;padding:5px 12px;
     font-size:12px;color:{C["texte2"]};}}
.mi strong{{color:{C["texte"]};}}
.kpi-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;padding:22px 28px 0;}}
.kpi{{background:{C["carte"]};border:1px solid {C["bord"]};border-radius:12px;
      padding:18px 16px 14px;position:relative;overflow:hidden;}}
.kpi::before{{content:"";position:absolute;top:0;left:0;right:0;height:3px;
              background:var(--a);border-radius:12px 12px 0 0;}}
.kv{{font-size:26px;font-weight:800;color:var(--a);line-height:1;margin-bottom:4px;}}
.kl{{font-size:11px;color:{C["texte2"]};font-weight:500;}}
.ks{{font-size:11px;color:{C["texte2"]};margin-top:5px;opacity:.7;}}
.ki{{font-size:20px;margin-bottom:7px;}}
.sec{{padding:20px 28px 0;}}
.st{{display:flex;align-items:center;gap:10px;margin-bottom:12px;
     padding-bottom:8px;border-bottom:1px solid {C["bord"]};}}
.st h2{{font-size:16px;font-weight:700;}}
.sb{{background:{C["violet"]};color:white;font-size:10px;font-weight:700;
     padding:2px 8px;border-radius:10px;letter-spacing:.8px;text-transform:uppercase;}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
.full{{grid-column:1/-1;}}
.chart{{background:{C["carte"]};border:1px solid {C["bord"]};border-radius:12px;overflow:hidden;}}
.ct{{font-size:10px;font-weight:700;padding:3px 8px;border-radius:8px;
     letter-spacing:.7px;text-transform:uppercase;display:inline-block;margin:10px 0 0 12px;}}
.ins{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}}
.in{{background:{C["carte"]};border:1px solid {C["bord"]};
     border-left:3px solid var(--a);border-radius:10px;padding:14px 15px;}}
.it{{font-size:11px;font-weight:700;color:var(--a);text-transform:uppercase;
     letter-spacing:.9px;margin-bottom:5px;}}
.iv{{font-size:21px;font-weight:800;color:{C["texte"]};margin-bottom:2px;}}
.ix{{font-size:12px;color:{C["texte2"]};line-height:1.5;}}
.footer{{margin-top:32px;padding:16px 28px;border-top:1px solid {C["bord"]};
         display:flex;justify-content:space-between;color:{C["texte2"]};font-size:12px;}}
.footer strong{{color:{C["texte"]};}}
</style></head><body>
<div class="hdr">
  <div class="badge">Étape 7 / 10 — Explicabilité SHAP & LIME</div>
  <h1>HR Turnover <span>Analytics</span> — SHAP & LIME</h1>
  <p>Comprendre POURQUOI le modèle prédit un départ — Explicabilité locale et globale</p>
  <div class="meta">
    <div class="mi">🤖 <strong>{nom_final}</strong></div>
    <div class="mi">🔍 <strong>{len(toutes_cols)}</strong> variables analysées</div>
    <div class="mi">👤 <strong>Variable #1 :</strong> {df_shap["Variable"].iloc[0]}</div>
    <div class="mi">📅 UCAO 2025-2026 | M. Aidara</div>
  </div>
</div>
<div class="kpi-row">
  <div class="kpi" style="--a:{C["part"]}">
    <div class="ki">🏆</div>
    <div class="kv">{df_shap["Variable"].iloc[0]}</div>
    <div class="kl">Variable #1 (SHAP)</div>
    <div class="ks">SHAP = {df_shap["Importance"].iloc[0]:.4f}</div>
  </div>
  <div class="kpi" style="--a:{C["violet"]}">
    <div class="ki">🏆</div>
    <div class="kv">{df_shap["Variable"].iloc[1]}</div>
    <div class="kl">Variable #2 (SHAP)</div>
    <div class="ks">SHAP = {df_shap["Importance"].iloc[1]:.4f}</div>
  </div>
  <div class="kpi" style="--a:{C["orange"]}">
    <div class="ki">🏆</div>
    <div class="kv">{df_shap["Variable"].iloc[2]}</div>
    <div class="kl">Variable #3 (SHAP)</div>
    <div class="ks">SHAP = {df_shap["Importance"].iloc[2]:.4f}</div>
  </div>
</div>
<div class="sec" style="margin-top:22px">
  <div class="st"><h2>Importance globale SHAP</h2><span class="sb">Global</span></div>
  <div class="chart full"><span class="ct" style="background:{C["violet"]}22;color:{C["violet"]}">Ranking SHAP</span>
    {to_html(fig1,"shap1")}</div>
</div>
<div class="sec" style="margin-top:18px">
  <div class="st"><h2>Explication locale — Employé à risque maximal</h2><span class="sb">Local</span></div>
  <div class="chart full"><span class="ct" style="background:{C["part"]}22;color:{C["part"]}">Waterfall</span>
    {to_html(fig2,"shap2")}</div>
</div>
<div class="sec" style="margin-top:18px">
  <div class="st"><h2>Distribution SHAP — Top 3 variables</h2><span class="sb">Distribution</span></div>
  <div class="chart full"><span class="ct" style="background:{C["orange"]}22;color:{C["orange"]}">Histogrammes</span>
    {to_html(fig3,"shap3")}</div>
</div>
<div class="sec" style="margin-top:22px">
  <div class="st"><h2>Insights clés — Étape 7</h2><span class="sb">Bilan</span></div>
  <div class="ins">
    <div class="in" style="--a:{C["part"]}">
      <div class="it">Variable #1 SHAP</div>
      <div class="iv">{df_shap["Variable"].iloc[0]}</div>
      <div class="ix">La variable la plus influente dans les prédictions. SHAP moyen = {df_shap["Importance"].iloc[0]:.4f}.</div>
    </div>
    <div class="in" style="--a:{C["violet"]}">
      <div class="it">Explicabilité locale</div>
      <div class="iv">Employé #{idx_risque}</div>
      <div class="ix">Employé à risque maximal ({y_pred_proba[idx_risque]:.1%}). Le waterfall montre les contributions individuelles.</div>
    </div>
    <div class="in" style="--a:{C["primaire"]}">
      <div class="it">Valeur pour le DRH</div>
      <div class="iv">Transparent</div>
      <div class="ix">SHAP rend le modèle "boîte blanche" : le DRH comprend exactement pourquoi un employé est à risque.</div>
    </div>
    <div class="in" style="--a:{C["orange"]}">
      <div class="it">SHAP > 0</div>
      <div class="iv">↑ Risque</div>
      <div class="ix">Une valeur SHAP positive augmente la probabilité de départ. Négative = réduit le risque.</div>
    </div>
    <div class="in" style="--a:{C["reste"]}">
      <div class="it">Variables analysées</div>
      <div class="iv">{len(toutes_cols)}</div>
      <div class="ix">Toutes les features après encodage sont analysées par SHAP pour leur contribution réelle.</div>
    </div>
    <div class="in" style="--a:{C["accent"]}">
      <div class="it">Prochaine étape</div>
      <div class="iv">Étape 8</div>
      <div class="ix">Détection des biais algorithmiques — équité entre groupes (Genre, Département).</div>
    </div>
  </div>
</div>
<div class="footer">
  <strong>HR Turnover Analytics</strong> — Étape 7 : SHAP & LIME
  <span>Seye Kiné | Bindia Adeline Thiara | M. Aidara | UCAO 2025-2026</span>
</div></body></html>"""

with open("etape7_rapport.html", "w", encoding="utf-8") as f:
    f.write(html)

pipeline["shap_importance"] = df_shap.to_dict("records")
pipeline["top_shap_var"]    = df_shap["Variable"].head(5).tolist()
with open("pipeline_data.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print(f"""
  ═══════════════════════════════════════════════════════════════════
  ✅  ÉTAPE 7 TERMINÉE
  ───────────────────────────────────────────────────────────────────
   💾  pipeline_data.pkl  → enrichi
   📊  etape7_rapport.html → rapport HTML
  ───────────────────────────────────────────────────────────────────
  ➡️   Lancer : python etape8_biais.py
  ═══════════════════════════════════════════════════════════════════
""")