# =============================================================================
#  GENERATIVE HR ANALYTICS — PRÉDICTION DU TURNOVER DES EMPLOYÉS
#  Auteurs   : Seye Kiné | Bindia Adeline Thiara
#  Encadrant : M. Aidara
#  Institut  : UCAO — Département Informatique de Gestion
#  Année     : 2025-2026
# =============================================================================
#  ÉTAPE 4 — ENTRAÎNEMENT ET COMPARAISON DES MODÈLES ML
#
#  Objectif  : Entraîner plusieurs modèles de Machine Learning,
#              comparer leurs performances et sélectionner le
#              meilleur pour la prédiction du turnover.
#
#  Modèles testés :
#    1. Régression Logistique  — modèle de référence (baseline)
#    2. Random Forest          — ensemble de base
#    3. XGBoost                — meilleur modèle attendu
#
#  Métriques d'évaluation :
#    - Accuracy   : % de prédictions correctes
#    - Précision  : parmi les "Parti" prédits, combien sont vrais ?
#    - Rappel     : parmi les vrais "Parti", combien détectés ?
#    - F1-Score   : équilibre Précision/Rappel (métrique principale)
#    - AUC-ROC    : capacité à discriminer les deux classes
#
#  Entrée    : pipeline_data.pkl  (produit par l'Étape 3)
#  Sorties   : pipeline_data.pkl  (enrichi + meilleur modèle)
#              etape4_rapport.html
#
#  Sections  :
#    A. Chargement du pipeline
#    B. Fonctions console
#    C. Définition des modèles
#    D. Entraînement et évaluation
#    E. Comparaison et sélection du meilleur modèle
#    F. Bilan + enrichissement pipeline
#    G. Rapport HTML interactif
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import time

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     roc_auc_score, confusion_matrix,
                                     classification_report,
                                     roc_curve)
from sklearn.model_selection import cross_val_score
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier

import plotly.graph_objects as go
from plotly.subplots        import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION A — CHARGEMENT DU PIPELINE
# =============================================================================

with open('pipeline_data.pkl', 'rb') as f:
    pipeline = pickle.load(f)

X_train      = pipeline['X_train_smote']   # Train équilibré (SMOTE)
X_test       = pipeline['X_test_proc']     # Test réel
y_train      = pipeline['y_train_smote']   # Labels train
y_test       = pipeline['y_test']          # Labels test
toutes_cols  = pipeline['toutes_cols']     # Noms des features
COULEURS     = pipeline['COULEURS']
taux         = pipeline['taux']
CIBLE        = pipeline['CIBLE']

C = COULEURS   # Alias court

LAYOUT_BASE = dict(
    paper_bgcolor = C['carte'],
    plot_bgcolor  = C['graphe'],
    font          = dict(color=C['texte'], family='Segoe UI, sans-serif'),
    legend        = dict(bgcolor=C['carte'], bordercolor=C['bord'],
                         font=dict(color=C['texte'])),
)

def axe(titre=''):
    return dict(title=titre, gridcolor=C['grille'], showgrid=True,
                zeroline=False, tickfont=dict(color=C['texte2']))

# =============================================================================
# SECTION B — FONCTIONS CONSOLE
# =============================================================================

L = 70

def banniere(texte):
    print()
    print(f" ╔{'═'*L}╗")
    print(f" ║{'  ◆  ' + texte.upper() + '  ◆':^{L}}║")
    print(f" ╚{'═'*L}╝")

def section(texte, icone=''):
    print(f"\n  ╭{'─'*(L-2)}╮")
    txt = f"{icone}  {texte}" if icone else texte
    print(f"  │  {txt:<{L-5}}│")
    print(f"  ╰{'─'*(L-2)}╯")

def sous_section(texte, icone='▸'):
    print(f"\n  {icone}  {texte}")
    print(f"  {'╌'*55}")

def sep():
    print(f"  └{'─'*58}")

def ok(texte):   print(f"  │  ✅  {texte}")
def info(texte): print(f"  │  ℹ️   {texte}")

def bilan(items, titre='BILAN'):
    print(f"\n  ╔══ {titre} {'═'*(L-len(titre)-5)}╗")
    for k, v in items.items():
        s = f"  ║  {k:<32} {v}"
        print(f"{s:<{L+4}}║")
    print(f"  ╚{'═'*(L+1)}╝")

# =============================================================================
# SECTION C — DÉFINITION DES MODÈLES
# =============================================================================

banniere("ÉTAPE 4 — ENTRAÎNEMENT ET COMPARAISON DES MODÈLES ML")

print(f"""
  Données reçues de l'Étape 3 :
  ─────────────────────────────────────────────────────
  Train : {X_train.shape[0]:,} exemples × {X_train.shape[1]} features (après SMOTE)
  Test  : {X_test.shape[0]:,}  exemples × {X_test.shape[1]} features (données réelles)
  ─────────────────────────────────────────────────────
  Objectif : Trouver le modèle le plus performant pour
             prédire le turnover des employés.
""")

section("DÉFINITION DES MODÈLES", "🤖")

print("""
  4 modèles sont comparés, du plus simple au plus complexe :

  ① Régression Logistique — Baseline (référence)
     Modèle linéaire probabiliste. Simple et interprétable.
     Sert de référence : si XGBoost ne fait pas mieux,
     la complexité n'est pas justifiée.

  ② Random Forest — Ensemble de base
     Forêt de 300 arbres de décision entraînés en parallèle.
     Robuste, résistant au bruit, peu sensible aux outliers.
     Bon équilibre performance / interprétabilité.

  ③ XGBoost — Modèle avancé
  ④ LightGBM — Alternative rapide à XGBoost
     Gradient Boosting avec algorithme leaf-wise (croissance
     par feuilles). Plus rapide et précis que XGBoost sur les
     grands datasets. Standard des compétitions data science.

     Arbres construits séquentiellement (chaque arbre corrige
     les erreurs du précédent). Standard de l'industrie pour
     les données tabulaires. Compatible avec SHAP (Étape 7).
""")

# Définition des 3 modèles avec leurs hyperparamètres de départ
# (ils seront optimisés à l'Étape 5)
MODELES = {

    'Régression Logistique': LogisticRegression(
        max_iter     = 1000,   # Nombre d'itérations max
        random_state = 42,     # Reproductibilité
        C            = 1.0,    # Force de régularisation (défaut)
        solver       = 'lbfgs' # Algorithme d'optimisation
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators = 300,    # 300 arbres dans la forêt
        max_depth    = None,   # Pas de limite de profondeur
        random_state = 42,
        n_jobs       = -1,     # Utilise tous les cœurs CPU
        class_weight = 'balanced'  # Gère le déséquilibre résiduel
    ),

    'XGBoost': XGBClassifier(
        n_estimators     = 300,    # 300 arbres
        learning_rate    = 0.1,    # Vitesse d'apprentissage
        max_depth        = 6,      # Profondeur max de chaque arbre
        random_state     = 42,
        eval_metric      = 'logloss',
        verbosity        = 0,      # Pas de messages parasites
        use_label_encoder= False
    ),
    'LightGBM': LGBMClassifier(
        n_estimators  = 300,    # 300 arbres (boosting)
        learning_rate = 0.1,    # Taux d apprentissage
        max_depth     = 6,      # Profondeur max des arbres
        random_state  = 42,
        n_jobs        = -1,     # Utilise tous les coeurs CPU
        verbosity     = -1,     # Aucun message parasite
        class_weight  = 'balanced'
    ),
}

ok(f"{len(MODELES)} modèles définis et prêts à l'entraînement")
sep()

# =============================================================================
# SECTION D — ENTRAÎNEMENT ET ÉVALUATION
# =============================================================================

section("ENTRAÎNEMENT ET ÉVALUATION", "🏋️")

print(f"\n  {'Modèle':<26} {'Accuracy':>9} {'Précision':>10} "
      f"{'Rappel':>8} {'F1-Score':>9} {'AUC-ROC':>8}  {'Temps':>7}")
print(f"  {'╌'*26} {'╌'*9} {'╌'*10} {'╌'*8} {'╌'*9} {'╌'*8}  {'╌'*7}")

resultats = {}   # Stockage de tous les résultats

for nom, modele in MODELES.items():

    # ── Entraînement ──────────────────────────────────────────────────────────
    debut = time.time()
    modele.fit(X_train, y_train)
    duree = time.time() - debut

    # ── Prédictions ───────────────────────────────────────────────────────────
    y_pred      = modele.predict(X_test)
    y_pred_proba= modele.predict_proba(X_test)[:, 1]  # Probabilité de partir

    # ── Métriques ─────────────────────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_pred_proba)
    cm   = confusion_matrix(y_test, y_pred)

    # ── Validation croisée (5-fold) ───────────────────────────────────────────
    # La validation croisée évalue le modèle sur 5 sous-ensembles différents
    # du train set → donne une estimation plus robuste de la performance
    cv_scores = cross_val_score(modele, X_train, y_train,
                                cv=5, scoring='f1', n_jobs=-1)

    # ── Courbe ROC ────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    resultats[nom] = {
        'modele'     : modele,
        'acc'        : acc,
        'prec'       : prec,
        'rec'        : rec,
        'f1'         : f1,
        'auc'        : auc,
        'cm'         : cm,
        'cv_mean'    : cv_scores.mean(),
        'cv_std'     : cv_scores.std(),
        'y_pred'     : y_pred,
        'y_pred_proba': y_pred_proba,
        'fpr'        : fpr,
        'tpr'        : tpr,
        'duree'      : duree,
    }

    print(f"  {nom:<26} {acc:>9.4f} {prec:>10.4f} "
          f"{rec:>8.4f} {f1:>9.4f} {auc:>8.4f}  {duree:>6.1f}s")

sep()

# ── Rapport de classification détaillé pour chaque modèle ────────────────────
for nom, res in resultats.items():
    sous_section(f"Rapport détaillé — {nom}", "📋")
    print(classification_report(y_test, res['y_pred'],
                                 target_names=['Resté (0)', 'Parti (1)']))
    print(f"  Cross-validation F1 (5-fold) : "
          f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
    sep()

# =============================================================================
# SECTION E — COMPARAISON ET SÉLECTION DU MEILLEUR MODÈLE
# =============================================================================

section("COMPARAISON ET SÉLECTION DU MEILLEUR MODÈLE", "🏆")

# On classe les modèles par F1-Score (métrique principale)
# Le F1-Score est le plus adapté car il équilibre précision et rappel
classement = sorted(resultats.items(), key=lambda x: x[1]['f1'], reverse=True)

print(f"\n  Classement par F1-Score (décroissant) :\n")
print(f"  {'Rang':<6} {'Modèle':<28} {'F1-Score':>9} {'AUC-ROC':>9} "
      f"{'CV F1 moy.':>11}  Verdict")
print(f"  {'╌'*6} {'╌'*28} {'╌'*9} {'╌'*9} {'╌'*11}  {'╌'*15}")

for i, (nom, res) in enumerate(classement):
    rang     = ['🥇', '🥈', '🥉', '4️⃣'][i]
    verdict  = '← MEILLEUR MODÈLE' if i == 0 else ''
    print(f"  {rang:<6} {nom:<28} {res['f1']:>9.4f} {res['auc']:>9.4f} "
          f"{res['cv_mean']:>10.4f}±{res['cv_std']:.3f}  {verdict}")

# Le meilleur modèle
nom_meilleur = classement[0][0]
meilleur     = classement[0][1]

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  MEILLEUR MODÈLE : {nom_meilleur:<42}│
  │                                                              │
  │  F1-Score   : {meilleur['f1']:.4f}                                    │
  │  AUC-ROC    : {meilleur['auc']:.4f}                                    │
  │  Accuracy   : {meilleur['acc']:.4f}                                    │
  │  Précision  : {meilleur['prec']:.4f}                                    │
  │  Rappel     : {meilleur['rec']:.4f}                                    │
  │  CV F1 moy. : {meilleur['cv_mean']:.4f} ± {meilleur['cv_std']:.4f}                         │
  └──────────────────────────────────────────────────────────────┘
""")

# ── Matrice de confusion du meilleur modèle ───────────────────────────────────
cm = meilleur['cm']
tn, fp, fn, tp = cm.ravel()

print(f"  Matrice de confusion ({nom_meilleur}) :")
print(f"  ┌─────────────────────────────────────────┐")
print(f"  │              Prédit Resté  Prédit Parti  │")
print(f"  │  Réel Resté  {tn:>10,}  {fp:>11,}  │")
print(f"  │  Réel Parti  {fn:>10,}  {tp:>11,}  │")
print(f"  └─────────────────────────────────────────┘")
print(f"\n  Interprétation :")
print(f"  ✅  Vrais Négatifs (TN) = {tn:,}  : restés correctement prédits")
print(f"  ✅  Vrais Positifs (TP) = {tp:,}   : partis correctement détectés")
print(f"  ⚠️   Faux Positifs  (FP) = {fp:,}   : alertes inutiles (restés → prédits partis)")
print(f"  ⚠️   Faux Négatifs  (FN) = {fn:,}   : partis manqués (partis → prédits restés)")
sep()

# =============================================================================
# SECTION F — BILAN + ENRICHISSEMENT PIPELINE
# =============================================================================

bilan({
    '🤖 Modèles entraînés'     : "Log. Reg., Random Forest, XGBoost, LightGBM",
    f'🥇 Meilleur modèle'      : nom_meilleur,
    '📊 F1-Score meilleur'     : f"{meilleur['f1']:.4f}",
    '📊 AUC-ROC meilleur'      : f"{meilleur['auc']:.4f}",
    '📊 CV F1 (5-fold)'        : f"{meilleur['cv_mean']:.4f} ± {meilleur['cv_std']:.4f}",
    '✅ Vrais Positifs (TP)'   : f"{tp:,} partis correctement détectés",
    '⚠️  Faux Négatifs (FN)'   : f"{fn:,} partis manqués",
    '➡️  Prochaine étape'       : "Étape 5 — Optimisation des hyperparamètres",
}, titre='BILAN — ÉTAPE 4')

# Enrichissement du pipeline
pipeline['resultats_modeles'] = {
    nom: {k: v for k, v in res.items() if k != 'modele'}
    for nom, res in resultats.items()
}
pipeline['modeles']       = {nom: res['modele'] for nom, res in resultats.items()}
pipeline['nom_meilleur']  = nom_meilleur
pipeline['meilleur_modele'] = meilleur['modele']
pipeline['classement']    = [(n, r['f1'], r['auc']) for n, r in classement]

with open('pipeline_data.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("\n  💾  pipeline_data.pkl enrichi → Étape 5")

# =============================================================================
# SECTION G — RAPPORT HTML INTERACTIF
# =============================================================================

print("\n  ⏳  Génération du rapport HTML interactif...")

noms_modeles = list(resultats.keys())
noms_courts  = ['Log. Reg.', 'Rnd. Forest', 'XGBoost', 'LightGBM']

# ── Fig 1 : Comparaison des métriques (barres groupées) ───────────────────────
metriques = ['acc', 'prec', 'rec', 'f1', 'auc']
labels_m  = ['Accuracy', 'Précision', 'Rappel', 'F1-Score', 'AUC-ROC']
coul_mod  = [C['primaire'], C['violet'], C['part'], C['orange']]

fig_comp = go.Figure()
for i, (nom, nom_c) in enumerate(zip(noms_modeles, noms_courts)):
    vals = [resultats[nom][m] for m in metriques]
    fig_comp.add_trace(go.Bar(
        name=nom_c, x=labels_m, y=vals,
        marker=dict(color=coul_mod[i], line=dict(color=C['fond'], width=1.5)),
        text=[f"{v:.3f}" for v in vals],
        textposition='outside', textfont=dict(size=9),
        hovertemplate=f'{nom}<br>%{{x}} : %{{y:.4f}}<extra></extra>',
    ))
fig_comp.update_layout(**LAYOUT_BASE,
    title=dict(text='Comparaison des métriques — 3 modèles',
               font=dict(size=14), x=0.5),
    yaxis=dict(**axe('Score'), range=[0, 1.12]),
    xaxis=axe('Métriques'),
    height=380, margin=dict(t=55, b=40, l=55, r=30),
    barmode='group')

# ── Fig 2 : Courbes ROC ───────────────────────────────────────────────────────
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=[0,1], y=[0,1], mode='lines',
    line=dict(color=C['texte2'], dash='dash', width=1),
    name='Aléatoire (AUC=0.5)', showlegend=True,
    hoverinfo='skip',
))
for i, (nom, nom_c) in enumerate(zip(noms_modeles, noms_courts)):
    res = resultats[nom]
    fig_roc.add_trace(go.Scatter(
        x=res['fpr'], y=res['tpr'], mode='lines',
        name=f"{nom_c} (AUC={res['auc']:.4f})",
        line=dict(color=coul_mod[i], width=2.5),
        hovertemplate=f'FPR=%{{x:.3f}}<br>TPR=%{{y:.3f}}<extra>{nom_c}</extra>',
    ))
fig_roc.update_layout(**LAYOUT_BASE,
    title=dict(text='Courbes ROC — Comparaison des 3 modèles',
               font=dict(size=14), x=0.5),
    xaxis=dict(**axe('Taux de faux positifs (FPR)'), range=[0,1]),
    yaxis=dict(**axe('Taux de vrais positifs (TPR)'), range=[0,1.02]),
    height=420, margin=dict(t=55, b=50, l=60, r=30))

# ── Fig 3 : Matrice de confusion du meilleur modèle ───────────────────────────
cm_vals  = meilleur['cm']
fig_cm   = go.Figure(go.Heatmap(
    z=cm_vals,
    x=['Prédit Resté (0)', 'Prédit Parti (1)'],
    y=['Réel Resté (0)', 'Réel Parti (1)'],
    text=[[f"<b>{v:,}</b>" for v in row] for row in cm_vals],
    texttemplate='%{text}',
    textfont=dict(size=22, color='white'),
    colorscale=[[0, C['graphe']], [0.5, C['primaire']], [1, C['reste']]],
    showscale=False,
    hovertemplate='%{y} → %{x}<br>Nombre : %{z:,}<extra></extra>',
))
# Annotations
labels_cm = [
    (0, 0, f"TN\n{cm_vals[0,0]:,}", C['reste']),
    (1, 0, f"FP\n{cm_vals[0,1]:,}", C['part']),
    (0, 1, f"FN\n{cm_vals[1,0]:,}", C['part']),
    (1, 1, f"TP\n{cm_vals[1,1]:,}", C['reste']),
]
fig_cm.update_layout(**LAYOUT_BASE,
    title=dict(text=f'Matrice de Confusion — {nom_meilleur}',
               font=dict(size=14), x=0.5),
    xaxis=dict(tickfont=dict(size=11, color=C['texte2'])),
    yaxis=dict(tickfont=dict(size=11, color=C['texte2'])),
    height=380, margin=dict(t=55, b=50, l=120, r=30))

# ── Fig 4 : CV F1 scores (barres d'erreur) ────────────────────────────────────
fig_cv = go.Figure()
for i, (nom, nom_c) in enumerate(zip(noms_modeles, noms_courts)):
    res = resultats[nom]
    fig_cv.add_trace(go.Bar(
        x=[nom_c], y=[res['cv_mean']],
        error_y=dict(type='data', array=[res['cv_std']*2],
                     color=C['texte2'], thickness=2, width=8),
        marker=dict(color=coul_mod[i], line=dict(color=C['fond'], width=2)),
        text=f"<b>{res['cv_mean']:.4f}</b>",
        textposition='outside',
        name=nom_c, showlegend=False,
        hovertemplate=f'{nom}<br>CV F1 = {res["cv_mean"]:.4f} ± {res["cv_std"]:.4f}<extra></extra>',
    ))
fig_cv.update_layout(**LAYOUT_BASE,
    title=dict(text='Validation croisée F1 (5-fold) — Moyenne ± 2×Écart-type',
               font=dict(size=14), x=0.5),
    yaxis=dict(**axe('F1-Score moyen'), range=[0, 1.1]),
    xaxis=axe(), height=340,
    margin=dict(t=55, b=40, l=55, r=30))

# ── Construction HTML ──────────────────────────────────────────────────────────
def to_html(fig, div_id):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
        config=dict(displayModeBar=True, displaylogo=False,
                    modeBarButtonsToRemove=['lasso2d','select2d']))

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Étape 4 — Modèles ML</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:{C['fond']};color:{C['texte']};font-family:'Segoe UI',sans-serif;}}
.hdr{{background:{C['carte']};border-bottom:2px solid {C['accent']};padding:28px 36px 20px;}}
.badge{{display:inline-block;background:{C['violet']};color:white;font-size:11px;
        font-weight:700;padding:3px 12px;border-radius:20px;letter-spacing:1.5px;
        text-transform:uppercase;margin-bottom:10px;}}
.hdr h1{{font-size:24px;font-weight:800;margin-bottom:4px;}}
.hdr h1 span{{color:{C['primaire']};}}
.hdr p{{color:{C['texte2']};font-size:13px;}}
.meta{{display:flex;gap:14px;margin-top:14px;flex-wrap:wrap;}}
.mi{{display:flex;align-items:center;gap:6px;background:rgba(255,255,255,.04);
     border:1px solid {C['bord']};border-radius:8px;padding:5px 12px;
     font-size:12px;color:{C['texte2']};}}
.mi strong{{color:{C['texte']};}}
.kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;padding:22px 28px 0;}}
.kpi{{background:{C['carte']};border:1px solid {C['bord']};border-radius:12px;
      padding:18px 16px 14px;position:relative;overflow:hidden;transition:transform .2s;}}
.kpi:hover{{transform:translateY(-3px);}}
.kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;
              background:var(--a);border-radius:12px 12px 0 0;}}
.kv{{font-size:28px;font-weight:800;color:var(--a);line-height:1;margin-bottom:4px;}}
.kl{{font-size:11px;color:{C['texte2']};font-weight:500;}}
.ks{{font-size:11px;color:{C['texte2']};margin-top:5px;opacity:.7;}}
.ki{{font-size:20px;margin-bottom:7px;}}
.sec{{padding:20px 28px 0;}}
.st{{display:flex;align-items:center;gap:10px;margin-bottom:12px;
     padding-bottom:8px;border-bottom:1px solid {C['bord']};}}
.st h2{{font-size:16px;font-weight:700;}}
.sb{{background:{C['violet']};color:white;font-size:10px;font-weight:700;
     padding:2px 8px;border-radius:10px;letter-spacing:.8px;text-transform:uppercase;}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
.full{{grid-column:1/-1;}}
.chart{{background:{C['carte']};border:1px solid {C['bord']};border-radius:12px;
        overflow:hidden;transition:box-shadow .2s;}}
.chart:hover{{box-shadow:0 4px 20px rgba(0,0,0,.35);}}
.ct{{font-size:10px;font-weight:700;padding:3px 8px;border-radius:8px;
     letter-spacing:.7px;text-transform:uppercase;display:inline-block;margin:10px 0 0 12px;}}
.winner{{background:{C['carte']};border:2px solid {C['accent']};border-radius:12px;
         padding:20px 24px;margin-bottom:0;}}
.winner h3{{color:{C['accent']};font-size:15px;font-weight:700;margin-bottom:12px;}}
.w-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;}}
.w-item{{text-align:center;}}
.w-val{{font-size:22px;font-weight:800;color:{C['texte']};}}
.w-lbl{{font-size:11px;color:{C['texte2']};margin-top:3px;}}
.ins{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}}
.in{{background:{C['carte']};border:1px solid {C['bord']};
     border-left:3px solid var(--a);border-radius:10px;padding:14px 15px;}}
.it{{font-size:11px;font-weight:700;color:var(--a);text-transform:uppercase;
     letter-spacing:.9px;margin-bottom:5px;}}
.iv{{font-size:21px;font-weight:800;color:{C['texte']};margin-bottom:2px;}}
.ix{{font-size:12px;color:{C['texte2']};line-height:1.5;}}
.footer{{margin-top:32px;padding:16px 28px;border-top:1px solid {C['bord']};
         display:flex;justify-content:space-between;color:{C['texte2']};font-size:12px;}}
.footer strong{{color:{C['texte']};}}
.footer span{{opacity:.6;}}
@media(max-width:900px){{.kpi-row,.g2,.ins,.w-grid{{grid-template-columns:1fr 1fr;}}}}
</style>
</head>
<body>

<div class="hdr">
  <div class="badge">Étape 4 / 10 — Entraînement des modèles ML</div>
  <h1>HR Turnover <span>Analytics</span> — Comparaison des Modèles</h1>
  <p>Régression Logistique · Random Forest · XGBoost — Sélection du meilleur modèle</p>
  <div class="meta">
    <div class="mi">🤖 <strong>4</strong> modèles comparés</div>
    <div class="mi">🏆 <strong>{nom_meilleur}</strong> meilleur</div>
    <div class="mi">📊 <strong>F1 = {meilleur['f1']:.4f}</strong></div>
    <div class="mi">📊 <strong>AUC = {meilleur['auc']:.4f}</strong></div>
    <div class="mi">📅 UCAO 2025-2026 | M. Aidara</div>
  </div>
</div>

<div class="kpi-row">
  <div class="kpi" style="--a:{C['part']}">
    <div class="ki">🏆</div>
    <div class="kv">{nom_meilleur.split()[0]}</div>
    <div class="kl">Meilleur modèle</div>
    <div class="ks">Sélectionné par F1-Score</div>
  </div>
  <div class="kpi" style="--a:{C['violet']}">
    <div class="ki">📊</div>
    <div class="kv">{meilleur['f1']:.4f}</div>
    <div class="kl">F1-Score</div>
    <div class="ks">Métrique principale</div>
  </div>
  <div class="kpi" style="--a:{C['primaire']}">
    <div class="ki">📈</div>
    <div class="kv">{meilleur['auc']:.4f}</div>
    <div class="kl">AUC-ROC</div>
    <div class="ks">Pouvoir discriminant</div>
  </div>
  <div class="kpi" style="--a:{C['reste']}">
    <div class="ki">✅</div>
    <div class="kv">{tp:,}</div>
    <div class="kl">Partis détectés (TP)</div>
    <div class="ks">Sur {(y_test==1).sum():,} partis réels</div>
  </div>
</div>

<div class="sec" style="margin-top:22px">
  <div class="st"><h2>Meilleur modèle — Résumé complet</h2><span class="sb">🏆 Winner</span></div>
  <div class="winner">
    <h3>🏆 {nom_meilleur} — Métriques sur le Test Set</h3>
    <div class="w-grid">
      <div class="w-item"><div class="w-val">{meilleur['acc']:.4f}</div><div class="w-lbl">Accuracy</div></div>
      <div class="w-item"><div class="w-val">{meilleur['prec']:.4f}</div><div class="w-lbl">Précision</div></div>
      <div class="w-item"><div class="w-val">{meilleur['rec']:.4f}</div><div class="w-lbl">Rappel</div></div>
      <div class="w-item"><div class="w-val">{meilleur['f1']:.4f}</div><div class="w-lbl">F1-Score</div></div>
      <div class="w-item"><div class="w-val">{meilleur['auc']:.4f}</div><div class="w-lbl">AUC-ROC</div></div>
    </div>
  </div>
</div>

<div class="sec" style="margin-top:18px">
  <div class="st"><h2>Comparaison des 3 modèles</h2><span class="sb">Métriques</span></div>
  <div class="chart full">
    <span class="ct" style="background:{C['violet']}22;color:{C['violet']}">Toutes métriques</span>
    {to_html(fig_comp,'comp')}
  </div>
</div>

<div class="sec" style="margin-top:18px">
  <div class="st"><h2>Courbes ROC & Validation croisée</h2><span class="sb">Robustesse</span></div>
  <div class="g2">
    <div class="chart">
      <span class="ct" style="background:{C['primaire']}22;color:{C['primaire']}">ROC</span>
      {to_html(fig_roc,'roc')}
    </div>
    <div class="chart">
      <span class="ct" style="background:{C['orange']}22;color:{C['orange']}">Cross-Validation</span>
      {to_html(fig_cv,'cv')}
    </div>
  </div>
</div>

<div class="sec" style="margin-top:18px">
  <div class="st"><h2>Matrice de confusion — {nom_meilleur}</h2><span class="sb">Erreurs</span></div>
  <div class="chart full">
    <span class="ct" style="background:{C['part']}22;color:{C['part']}">Confusion</span>
    {to_html(fig_cm,'cm')}
  </div>
</div>

<div class="sec" style="margin-top:22px">
  <div class="st"><h2>Insights clés — Étape 4</h2><span class="sb">Bilan</span></div>
  <div class="ins">
    <div class="in" style="--a:{C['part']}">
      <div class="it">Meilleur modèle</div>
      <div class="iv">{nom_meilleur.split()[0]}</div>
      <div class="ix">{nom_meilleur} sélectionné avec F1={meilleur['f1']:.4f} et AUC={meilleur['auc']:.4f} sur le test set.</div>
    </div>
    <div class="in" style="--a:{C['violet']}">
      <div class="it">Vrais positifs</div>
      <div class="iv">{tp:,} / {(y_test==1).sum():,}</div>
      <div class="ix">{tp} partis correctement détectés sur {(y_test==1).sum()} réels. {fn} partis manqués (faux négatifs).</div>
    </div>
    <div class="in" style="--a:{C['primaire']}">
      <div class="it">Validation croisée</div>
      <div class="iv">{meilleur['cv_mean']:.4f}</div>
      <div class="ix">F1 moyen en 5-fold CV = {meilleur['cv_mean']:.4f} ± {meilleur['cv_std']:.4f}. Modèle robuste et stable.</div>
    </div>
    <div class="in" style="--a:{C['orange']}">
      <div class="it">Baseline (Log. Reg.)</div>
      <div class="iv">F1={resultats['Régression Logistique']['f1']:.4f}</div>
      <div class="ix">Le meilleur modèle surpasse la baseline de {(meilleur['f1']-resultats['Régression Logistique']['f1'])*100:.1f} pts de F1.</div>
    </div>
    <div class="in" style="--a:{C['reste']}">
      <div class="it">AUC-ROC</div>
      <div class="iv">{meilleur['auc']:.4f}</div>
      <div class="ix">AUC > 0.5 = meilleur que l'aléatoire. Plus proche de 1 = meilleure discrimination.</div>
    </div>
    <div class="in" style="--a:{C['accent']}">
      <div class="it">Prochaine étape</div>
      <div class="iv">Étape 5</div>
      <div class="ix">Optimisation des hyperparamètres du meilleur modèle avec GridSearchCV pour améliorer encore les performances.</div>
    </div>
  </div>
</div>

<div class="footer">
  <strong>HR Turnover Analytics</strong> — Étape 4 : Entraînement & Comparaison des Modèles ML
  <span>Seye Kiné | Bindia Adeline Thiara | M. Aidara | UCAO 2025-2026</span>
</div>
</body>
</html>"""

with open('etape4_rapport.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"""
  {'═'*66}
  ✅  ÉTAPE 4 TERMINÉE AVEC SUCCÈS
  {'─'*66}
   💾  pipeline_data.pkl    → enrichi et transmis à l'Étape 5
   📊  etape4_rapport.html → rapport HTML interactif
  {'─'*66}
  ➡️   Lancer :  python etape5_optimisation.py
  {'═'*66}
""")
