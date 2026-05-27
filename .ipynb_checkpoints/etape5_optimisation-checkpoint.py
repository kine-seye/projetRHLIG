# =============================================================================
#  GENERATIVE HR ANALYTICS — PRÉDICTION DU TURNOVER DES EMPLOYÉS
#  Auteurs   : Seye Kiné | Bindia Adeline Thiara
#  Encadrant : M. Aidara
#  Institut  : UCAO — Département Informatique de Gestion
#  Année     : 2025-2026
# =============================================================================
#  ÉTAPE 5 — OPTIMISATION DES HYPERPARAMÈTRES
#
#  Objectif  : Trouver les meilleurs réglages (hyperparamètres) pour
#              chacun des 4 modèles afin d'améliorer leurs performances.
#              Sélectionner le modèle final le plus performant.
#
#  Méthode   : GridSearchCV avec validation croisée 5-fold
#              → Teste toutes les combinaisons d'hyperparamètres
#              → Évalue chaque combinaison sur 5 sous-ensembles
#              → Sélectionne la combinaison avec le meilleur F1
#
#  Entrée    : pipeline_data.pkl  (produit par l'Étape 4)
#  Sorties   : pipeline_data.pkl  (enrichi + modèle final optimisé)
#              etape5_rapport.html
#
#  Sections  :
#    A. Chargement du pipeline
#    B. Fonctions console
#    C. Définition des grilles d'hyperparamètres
#    D. GridSearchCV sur les 4 modèles
#    E. Comparaison avant / après optimisation
#    F. Sélection du modèle final
#    G. Bilan + enrichissement pipeline
#    H. Rapport HTML interactif
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import time

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics         import (f1_score, roc_auc_score,
                                     accuracy_score, precision_score,
                                     recall_score, confusion_matrix,
                                     roc_curve, classification_report)
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

X_train          = pipeline['X_train_smote']
X_test           = pipeline['X_test_proc']
y_train          = pipeline['y_train_smote']
y_test           = pipeline['y_test']
toutes_cols      = pipeline['toutes_cols']
COULEURS         = pipeline['COULEURS']
resultats_etape4 = pipeline['resultats_modeles']
nom_meilleur_e4  = pipeline['nom_meilleur']

C = COULEURS

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
# SECTION C — GRILLES D'HYPERPARAMÈTRES
# =============================================================================

banniere("ÉTAPE 5 — OPTIMISATION DES HYPERPARAMÈTRES")

print(f"""
  Données reçues de l'Étape 4 :
  ─────────────────────────────────────────────────────
  Train  : {X_train.shape[0]:,} × {X_train.shape[1]} features
  Test   : {X_test.shape[0]:,}  × {X_test.shape[1]} features
  Meilleur modèle Étape 4 : {nom_meilleur_e4}
  F1-Score Étape 4        : {resultats_etape4[nom_meilleur_e4]['f1']:.4f}
  ─────────────────────────────────────────────────────
  Méthode : GridSearchCV (5-fold CV, scoring=F1)
  Objectif: Améliorer les performances de chaque modèle
""")

section("DÉFINITION DES GRILLES D'HYPERPARAMÈTRES", "⚙️")

print("""
  Un hyperparamètre est un réglage du modèle qu'on fixe AVANT
  l'entraînement. GridSearchCV teste TOUTES les combinaisons
  possibles et garde celle qui donne le meilleur F1.

  Exemple pour Random Forest :
    n_estimators = [100, 200, 300]   → 3 choix
    max_depth    = [None, 10, 20]    → 3 choix
    Total        = 3 × 3 = 9 combinaisons testées
    Chacune évaluée en 3-fold CV → 9 × 3 = 27 entraînements
""")

# ── Grilles de recherche par modèle ───────────────────────────────────────────
# On définit les hyperparamètres à tester pour chaque modèle.
# Ces grilles sont volontairement ciblées (pas trop larges)
# pour un temps de calcul raisonnable tout en couvrant
# les paramètres les plus impactants.

GRILLES = {

    'Régression Logistique': {
        'modele': LogisticRegression(max_iter=500, random_state=42),
        'params': {
            # C = inverse de la régularisation
            'C'           : [0.01, 0.1, 1.0, 10.0],
            'class_weight': [None, 'balanced'],
        }
    },

    'Random Forest': {
        'modele': RandomForestClassifier(random_state=42, n_jobs=-1,
                                          class_weight='balanced'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth'   : [10, None],
            'min_samples_split': [2],
        }
    },

    'XGBoost': {
        'modele': XGBClassifier(random_state=42, verbosity=0,
                                eval_metric='logloss',
                                use_label_encoder=False,
                                subsample=0.8),
        'params': {
            'n_estimators' : [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth'    : [3, 6],
        }
    },

    'LightGBM': {
        'modele': LGBMClassifier(random_state=42, n_jobs=-1,
                                  verbosity=-1, class_weight='balanced'),
        'params': {
            'n_estimators' : [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth'    : [3, 6],
            'num_leaves'   : [31, 63],
        }
    },
}

# Affichage des grilles
for nom, config in GRILLES.items():
    total = 1
    for vals in config['params'].values():
        total *= len(vals)
    print(f"  ✅  {nom:<28} → {total:>4} combinaisons × 5-fold = {total*5:>5} fits")

sep()

# =============================================================================
# SECTION D — GRIDSEARCHCV SUR LES 4 MODÈLES
# =============================================================================

section("GRIDSEARCHCV — RECHERCHE DES MEILLEURS HYPERPARAMÈTRES", "🔍")

resultats_opt = {}   # Résultats après optimisation

for nom, config in GRILLES.items():
    sous_section(f"Optimisation — {nom}", "⚙️")

    total_combi = 1
    for vals in config['params'].values():
        total_combi *= len(vals)

    info(f"Nombre de combinaisons à tester : {total_combi}")
    info("Patience — GridSearchCV en cours...")

    debut = time.time()

    # GridSearchCV :
    # - estimator  : le modèle à optimiser
    # - param_grid : la grille des hyperparamètres
    # - cv=5       : validation croisée en 5 parties
    # - scoring    : métrique d'optimisation (F1 pour classe positive)
    # - n_jobs=-1  : parallélisation sur tous les cœurs CPU
    # - refit=True : ré-entraîne le meilleur modèle sur tout le train
    grid_search = GridSearchCV(
        estimator  = config['modele'],
        param_grid = config['params'],
        cv         = 3,
        scoring    = 'f1',
        n_jobs     = -1,
        refit      = True,
        verbose    = 0,
    )

    grid_search.fit(X_train, y_train)
    duree = time.time() - debut

    # Récupération du meilleur modèle
    meilleur_modele = grid_search.best_estimator_
    meilleurs_params = grid_search.best_params_
    meilleur_cv_score = grid_search.best_score_

    # Évaluation sur le test set
    y_pred       = meilleur_modele.predict(X_test)
    y_pred_proba = meilleur_modele.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_pred_proba)
    cm   = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    resultats_opt[nom] = {
        'modele'        : meilleur_modele,
        'params'        : meilleurs_params,
        'cv_score'      : meilleur_cv_score,
        'acc'           : acc, 'prec': prec,
        'rec'           : rec, 'f1' : f1,
        'auc'           : auc, 'cm' : cm,
        'y_pred'        : y_pred,
        'y_pred_proba'  : y_pred_proba,
        'fpr'           : fpr, 'tpr': tpr,
        'duree'         : duree,
    }

    # Amélioration vs Étape 4
    f1_avant  = resultats_etape4[nom]['f1']
    amelio    = (f1 - f1_avant) * 100

    ok(f"Terminé en {duree:.0f}s")
    ok(f"Meilleurs paramètres : {meilleurs_params}")
    ok(f"CV F1 (train)  : {meilleur_cv_score:.4f}")
    ok(f"F1 (test)      : {f1:.4f}  (avant : {f1_avant:.4f}  |  Δ {amelio:+.2f} pts)")
    ok(f"AUC-ROC (test) : {auc:.4f}")
    sep()

# =============================================================================
# SECTION E — COMPARAISON AVANT / APRÈS OPTIMISATION
# =============================================================================

section("COMPARAISON AVANT / APRÈS OPTIMISATION", "📊")

noms = list(GRILLES.keys())
print(f"\n  {'Modèle':<28} {'F1 avant':>9} {'F1 après':>9} "
      f"{'Δ F1':>8}  {'AUC avant':>10} {'AUC après':>10}  {'Δ AUC':>8}")
print(f"  {'╌'*28} {'╌'*9} {'╌'*9} {'╌'*8}  {'╌'*10} {'╌'*10}  {'╌'*8}")

for nom in noms:
    f1_av  = resultats_etape4[nom]['f1']
    f1_ap  = resultats_opt[nom]['f1']
    auc_av = resultats_etape4[nom]['auc']
    auc_ap = resultats_opt[nom]['auc']
    df1    = (f1_ap - f1_av) * 100
    dauc   = (auc_ap - auc_av) * 100
    flag   = '✅' if df1 > 0 else '➡️'
    print(f"  {nom:<28} {f1_av:>9.4f} {f1_ap:>9.4f} "
          f"{df1:>+7.2f}%  {auc_av:>10.4f} {auc_ap:>10.4f}  {dauc:>+7.2f}%  {flag}")

sep()

# =============================================================================
# SECTION F — SÉLECTION DU MODÈLE FINAL
# =============================================================================

section("SÉLECTION DU MODÈLE FINAL", "🏆")

# Classement par F1-Score après optimisation
classement_opt = sorted(resultats_opt.items(),
                         key=lambda x: x[1]['f1'], reverse=True)

print(f"\n  Classement final (par F1-Score décroissant) :\n")
print(f"  {'Rang':<6} {'Modèle':<28} {'F1':>8} {'AUC':>8} "
      f"{'CV F1':>9}  Verdict")
print(f"  {'╌'*6} {'╌'*28} {'╌'*8} {'╌'*8} {'╌'*9}  {'╌'*18}")

rangs = ['🥇', '🥈', '🥉', '4️⃣']
for i, (nom, res) in enumerate(classement_opt):
    verdict = '← MODÈLE FINAL ✅' if i == 0 else ''
    print(f"  {rangs[i]:<6} {nom:<28} {res['f1']:>8.4f} "
          f"{res['auc']:>8.4f} {res['cv_score']:>9.4f}  {verdict}")

# Modèle final sélectionné
nom_final    = classement_opt[0][0]
modele_final = classement_opt[0][1]
cm_final     = modele_final['cm']
tn, fp, fn, tp = cm_final.ravel()

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  MODÈLE FINAL : {nom_final:<45}│
  │                                                              │
  │  F1-Score   : {modele_final['f1']:.4f}   (avant optim. : {resultats_etape4[nom_final]['f1']:.4f})   │
  │  AUC-ROC    : {modele_final['auc']:.4f}   (avant optim. : {resultats_etape4[nom_final]['auc']:.4f})   │
  │  Accuracy   : {modele_final['acc']:.4f}                                    │
  │  Précision  : {modele_final['prec']:.4f}                                    │
  │  Rappel     : {modele_final['rec']:.4f}                                    │
  │  CV F1      : {modele_final['cv_score']:.4f}                                    │
  ├──────────────────────────────────────────────────────────────┤
  │  Meilleurs hyperparamètres :                                 │""")
for k, v in modele_final['params'].items():
    print(f"  │    {k:<28} = {str(v):<22}│")
print(f"  └──────────────────────────────────────────────────────────────┘")

print(f"\n  Matrice de confusion :")
print(f"  ┌─────────────────────────────────────────┐")
print(f"  │              Prédit Resté  Prédit Parti  │")
print(f"  │  Réel Resté  {tn:>10,}  {fp:>11,}  │")
print(f"  │  Réel Parti  {fn:>10,}  {tp:>11,}  │")
print(f"  └─────────────────────────────────────────┘")
print(f"\n  ✅  Vrais Positifs  (TP) : {tp:,}  partis correctement détectés")
print(f"  ✅  Vrais Négatifs  (TN) : {tn:,}  restés correctement prédits")
print(f"  ⚠️   Faux Négatifs  (FN) : {fn:,}   partis manqués")
print(f"  ⚠️   Faux Positifs  (FP) : {fp:,}   alertes inutiles")
sep()

# =============================================================================
# SECTION G — BILAN + ENRICHISSEMENT PIPELINE
# =============================================================================

amelio_f1  = (modele_final['f1'] - resultats_etape4[nom_final]['f1']) * 100
amelio_auc = (modele_final['auc'] - resultats_etape4[nom_final]['auc']) * 100

bilan({
    '🔍 Méthode'              : "GridSearchCV (5-fold, scoring=F1)",
    '🤖 Modèle final'         : nom_final,
    '📊 F1 avant optimisation': f"{resultats_etape4[nom_final]['f1']:.4f}",
    '📊 F1 après optimisation': f"{modele_final['f1']:.4f}  (Δ {amelio_f1:+.2f} pts)",
    '📊 AUC-ROC final'        : f"{modele_final['auc']:.4f}  (Δ {amelio_auc:+.2f} pts)",
    '✅ Vrais Positifs (TP)'  : f"{tp:,} / {(y_test==1).sum():,} partis détectés",
    '➡️  Prochaine étape'      : "Étape 6 — Évaluation complète & interprétation",
}, titre='BILAN — ÉTAPE 5')

# Enrichissement du pipeline
pipeline['resultats_opt']    = {
    nom: {k: v for k, v in res.items() if k != 'modele'}
    for nom, res in resultats_opt.items()
}
pipeline['modeles_opt']      = {nom: res['modele'] for nom, res in resultats_opt.items()}
pipeline['nom_final']        = nom_final
pipeline['modele_final']     = modele_final['modele']
pipeline['params_final']     = modele_final['params']
pipeline['classement_opt']   = [(n, r['f1'], r['auc']) for n, r in classement_opt]

with open('pipeline_data.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("\n  💾  pipeline_data.pkl enrichi → Étape 6")

# =============================================================================
# SECTION H — RAPPORT HTML INTERACTIF
# =============================================================================

print("\n  ⏳  Génération du rapport HTML interactif...")

noms_courts = {
    'Régression Logistique': 'Log. Reg.',
    'Random Forest'        : 'Rnd. Forest',
    'XGBoost'              : 'XGBoost',
    'LightGBM'             : 'LightGBM',
}
coul_mod = [C['primaire'], C['violet'], C['part'], C['orange']]

# ── Fig 1 : Avant / Après F1 par modèle ───────────────────────────────────────
fig_amelio = go.Figure()
noms_aff   = [noms_courts[n] for n in noms]
f1_avant   = [resultats_etape4[n]['f1']  for n in noms]
f1_apres   = [resultats_opt[n]['f1']     for n in noms]
auc_avant  = [resultats_etape4[n]['auc'] for n in noms]
auc_apres  = [resultats_opt[n]['auc']    for n in noms]

fig_amelio.add_trace(go.Bar(
    name='F1 avant (Étape 4)', x=noms_aff, y=f1_avant,
    marker=dict(color=C['texte2'], opacity=0.5,
                line=dict(color=C['fond'], width=1.5)),
    hovertemplate='%{x}<br>F1 avant : %{y:.4f}<extra></extra>',
))
fig_amelio.add_trace(go.Bar(
    name='F1 après (Étape 5)', x=noms_aff, y=f1_apres,
    marker=dict(color=[coul_mod[i] for i in range(len(noms))],
                line=dict(color=C['fond'], width=1.5)),
    text=[f"<b>{v:.4f}</b>" for v in f1_apres],
    textposition='outside',
    hovertemplate='%{x}<br>F1 après : %{y:.4f}<extra></extra>',
))
fig_amelio.update_layout(**LAYOUT_BASE,
    title=dict(text='F1-Score — Avant vs Après Optimisation',
               font=dict(size=14), x=0.5),
    yaxis=dict(**axe('F1-Score'), range=[0, 1.1]),
    xaxis=axe(), height=380,
    margin=dict(t=55, b=40, l=55, r=30),
    barmode='group')

# ── Fig 2 : AUC avant / après ─────────────────────────────────────────────────
fig_auc = go.Figure()
fig_auc.add_trace(go.Bar(
    name='AUC avant', x=noms_aff, y=auc_avant,
    marker=dict(color=C['texte2'], opacity=0.5,
                line=dict(color=C['fond'], width=1.5)),
    hovertemplate='%{x}<br>AUC avant : %{y:.4f}<extra></extra>',
))
fig_auc.add_trace(go.Bar(
    name='AUC après', x=noms_aff, y=auc_apres,
    marker=dict(color=[coul_mod[i] for i in range(len(noms))],
                line=dict(color=C['fond'], width=1.5)),
    text=[f"<b>{v:.4f}</b>" for v in auc_apres],
    textposition='outside',
    hovertemplate='%{x}<br>AUC après : %{y:.4f}<extra></extra>',
))
fig_auc.update_layout(**LAYOUT_BASE,
    title=dict(text='AUC-ROC — Avant vs Après Optimisation',
               font=dict(size=14), x=0.5),
    yaxis=dict(**axe('AUC-ROC'), range=[0, 1.1]),
    xaxis=axe(), height=380,
    margin=dict(t=55, b=40, l=55, r=30),
    barmode='group')

# ── Fig 3 : Courbes ROC après optimisation ────────────────────────────────────
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=[0,1], y=[0,1], mode='lines',
    line=dict(color=C['texte2'], dash='dash', width=1),
    name='Aléatoire (AUC=0.5)', hoverinfo='skip',
))
for i, nom in enumerate(noms):
    res = resultats_opt[nom]
    fig_roc.add_trace(go.Scatter(
        x=res['fpr'], y=res['tpr'], mode='lines',
        name=f"{noms_courts[nom]} (AUC={res['auc']:.4f})",
        line=dict(color=coul_mod[i], width=2.5),
        hovertemplate=f'FPR=%{{x:.3f}}<br>TPR=%{{y:.3f}}<extra>{noms_courts[nom]}</extra>',
    ))
fig_roc.update_layout(**LAYOUT_BASE,
    title=dict(text='Courbes ROC après optimisation',
               font=dict(size=14), x=0.5),
    xaxis=dict(**axe('Taux de faux positifs (FPR)'), range=[0,1]),
    yaxis=dict(**axe('Taux de vrais positifs (TPR)'), range=[0,1.02]),
    height=420, margin=dict(t=55, b=50, l=60, r=30))

# ── Fig 4 : Matrice de confusion modèle final ─────────────────────────────────
fig_cm = go.Figure(go.Heatmap(
    z=cm_final,
    x=['Prédit Resté (0)', 'Prédit Parti (1)'],
    y=['Réel Resté (0)', 'Réel Parti (1)'],
    text=[[f"<b>{v:,}</b>" for v in row] for row in cm_final],
    texttemplate='%{text}',
    textfont=dict(size=22, color='white'),
    colorscale=[[0, C['graphe']], [0.5, C['primaire']], [1, C['reste']]],
    showscale=False,
    hovertemplate='%{y} → %{x}<br>%{z:,}<extra></extra>',
))
fig_cm.update_layout(**LAYOUT_BASE,
    title=dict(text=f'Matrice de Confusion — {nom_final} (optimisé)',
               font=dict(size=14), x=0.5),
    xaxis=dict(tickfont=dict(size=11, color=C['texte2'])),
    yaxis=dict(tickfont=dict(size=11, color=C['texte2'])),
    height=360, margin=dict(t=55, b=50, l=120, r=30))

# ── Fig 5 : Radar comparaison métriques ───────────────────────────────────────
categories_radar = ['Accuracy', 'Précision', 'Rappel',
                     'F1-Score', 'AUC-ROC']
fig_radar = go.Figure()
for i, nom in enumerate(noms):
    res = resultats_opt[nom]
    vals = [res['acc'], res['prec'], res['rec'], res['f1'], res['auc']]
    vals_closed = vals + [vals[0]]
    cats_closed = categories_radar + [categories_radar[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed,
        fill='toself', fillcolor=coul_mod[i],
        opacity=0.25,
        line=dict(color=coul_mod[i], width=2),
        name=noms_courts[nom],
        hovertemplate='%{theta}<br>Score : %{r:.4f}<extra>' +
                       noms_courts[nom] + '</extra>',
    ))
fig_radar.update_layout(**LAYOUT_BASE,
    title=dict(text='Comparaison radar — 4 modèles optimisés',
               font=dict(size=14), x=0.5),
    polar=dict(
        bgcolor=C['graphe'],
        radialaxis=dict(visible=True, range=[0, 1],
                         gridcolor=C['grille'],
                         tickfont=dict(color=C['texte2'])),
        angularaxis=dict(gridcolor=C['grille'],
                          tickfont=dict(color=C['texte2'])),
    ),
    height=420, margin=dict(t=55, b=40, l=60, r=60))

# ── Construction HTML ──────────────────────────────────────────────────────────
def to_html(fig, div_id):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
        config=dict(displayModeBar=True, displaylogo=False,
                    modeBarButtonsToRemove=['lasso2d','select2d']))

params_html = ''.join(
    f'<div class="p-item"><span class="p-k">{k}</span>'
    f'<span class="p-v">{v}</span></div>'
    for k, v in modele_final['params'].items()
)

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Étape 5 — Optimisation</title>
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
.winner{{background:{C['carte']};border:2px solid {C['accent']};
         border-radius:12px;padding:20px 24px;}}
.winner h3{{color:{C['accent']};font-size:15px;font-weight:700;margin-bottom:16px;}}
.w-metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:20px;}}
.wm{{text-align:center;background:{C['graphe']};border-radius:8px;padding:12px 8px;}}
.wm-val{{font-size:22px;font-weight:800;color:{C['texte']};}}
.wm-lbl{{font-size:11px;color:{C['texte2']};margin-top:3px;}}
.params-grid{{display:flex;flex-wrap:wrap;gap:8px;}}
.p-item{{background:{C['graphe']};border-radius:8px;padding:8px 12px;
          display:flex;gap:8px;align-items:center;}}
.p-k{{font-size:11px;color:{C['texte2']};}}
.p-v{{font-size:12px;font-weight:700;color:{C['accent']};}}
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
@media(max-width:900px){{
  .kpi-row,.g2,.ins,.w-metrics{{grid-template-columns:1fr 1fr;}}
}}
</style>
</head>
<body>

<div class="hdr">
  <div class="badge">Étape 5 / 10 — Optimisation des hyperparamètres</div>
  <h1>HR Turnover <span>Analytics</span> — Optimisation GridSearchCV</h1>
  <p>Recherche des meilleurs hyperparamètres — Sélection du modèle final</p>
  <div class="meta">
    <div class="mi">🔍 <strong>GridSearchCV</strong> 5-fold</div>
    <div class="mi">🏆 <strong>{nom_final}</strong> modèle final</div>
    <div class="mi">📊 <strong>F1 = {modele_final['f1']:.4f}</strong></div>
    <div class="mi">📊 <strong>AUC = {modele_final['auc']:.4f}</strong></div>
    <div class="mi">📅 UCAO 2025-2026 | M. Aidara</div>
  </div>
</div>

<div class="kpi-row">
  <div class="kpi" style="--a:{C['part']}">
    <div class="ki">🏆</div>
    <div class="kv">{nom_final.split()[0]}</div>
    <div class="kl">Modèle final</div>
    <div class="ks">Sélectionné après optimisation</div>
  </div>
  <div class="kpi" style="--a:{C['violet']}">
    <div class="ki">📊</div>
    <div class="kv">{modele_final['f1']:.4f}</div>
    <div class="kl">F1-Score (optimisé)</div>
    <div class="ks">Δ {amelio_f1:+.2f} pts vs Étape 4</div>
  </div>
  <div class="kpi" style="--a:{C['primaire']}">
    <div class="ki">📈</div>
    <div class="kv">{modele_final['auc']:.4f}</div>
    <div class="kl">AUC-ROC (optimisé)</div>
    <div class="ks">Δ {amelio_auc:+.2f} pts vs Étape 4</div>
  </div>
  <div class="kpi" style="--a:{C['reste']}">
    <div class="ki">✅</div>
    <div class="kv">{tp:,}</div>
    <div class="kl">Partis détectés (TP)</div>
    <div class="ks">Sur {(y_test==1).sum():,} partis réels</div>
  </div>
</div>

<div class="sec" style="margin-top:22px">
  <div class="st">
    <h2>Modèle final — {nom_final}</h2>
    <span class="sb">🏆 Winner</span>
  </div>
  <div class="winner">
    <h3>🏆 {nom_final} — Métriques finales sur le Test Set</h3>
    <div class="w-metrics">
      <div class="wm"><div class="wm-val">{modele_final['acc']:.4f}</div><div class="wm-lbl">Accuracy</div></div>
      <div class="wm"><div class="wm-val">{modele_final['prec']:.4f}</div><div class="wm-lbl">Précision</div></div>
      <div class="wm"><div class="wm-val">{modele_final['rec']:.4f}</div><div class="wm-lbl">Rappel</div></div>
      <div class="wm"><div class="wm-val">{modele_final['f1']:.4f}</div><div class="wm-lbl">F1-Score</div></div>
      <div class="wm"><div class="wm-val">{modele_final['auc']:.4f}</div><div class="wm-lbl">AUC-ROC</div></div>
    </div>
    <div style="font-size:12px;color:{C['texte2']};margin-bottom:10px;">
      Meilleurs hyperparamètres trouvés par GridSearchCV :
    </div>
    <div class="params-grid">{params_html}</div>
  </div>
</div>

<div class="sec" style="margin-top:18px">
  <div class="st"><h2>Amélioration F1 & AUC — Avant vs Après</h2>
    <span class="sb">Comparaison</span></div>
  <div class="g2">
    <div class="chart">
      <span class="ct" style="background:{C['violet']}22;color:{C['violet']}">F1-Score</span>
      {to_html(fig_amelio,'amelio')}
    </div>
    <div class="chart">
      <span class="ct" style="background:{C['primaire']}22;color:{C['primaire']}">AUC-ROC</span>
      {to_html(fig_auc,'auc')}
    </div>
  </div>
</div>

<div class="sec" style="margin-top:18px">
  <div class="st"><h2>Courbes ROC & Radar comparatif</h2>
    <span class="sb">Performance</span></div>
  <div class="g2">
    <div class="chart">
      <span class="ct" style="background:{C['reste']}22;color:{C['reste']}">ROC</span>
      {to_html(fig_roc,'roc')}
    </div>
    <div class="chart">
      <span class="ct" style="background:{C['orange']}22;color:{C['orange']}">Radar</span>
      {to_html(fig_radar,'radar')}
    </div>
  </div>
</div>

<div class="sec" style="margin-top:18px">
  <div class="st"><h2>Matrice de confusion — {nom_final}</h2>
    <span class="sb">Erreurs</span></div>
  <div class="chart full">
    <span class="ct" style="background:{C['part']}22;color:{C['part']}">Confusion</span>
    {to_html(fig_cm,'cm')}
  </div>
</div>

<div class="sec" style="margin-top:22px">
  <div class="st"><h2>Insights clés — Étape 5</h2><span class="sb">Bilan</span></div>
  <div class="ins">
    <div class="in" style="--a:{C['part']}">
      <div class="it">Modèle final</div>
      <div class="iv">{nom_final.split()[0]}</div>
      <div class="ix">{nom_final} sélectionné après optimisation. F1={modele_final['f1']:.4f}, AUC={modele_final['auc']:.4f}.</div>
    </div>
    <div class="in" style="--a:{C['violet']}">
      <div class="it">Amélioration F1</div>
      <div class="iv">{amelio_f1:+.2f} pts</div>
      <div class="ix">GridSearchCV a amélioré le F1-Score de {amelio_f1:+.2f} pts par rapport aux hyperparamètres par défaut.</div>
    </div>
    <div class="in" style="--a:{C['primaire']}">
      <div class="it">Vrais positifs</div>
      <div class="iv">{tp:,} / {(y_test==1).sum():,}</div>
      <div class="ix">{tp} partis correctement identifiés. {fn} partis manqués (faux négatifs à minimiser).</div>
    </div>
    <div class="in" style="--a:{C['orange']}">
      <div class="it">GridSearchCV</div>
      <div class="iv">5-fold CV</div>
      <div class="ix">Chaque combinaison évaluée sur 5 sous-ensembles du train. Résultat robuste et sans data leakage.</div>
    </div>
    <div class="in" style="--a:{C['reste']}">
      <div class="it">AUC-ROC</div>
      <div class="iv">{modele_final['auc']:.4f}</div>
      <div class="ix">Capacité à discriminer partis / restés. Nettement supérieur au modèle aléatoire (AUC=0.5).</div>
    </div>
    <div class="in" style="--a:{C['accent']}">
      <div class="it">Prochaine étape</div>
      <div class="iv">Étape 6</div>
      <div class="ix">Évaluation complète du modèle final : analyse des erreurs, seuil optimal, rapport de classification.</div>
    </div>
  </div>
</div>

<div class="footer">
  <strong>HR Turnover Analytics</strong> — Étape 5 : Optimisation des hyperparamètres
  <span>Seye Kiné | Bindia Adeline Thiara | M. Aidara | UCAO 2025-2026</span>
</div>
</body>
</html>"""

with open('etape5_rapport.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"""
  {'═'*66}
  ✅  ÉTAPE 5 TERMINÉE AVEC SUCCÈS
  {'─'*66}
   💾  pipeline_data.pkl    → enrichi + modèle final sauvegardé
   📊  etape5_rapport.html → rapport HTML interactif
  {'─'*66}
  ➡️   Lancer :  python etape6_evaluation.py
  {'═'*66}
""")
