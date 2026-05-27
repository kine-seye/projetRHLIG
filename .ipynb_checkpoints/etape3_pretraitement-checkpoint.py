# =============================================================================
#  GENERATIVE HR ANALYTICS — PRÉDICTION DU TURNOVER DES EMPLOYÉS
#  Auteurs   : Seye Kiné | Bindia Adeline Thiara
#  Encadrant : M. Aidara
#  Institut  : UCAO — Département Informatique de Gestion
#  Année     : 2025-2026
# =============================================================================
#  ÉTAPE 3 — PRÉTRAITEMENT ET CONSTRUCTION DU PIPELINE
#
#  Objectif  : Transformer les données brutes en données prêtes
#              pour les modèles de Machine Learning :
#              1. Encoder les variables catégorielles (texte → nombres)
#              2. Normaliser les variables numériques (même échelle)
#              3. Construire le pipeline sklearn (ColumnTransformer)
#              4. Séparer train / test (80% / 20%)
#              5. Appliquer SMOTE si nécessaire (équilibrage des classes)
#
#  Entrée    : pipeline_data.pkl  (produit par l'Étape 2)
#  Sorties   : pipeline_data.pkl  (enrichi)
#              etape3_rapport.html
#
#  Sections  :
#    A. Chargement du pipeline
#    B. Fonctions console
#    C. Préparation des features (X) et de la cible (y)
#    D. Encodage des variables catégorielles
#    E. Normalisation des variables numériques
#    F. Construction du ColumnTransformer (pipeline sklearn)
#    G. Split Train / Test
#    H. SMOTE (équilibrage des classes)
#    I. Vérifications finales
#    J. Bilan + enrichissement pipeline
#    K. Rapport HTML interactif
# =============================================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline
from imblearn.over_sampling    import SMOTE
import plotly.graph_objects    as go
from plotly.subplots           import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION A — CHARGEMENT DU PIPELINE
# =============================================================================

with open('pipeline_data.pkl', 'rb') as f:
    pipeline = pickle.load(f)

df                = pipeline['df']
COLS_NUM          = pipeline['COLS_NUM']
COLS_CAT          = pipeline['COLS_CAT']
CIBLE             = pipeline['CIBLE']
n                 = pipeline['n']
taux              = pipeline['taux']
COULEURS          = pipeline['COULEURS']
vars_num_retenues = pipeline['vars_num_retenues']
vars_cat_retenues = pipeline['vars_cat_retenues']

# Paramètres Plotly (cohérence visuelle avec les étapes précédentes)
LAYOUT_BASE = dict(
    paper_bgcolor = COULEURS['carte'],
    plot_bgcolor  = COULEURS['graphe'],
    font          = dict(color=COULEURS['texte'], family='Segoe UI, sans-serif'),
    legend        = dict(bgcolor=COULEURS['carte'],
                         bordercolor=COULEURS['bord'],
                         font=dict(color=COULEURS['texte'])),
)

def axe(titre=''):
    return dict(title=titre, gridcolor=COULEURS['grille'],
                showgrid=True, zeroline=False,
                tickfont=dict(color=COULEURS['texte2']))

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
def warn(texte): print(f"  │  ⚠️   {texte}")

def tableau(x, indent=5):
    t = x if isinstance(x, str) else x.to_string()
    for l in t.split('\n'):
        print(' '*indent + l)

def barre(valeur, max_val=100, largeur=30):
    n_ = int(valeur / max_val * largeur)
    return f"[{'█'*n_}{'░'*(largeur-n_)}] {valeur:.1f}%"

def bilan(items, titre='BILAN'):
    print(f"\n  ╔══ {titre} {'═'*(L-len(titre)-5)}╗")
    for k, v in items.items():
        s = f"  ║  {k:<32} {v}"
        print(f"{s:<{L+4}}║")
    print(f"  ╚{'═'*(L+1)}╝")

# =============================================================================
# SECTION C — PRÉPARATION DES FEATURES ET DE LA CIBLE
# =============================================================================

banniere("ÉTAPE 3 — PRÉTRAITEMENT ET PIPELINE SKLEARN")
print(f"""
  Données reçues des Étapes 1 & 2 :
  ─────────────────────────────────────────────────────
  Dataset   : {n:,} employés × {df.shape[1]} variables
  Num. ret. : {vars_num_retenues}
  Cat. ret. : {vars_cat_retenues}
  Cible     : {CIBLE}  (0=Reste | 1=Part)
""")

section("PRÉPARATION DES FEATURES (X) ET DE LA CIBLE (y)", "🗂️")

# X = toutes les variables prédictives (features)
# y = la variable qu'on veut prédire (Attrition)
# On garde uniquement les variables retenues à l'Étape 2

FEATURES = vars_num_retenues + vars_cat_retenues
X = df[FEATURES].copy()
y = df[CIBLE].copy()

print(f"\n  Variables utilisées comme features ({len(FEATURES)}) :")
print(f"\n  Numériques ({len(vars_num_retenues)}) :")
for v in vars_num_retenues:
    print(f"    • {v:<35} type : {df[v].dtype}")
print(f"\n  Catégorielles ({len(vars_cat_retenues)}) :")
for v in vars_cat_retenues:
    vals = df[v].unique().tolist()
    print(f"    • {v:<35} modalités : {vals}")

print(f"\n  Variable cible (y) : {CIBLE}")
print(f"    0 (Reste) : {(y==0).sum():,}  |  1 (Part) : {(y==1).sum():,}")
sep()

# =============================================================================
# SECTION D — ENCODAGE DES VARIABLES CATÉGORIELLES
# =============================================================================
# Les modèles ML ne comprennent que des nombres.
# On doit donc transformer les variables texte en nombres.
#
# DEUX méthodes possibles :
#
# 1. OneHotEncoder (OHE) — recommandé pour les variables nominales
#    Transforme chaque modalité en une colonne binaire (0 ou 1)
#    Exemple : Gender [Male, Female] → Gender_Male (0/1) + Gender_Female (0/1)
#    Avantage : pas d'ordre artificiel entre les modalités
#    Utilisé pour : Gender, Department, JobRole, OverTime
#
# 2. LabelEncoder — pour les variables ordinales (avec ordre naturel)
#    Transforme les modalités en entiers : No=0, Yes=1
#    Exemple : OverTime [No, Yes] → OverTime [0, 1]
#    Utilisé comme alternative simple pour les variables binaires

section("ENCODAGE DES VARIABLES CATÉGORIELLES", "🔤")

sous_section("d1. Avant encodage — aperçu des variables catégorielles", "👁️")
print()
for col in vars_cat_retenues:
    counts = df[col].value_counts()
    print(f"  {col} :")
    for val, cnt in counts.items():
        pct = cnt / n * 100
        bar = '█' * int(pct/4) + '░'*(25-int(pct/4))
        print(f"    {str(val):<20} {cnt:>6,}  ({pct:.1f}%)  {bar}")
    print()
sep()

sous_section("d2. Méthode choisie — OneHotEncoder", "⚙️")
print("""
  Pourquoi OneHotEncoder ?
  → Nos variables catégorielles sont NOMINALES (pas d'ordre naturel)
  → Gender : Male/Female     → pas de Male > Female
  → Department : HR/Sales/IT → pas de HR > Sales
  → OneHotEncoder évite d'introduire un ordre artificiel
  → C'est la méthode recommandée dans scikit-learn pour ce type de données

  Paramètre drop='first' :
  → On supprime la première colonne de chaque variable
  → Evite la multicolinéarité parfaite (problème statistique)
  → Exemple : Gender → Gender_Male uniquement (Female = si Male=0)
""")

# Simulation manuelle pour afficher le résultat (la vraie transformation
# sera faite dans le pipeline sklearn à la Section F)
ohe_demo = OneHotEncoder(drop='first', sparse_output=False)
X_cat_demo = ohe_demo.fit_transform(X[vars_cat_retenues])
cols_ohe   = ohe_demo.get_feature_names_out(vars_cat_retenues)

print(f"  Résultat après OneHotEncoder :")
print(f"    Avant : {len(vars_cat_retenues)} colonnes catégorielles")
print(f"    Après : {len(cols_ohe)} colonnes binaires (0/1)")
print(f"\n  Nouvelles colonnes créées :")
for col in cols_ohe:
    print(f"    ✅  {col}")
sep()

# =============================================================================
# SECTION E — NORMALISATION DES VARIABLES NUMÉRIQUES
# =============================================================================
# Les variables numériques ont des échelles très différentes :
#   - Age : de 20 à 59
#   - MonthlyIncome : de 2 000 à 15 000
#   - YearsAtCompany : de 0 à 19
#
# Sans normalisation, le modèle pense que MonthlyIncome (15 000)
# est beaucoup plus important que YearsAtCompany (19).
#
# StandardScaler — méthode z-score :
#   Formule : z = (valeur - moyenne) / écart-type
#   Résultat : moyenne=0, écart-type=1 pour chaque variable
#   Toutes les variables sont alors sur la même échelle.

section("NORMALISATION DES VARIABLES NUMÉRIQUES", "📏")

sous_section("e1. Avant normalisation — statistiques des variables numériques", "👁️")
stats_avant = df[vars_num_retenues].describe().round(2).loc[['mean','std','min','max']]
tableau(stats_avant.to_string())

sous_section("e2. Méthode choisie — StandardScaler (z-score)", "⚙️")
print("""
  Formule  : z = (x - moyenne) / écart-type
  Résultat : chaque variable aura moyenne=0 et écart-type=1

  Avant : JobSatisfaction va de 1 à 4
          MonthlyIncome   va de 2 000 à 15 000
  Après : les deux variables sont sur la même échelle [-3, +3]

  Important : on calcule moyenne et écart-type UNIQUEMENT sur
  les données d'entraînement (train set), puis on applique
  les mêmes paramètres au test set.
  → Evite le data leakage (fuite de données du test vers le train)
""")

# Simulation pour afficher le résultat
scaler_demo = StandardScaler()
X_num_demo  = scaler_demo.fit_transform(X[vars_num_retenues])
df_apres    = pd.DataFrame(X_num_demo, columns=vars_num_retenues)
stats_apres = df_apres.describe().round(3).loc[['mean','std','min','max']]

sous_section("e3. Après normalisation — statistiques", "👁️")
tableau(stats_apres.to_string())
print(f"\n  ✅  Toutes les variables ont maintenant :")
print(f"      moyenne ≈ 0.000  |  écart-type ≈ 1.000")
sep()

# =============================================================================
# SECTION F — CONSTRUCTION DU PIPELINE SKLEARN
# =============================================================================
# Le ColumnTransformer applique des transformations différentes
# selon le type de colonne :
#   - Variables numériques  → StandardScaler
#   - Variables catégorielles → OneHotEncoder
#
# Avantage du pipeline sklearn :
#   1. Tout est automatique et cohérent train/test
#   2. Pas de data leakage possible
#   3. Reproductible et facile à sauvegarder
#   4. Compatible avec SHAP et LIME (Étape 8)

section("CONSTRUCTION DU PIPELINE SKLEARN", "⚙️")

print("""
  Architecture du pipeline :

  Données brutes (X)
       │
       ├── Variables numériques  →  StandardScaler  →  z-scores
       │   (YearsSinceLastPromotion, DistanceFromHome,
       │    JobSatisfaction, WorkLifeBalance)
       │
       └── Variables catégorielles → OneHotEncoder  →  colonnes 0/1
           (Gender, Department, JobRole, OverTime)
       │
       └── Toutes réunies → X_transformé → prêt pour le modèle ML
""")

# Construction du ColumnTransformer
# Il applique StandardScaler sur les colonnes numériques
# et OneHotEncoder sur les colonnes catégorielles
preprocesseur = ColumnTransformer(
    transformers=[
        (
            'numerique',                     # Nom du transformateur
            StandardScaler(),                # Méthode appliquée
            vars_num_retenues                # Colonnes concernées
        ),
        (
            'categoriel',                    # Nom du transformateur
            OneHotEncoder(                   # Méthode appliquée
                drop         = 'first',      # Évite multicolinéarité
                handle_unknown = 'ignore',   # Gère les modalités inconnues
                sparse_output  = False       # Retourne un tableau dense
            ),
            vars_cat_retenues                # Colonnes concernées
        ),
    ],
    remainder = 'drop'   # Supprime toutes les autres colonnes
)

ok("ColumnTransformer créé avec succès")
info(f"Numériques  : {len(vars_num_retenues)} colonnes → StandardScaler")
info(f"Catégoriels : {len(vars_cat_retenues)} colonnes → OneHotEncoder")
sep()

# =============================================================================
# SECTION G — SPLIT TRAIN / TEST
# =============================================================================
# On divise le dataset en deux parties :
#   - Train set (80%) : utilisé pour entraîner le modèle
#   - Test set  (20%) : utilisé pour évaluer le modèle sur des données inconnues
#
# Paramètre stratify=y :
#   Garantit que la proportion de 0/1 est la même dans train et test.
#   Sans ça, on pourrait avoir 60% de 1 dans train et 30% dans test.
#
# Paramètre random_state=42 :
#   Fixe la graine aléatoire → résultats reproductibles à chaque exécution.

section("SPLIT TRAIN / TEST (80% / 20%)", "✂️")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,   # 20% pour le test
    random_state = 42,     # Reproductibilité
    stratify     = y       # Même proportion de 0/1 dans train et test
)

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  TRAIN SET (entraînement)                                    │
  │  Taille   : {len(X_train):>5,} employés  ({len(X_train)/n*100:.0f}% du dataset)           │
  │  Restés   : {(y_train==0).sum():>5,}  ({(y_train==0).mean()*100:.1f}%)                           │
  │  Partis   : {(y_train==1).sum():>5,}  ({(y_train==1).mean()*100:.1f}%)                           │
  ├──────────────────────────────────────────────────────────────┤
  │  TEST SET (évaluation)                                       │
  │  Taille   : {len(X_test):>5,} employés  ({len(X_test)/n*100:.0f}% du dataset)            │
  │  Restés   : {(y_test==0).sum():>5,}  ({(y_test==0).mean()*100:.1f}%)                           │
  │  Partis   : {(y_test==1).sum():>5,}  ({(y_test==1).mean()*100:.1f}%)                           │
  ├──────────────────────────────────────────────────────────────┤
  │  Stratification : ✅  Proportions identiques dans train/test │
  │  Reproductibilité: ✅  random_state=42                       │
  └──────────────────────────────────────────────────────────────┘
""")

# Application du préprocesseur sur train et test
# FIT sur train uniquement → évite data leakage
# TRANSFORM sur train ET test avec les mêmes paramètres
X_train_proc = preprocesseur.fit_transform(X_train, y_train)
X_test_proc  = preprocesseur.transform(X_test)

# Récupération des noms de colonnes après transformation
cols_num_apres = vars_num_retenues
cols_cat_apres = preprocesseur.named_transformers_['categoriel']\
                              .get_feature_names_out(vars_cat_retenues).tolist()
toutes_cols    = cols_num_apres + cols_cat_apres

print(f"  Après transformation :")
print(f"    X_train : {X_train_proc.shape[0]:,} lignes × {X_train_proc.shape[1]} colonnes")
print(f"    X_test  : {X_test_proc.shape[0]:,} lignes  × {X_test_proc.shape[1]} colonnes")
print(f"\n  Colonnes après encodage ({len(toutes_cols)}) :")
for col in toutes_cols:
    print(f"    • {col}")
sep()

# =============================================================================
# SECTION H — SMOTE (ÉQUILIBRAGE DES CLASSES)
# =============================================================================
# SMOTE = Synthetic Minority Over-sampling TEchnique
#
# Principe :
#   On génère des exemples SYNTHÉTIQUES (artificiels) de la classe minoritaire
#   en interpolant entre des exemples existants.
#
#   Exemple simplifié :
#   Employé A (parti) : Age=30, Salaire=5000, JobSat=1
#   Employé B (parti) : Age=35, Salaire=6000, JobSat=2
#   SMOTE crée : Age=32, Salaire=5400, JobSat=1.4 (entre A et B)
#
# Quand l'utiliser ?
#   Notre dataset : 55.9% restés / 44.1% partis → ratio 1.27:1
#   Ce déséquilibre est modéré → SMOTE optionnel
#   On l'applique quand même pour améliorer la détection des partis.
#
# IMPORTANT : SMOTE s'applique UNIQUEMENT sur le train set
#             JAMAIS sur le test set (fausserait l'évaluation)

section("SMOTE — ÉQUILIBRAGE DES CLASSES", "⚖️")

print(f"""
  Avant SMOTE (train set) :
  Restés (0) : {(y_train==0).sum():,}  ({(y_train==0).mean()*100:.1f}%)
  Partis (1) : {(y_train==1).sum():,}  ({(y_train==1).mean()*100:.1f}%)
  Ratio      : {(y_train==0).sum()/(y_train==1).sum():.2f} : 1
""")

# Application de SMOTE
smote = SMOTE(
    random_state  = 42,    # Reproductibilité
    k_neighbors   = 5,     # Nombre de voisins pour l'interpolation
    sampling_strategy = 'auto'  # Équilibre parfait 50/50
)
X_train_smote, y_train_smote = smote.fit_resample(X_train_proc, y_train)

print(f"  Après SMOTE (train set) :")
print(f"  Restés (0) : {(y_train_smote==0).sum():,}  ({(y_train_smote==0).mean()*100:.1f}%)")
print(f"  Partis (1) : {(y_train_smote==1).sum():,}  ({(y_train_smote==1).mean()*100:.1f}%)")
print(f"  Ratio      : {(y_train_smote==0).sum()/(y_train_smote==1).sum():.2f} : 1")
print(f"\n  Exemples synthétiques créés : {len(X_train_smote)-len(X_train_proc):,}")
print(f"  Test set   : NON modifié (évaluation sur données réelles)")
sep()

# =============================================================================
# SECTION I — VÉRIFICATIONS FINALES
# =============================================================================

section("VÉRIFICATIONS FINALES", "🔍")

print(f"\n  Vérification 1 — Dimensions :")
ok(f"X_train_smote : {X_train_smote.shape[0]:,} × {X_train_smote.shape[1]}")
ok(f"X_test_proc   : {X_test_proc.shape[0]:,}  × {X_test_proc.shape[1]}")
ok(f"y_train_smote : {len(y_train_smote):,} valeurs")
ok(f"y_test        : {len(y_test):,}  valeurs")

print(f"\n  Vérification 2 — Pas de NaN après transformation :")
nan_train = np.isnan(X_train_smote).sum()
nan_test  = np.isnan(X_test_proc).sum()
ok(f"NaN dans train : {nan_train}") if nan_train==0 else warn(f"NaN dans train : {nan_train}")
ok(f"NaN dans test  : {nan_test}")  if nan_test==0  else warn(f"NaN dans test  : {nan_test}")

print(f"\n  Vérification 3 — Équilibre des classes :")
ok(f"Train (après SMOTE) : {(y_train_smote==0).sum():,} restés / {(y_train_smote==1).sum():,} partis")
ok(f"Test  (réel)        : {(y_test==0).sum():,} restés / {(y_test==1).sum():,} partis")

print(f"\n  Vérification 4 — Séquence de transformation :")
ok("fit_transform() sur train uniquement → pas de data leakage")
ok("transform() sur test avec les mêmes paramètres")
ok("SMOTE appliqué sur train uniquement")
sep()

# =============================================================================
# SECTION J — BILAN + ENRICHISSEMENT DU PIPELINE
# =============================================================================

bilan({
    '📊 Features (X)'         : f"{len(FEATURES)} variables ({len(vars_num_retenues)} num. + {len(vars_cat_retenues)} cat.)",
    '🔤 Après encodage'        : f"{len(toutes_cols)} colonnes au total",
    '✂️  Train set'            : f"{len(X_train_smote):,} exemples (après SMOTE)",
    '✂️  Test set'             : f"{len(X_test_proc):,}  exemples (données réelles)",
    '⚖️  Équilibre train'      : f"50.0% / 50.0% (après SMOTE)",
    '⚖️  Équilibre test'       : f"{(y_test==0).mean()*100:.1f}% / {(y_test==1).mean()*100:.1f}% (réel)",
    '➡️  Prochaine étape'      : "Étape 4 — Entraînement des modèles ML",
}, titre='BILAN — ÉTAPE 3')

# Enrichissement du pipeline
pipeline['preprocesseur']    = preprocesseur    # Le ColumnTransformer ajusté
pipeline['X_train_smote']    = X_train_smote    # Train transformé + équilibré
pipeline['X_test_proc']      = X_test_proc      # Test transformé
pipeline['y_train_smote']    = y_train_smote    # Labels train
pipeline['y_test']           = y_test           # Labels test (réels)
pipeline['X_train_raw']      = X_train          # Train brut (pour SHAP)
pipeline['X_test_raw']       = X_test           # Test brut (pour SHAP)
pipeline['toutes_cols']      = toutes_cols       # Noms des colonnes après encodage
pipeline['FEATURES']         = FEATURES         # Liste des features

with open('pipeline_data.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("\n  💾  pipeline_data.pkl enrichi → Étape 4")

# =============================================================================
# SECTION K — RAPPORT HTML INTERACTIF
# =============================================================================

print("\n  ⏳  Génération du rapport HTML interactif...")

C = COULEURS

# ── Fig 1 : Avant / Après SMOTE ───────────────────────────────────────────────
fig_smote = go.Figure()
categories = ['Train avant SMOTE', 'Train après SMOTE', 'Test (réel)']
restés     = [(y_train==0).sum(), (y_train_smote==0).sum(), (y_test==0).sum()]
partis     = [(y_train==1).sum(), (y_train_smote==1).sum(), (y_test==1).sum()]

fig_smote.add_trace(go.Bar(
    name='Resté (0)', x=categories, y=restés,
    marker=dict(color=C['reste'], line=dict(color=C['fond'], width=2)),
    text=[f"{v:,}" for v in restés], textposition='inside',
    hovertemplate='%{x}<br>Restés : %{y:,}<extra></extra>',
))
fig_smote.add_trace(go.Bar(
    name='Parti  (1)', x=categories, y=partis,
    marker=dict(color=C['part'], line=dict(color=C['fond'], width=2)),
    text=[f"{v:,}" for v in partis], textposition='inside',
    hovertemplate='%{x}<br>Partis : %{y:,}<extra></extra>',
))
fig_smote.update_layout(**LAYOUT_BASE,
    title=dict(text='Équilibrage des classes — Avant / Après SMOTE',
               font=dict(size=14), x=0.5),
    yaxis=dict(**axe('Nombre d\'employés')),
    xaxis=axe(), height=360,
    margin=dict(t=55, b=40, l=60, r=30),
    barmode='group')

# ── Fig 2 : Variables avant / après normalisation ─────────────────────────────
fig_scale = make_subplots(rows=1, cols=2,
    subplot_titles=['Avant normalisation', 'Après normalisation'],
    horizontal_spacing=0.12)

# Avant — box plots bruts
for col in vars_num_retenues:
    fig_scale.add_trace(go.Box(
        y=X[col], name=col[:12],
        marker_color=C['primaire'], showlegend=False,
        hovertemplate=f'{col}<br>%{{y:.2f}}<extra>Avant</extra>',
    ), row=1, col=1)

# Après — box plots normalisés
X_norm_demo = pd.DataFrame(
    StandardScaler().fit_transform(X[vars_num_retenues]),
    columns=vars_num_retenues
)
for col in vars_num_retenues:
    fig_scale.add_trace(go.Box(
        y=X_norm_demo[col], name=col[:12],
        marker_color=C['reste'], showlegend=False,
        hovertemplate=f'{col}<br>%{{y:.2f}}<extra>Après</extra>',
    ), row=1, col=2)

fig_scale.update_layout(**LAYOUT_BASE,
    title=dict(text='StandardScaler — Effet de la normalisation',
               font=dict(size=14), x=0.5),
    height=380, margin=dict(t=60, b=40, l=50, r=30))
for i in [1, 2]:
    fig_scale.update_yaxes(gridcolor=C['grille'], row=1, col=i)
    fig_scale.update_xaxes(gridcolor=C['grille'], row=1, col=i)

# ── Fig 3 : Colonnes après encodage ───────────────────────────────────────────
# Barres montrant combien de colonnes viennent de chaque variable
nb_par_var = {}
for col in vars_num_retenues:
    nb_par_var[col] = 1   # 1 colonne = elle-même (normalisée)
for col in vars_cat_retenues:
    nb_par_var[col] = sum(1 for c in cols_cat_apres if c.startswith(col))

fig_cols = go.Figure(go.Bar(
    x=list(nb_par_var.keys()),
    y=list(nb_par_var.values()),
    marker=dict(
        color=[C['primaire']]*len(vars_num_retenues) +
              [C['violet']]*len(vars_cat_retenues),
        line=dict(color=C['fond'], width=2)
    ),
    text=[str(v) for v in nb_par_var.values()],
    textposition='outside',
    textfont=dict(size=12),
    hovertemplate='%{x}<br>%{y} colonne(s) après encodage<extra></extra>',
))
fig_cols.update_layout(**LAYOUT_BASE,
    title=dict(text='Nombre de colonnes générées par variable après encodage',
               font=dict(size=14), x=0.5),
    yaxis=dict(**axe('Nombre de colonnes'), dtick=1),
    xaxis=dict(**axe(), tickangle=15),
    height=340, margin=dict(t=55, b=80, l=55, r=30),
    showlegend=False)

# ── Fig 4 : Proportions train / test ──────────────────────────────────────────
fig_split = go.Figure()
labels_split = ['Train (80%)', 'Test (20%)']
tailles      = [len(X_train), len(X_test)]
fig_split.add_trace(go.Pie(
    values=tailles, labels=labels_split,
    hole=0.55,
    marker=dict(colors=[C['primaire'], C['violet']],
                line=dict(color=C['fond'], width=3)),
    textinfo='label+percent',
    textfont=dict(size=13, color='white'),
    pull=[0, 0.05],
    hovertemplate='%{label}<br>%{value:,} employés<extra></extra>',
))
fig_split.add_annotation(
    text=f"<b>{n:,}</b><br><span style='font-size:11px'>total</span>",
    x=0.5, y=0.5, showarrow=False,
    font=dict(size=18, color=C['texte'])
)
fig_split.update_layout(**LAYOUT_BASE,
    title=dict(text='Répartition Train / Test', font=dict(size=14), x=0.5),
    height=320, margin=dict(t=50, b=20, l=20, r=20))

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
<title>Étape 3 — Prétraitement</title>
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
.pipe{{background:{C['carte']};border:1px solid {C['bord']};border-radius:12px;
       padding:24px 28px;font-family:'Courier New',monospace;font-size:13px;
       color:{C['texte2']};line-height:1.8;}}
.pipe .hl{{color:{C['reste']};font-weight:700;}}
.pipe .kw{{color:{C['primaire']};}}
.pipe .cm{{color:{C['texte2']};opacity:.6;}}
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
@media(max-width:900px){{.kpi-row,.g2,.ins{{grid-template-columns:1fr 1fr;}}}}
</style>
</head>
<body>
<div class="hdr">
  <div class="badge">Étape 3 / 10 — Prétraitement & Pipeline</div>
  <h1>HR Turnover <span>Analytics</span> — Prétraitement</h1>
  <p>Encodage, normalisation, split train/test et équilibrage SMOTE</p>
  <div class="meta">
    <div class="mi">📊 <strong>{len(FEATURES)}</strong> features en entrée</div>
    <div class="mi">🔤 <strong>{len(toutes_cols)}</strong> colonnes après encodage</div>
    <div class="mi">✂️ <strong>{len(X_train_smote):,}</strong> exemples train (SMOTE)</div>
    <div class="mi">🧪 <strong>{len(X_test_proc):,}</strong> exemples test</div>
    <div class="mi">📅 UCAO 2025-2026 | M. Aidara</div>
  </div>
</div>

<div class="kpi-row">
  <div class="kpi" style="--a:{C['primaire']}">
    <div class="ki">📊</div><div class="kv">{len(FEATURES)}</div>
    <div class="kl">Features en entrée</div>
    <div class="ks">{len(vars_num_retenues)} num. + {len(vars_cat_retenues)} cat.</div>
  </div>
  <div class="kpi" style="--a:{C['violet']}">
    <div class="ki">🔤</div><div class="kv">{len(toutes_cols)}</div>
    <div class="kl">Colonnes après encodage</div>
    <div class="ks">OneHotEncoder + StandardScaler</div>
  </div>
  <div class="kpi" style="--a:{C['reste']}">
    <div class="ki">✂️</div><div class="kv">{len(X_train_smote):,}</div>
    <div class="kl">Exemples train</div>
    <div class="ks">Après SMOTE — 50% / 50%</div>
  </div>
  <div class="kpi" style="--a:{C['orange']}">
    <div class="ki">🧪</div><div class="kv">{len(X_test_proc):,}</div>
    <div class="kl">Exemples test</div>
    <div class="ks">Données réelles non modifiées</div>
  </div>
</div>

<div class="sec" style="margin-top:22px">
  <div class="st"><h2>Pipeline sklearn — Architecture</h2><span class="sb">Code</span></div>
  <div class="pipe">
<span class="cm"># Construction du pipeline de prétraitement</span>
preprocesseur = <span class="kw">ColumnTransformer</span>([
    (<span class="hl">'numerique'</span>,  <span class="kw">StandardScaler()</span>,           {vars_num_retenues}),
    (<span class="hl">'categoriel'</span>, <span class="kw">OneHotEncoder(drop='first')</span>,  {vars_cat_retenues}),
])

<span class="cm"># Split stratifié 80% / 20%</span>
X_train, X_test, y_train, y_test = <span class="kw">train_test_split</span>(X, y,
    test_size=0.20, random_state=42, stratify=y)

<span class="cm"># Ajustement sur train uniquement (évite data leakage)</span>
X_train_proc = preprocesseur.<span class="kw">fit_transform</span>(X_train)   <span class="cm"># fit + transform</span>
X_test_proc  = preprocesseur.<span class="kw">transform</span>(X_test)        <span class="cm"># transform uniquement</span>

<span class="cm"># Équilibrage des classes sur train uniquement</span>
smote = <span class="kw">SMOTE</span>(random_state=42)
X_train_smote, y_train_smote = smote.<span class="kw">fit_resample</span>(X_train_proc, y_train)
  </div>
</div>

<div class="sec" style="margin-top:18px">
  <div class="st"><h2>Équilibrage SMOTE & Split Train/Test</h2><span class="sb">Données</span></div>
  <div class="g2">
    <div class="chart full">
      <span class="ct" style="background:{C['reste']}22;color:{C['reste']}">SMOTE</span>
      {to_html(fig_smote,'smote')}
    </div>
  </div>
  <div class="g2" style="margin-top:14px">
    <div class="chart">
      <span class="ct" style="background:{C['primaire']}22;color:{C['primaire']}">Split</span>
      {to_html(fig_split,'split')}
    </div>
    <div class="chart">
      <span class="ct" style="background:{C['violet']}22;color:{C['violet']}">Colonnes encodées</span>
      {to_html(fig_cols,'cols')}
    </div>
  </div>
</div>

<div class="sec" style="margin-top:18px">
  <div class="st"><h2>Effet de la normalisation StandardScaler</h2><span class="sb">Transformation</span></div>
  <div class="chart full">
    <span class="ct" style="background:{C['orange']}22;color:{C['orange']}">Avant / Après</span>
    {to_html(fig_scale,'scale')}
  </div>
</div>

<div class="sec" style="margin-top:22px">
  <div class="st"><h2>Insights clés — Étape 3</h2><span class="sb">Bilan</span></div>
  <div class="ins">
    <div class="in" style="--a:{C['primaire']}">
      <div class="it">Pipeline sklearn</div>
      <div class="iv">ColumnTransformer</div>
      <div class="ix">Architecture robuste : StandardScaler pour {len(vars_num_retenues)} variables numériques + OneHotEncoder pour {len(vars_cat_retenues)} catégorielles.</div>
    </div>
    <div class="in" style="--a:{C['violet']}">
      <div class="it">Encodage</div>
      <div class="iv">{len(FEATURES)} → {len(toutes_cols)} cols</div>
      <div class="ix">OneHotEncoder avec drop='first' transforme {len(vars_cat_retenues)} variables catégorielles en {len(cols_cat_apres)} colonnes binaires.</div>
    </div>
    <div class="in" style="--a:{C['reste']}">
      <div class="it">SMOTE</div>
      <div class="iv">50% / 50%</div>
      <div class="ix">{(y_train==1).sum():,} exemples réels → {(y_train_smote==1).sum():,} après SMOTE. {len(X_train_smote)-len(X_train_proc):,} exemples synthétiques créés.</div>
    </div>
    <div class="in" style="--a:{C['orange']}">
      <div class="it">Data leakage</div>
      <div class="iv">Évité ✅</div>
      <div class="ix">fit() uniquement sur train. transform() sur train ET test avec les mêmes paramètres calculés sur train.</div>
    </div>
    <div class="in" style="--a:{C['part']}">
      <div class="it">Reproductibilité</div>
      <div class="iv">random_state=42</div>
      <div class="ix">Train/test split et SMOTE fixés avec random_state=42. Résultats identiques à chaque exécution.</div>
    </div>
    <div class="in" style="--a:{C['accent']}">
      <div class="it">Prochaine étape</div>
      <div class="iv">Étape 4</div>
      <div class="ix">Entraînement et comparaison des modèles ML : Logistic Regression, Random Forest, XGBoost.</div>
    </div>
  </div>
</div>

<div class="footer">
  <strong>HR Turnover Analytics</strong> — Étape 3 : Prétraitement & Pipeline
  <span>Seye Kiné | Bindia Adeline Thiara | M. Aidara | UCAO 2025-2026</span>
</div>
</body>
</html>"""

with open('etape3_rapport.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"""
  {'═'*66}
  ✅  ÉTAPE 3 TERMINÉE AVEC SUCCÈS
  {'─'*66}
   💾  pipeline_data.pkl    → enrichi et transmis à l'Étape 4
   📊  etape3_rapport.html → rapport HTML interactif
  {'─'*66}
  ➡️   Lancer :  python etape4_modeles.py
  {'═'*66}
""")
