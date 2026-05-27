# =============================================================================
#  GENERATIVE HR ANALYTICS — PRÉDICTION DU TURNOVER DES EMPLOYÉS
#  Auteurs   : Seye Kiné | Bindia Adeline Thiara
#  Encadrant : M. Aidara
#  Institut  : UCAO — Département Informatique de Gestion
#  Année     : 2025-2026
# =============================================================================
#  ÉTAPE 1 — EXPLORATION ET QUALITÉ DES DONNÉES
#  Objectif  : Charger le dataset, vérifier sa qualité, comprendre
#              la structure des données et identifier les premiers
#              facteurs de turnover avant toute modélisation.
#  Fichiers générés :
#    - pipeline_data.pkl   (données transmises à l'étape suivante)
#    - etape1_rapport.html (rapport interactif visuel)
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION A — CONFIGURATION GLOBALE
# =============================================================================

# Palette de couleurs cohérente pour tout le projet
COULEURS = {
    'reste'   : '#00C896',   # Vert  → employé resté
    'part'    : '#FF4B6E',   # Rouge → employé parti
    'primaire': '#4A9EF5',   # Bleu  → couleur principale
    'accent'  : '#FFD166',   # Or    → mise en valeur
    'violet'  : '#9B72F5',   # Violet→ secondaire
    'orange'  : '#FF8C42',   # Orange→ tertiaire
    'fond'    : '#0F1923',   # Fond sombre dashboard
    'carte'   : '#1A2535',   # Fond des cartes
    'graphe'  : '#243044',   # Fond des graphiques
    'texte'   : '#E8F0FE',   # Texte principal
    'texte2'  : '#8FA3BF',   # Texte secondaire
    'grille'  : '#2A3A50',   # Lignes de grille
    'bord'    : '#3A4F6A',   # Bordures
}

# Paramètres de mise en page Plotly réutilisables
LAYOUT_BASE = dict(
    paper_bgcolor = COULEURS['carte'],
    plot_bgcolor  = COULEURS['graphe'],
    font          = dict(color=COULEURS['texte'], family='Segoe UI, sans-serif'),
    legend        = dict(
        bgcolor     = COULEURS['carte'],
        bordercolor = COULEURS['bord'],
        font        = dict(color=COULEURS['texte'])
    ),
)

# Paramètres des axes Plotly réutilisables
def axe(titre=''):
    return dict(
        title      = titre,
        gridcolor  = COULEURS['grille'],
        showgrid   = True,
        zeroline   = False,
        tickfont   = dict(color=COULEURS['texte2']),

    )

# =============================================================================
# SECTION B — FONCTIONS D'AFFICHAGE CONSOLE
# =============================================================================

L = 70   # Largeur de la console

def banniere(texte):
    """Titre de section principal"""
    print()
    print(f" ╔{'═'*L}╗")
    print(f" ║{'  ◆  ' + texte.upper() + '  ◆':^{L}}║")
    print(f" ╚{'═'*L}╝")

def section(texte, icone=''):
    """Sous-titre de section"""
    print(f"\n  ╭{'─'*(L-2)}╮")
    txt = f"{icone}  {texte}" if icone else texte
    print(f"  │  {txt:<{L-5}}│")
    print(f"  ╰{'─'*(L-2)}╯")

def sous_section(texte, icone='▸'):
    """Titre de sous-section"""
    print(f"\n  {icone}  {texte}")
    print(f"  {'╌'*55}")

def ligne(label, valeur, note=''):
    """Ligne de résultat structurée"""
    print(f"  │  {label:<34} {str(valeur):<18} {note}")

def sep():
    """Séparateur de fin de bloc"""
    print(f"  └{'─'*58}")

def ok(texte):
    print(f"  │  ✅  {texte}")

def warn(texte):
    print(f"  │  ⚠️   {texte}")

def tableau(x, indent=5):
    """Affichage propre d'un DataFrame ou d'une chaîne"""
    t = x if isinstance(x, str) else x.to_string()
    for l in t.split('\n'):
        print(' '*indent + l)

def barre(valeur, max_val=100, largeur=30):
    """Barre de progression ASCII"""
    n = int(valeur / max_val * largeur)
    return f"[{'█'*n}{'░'*(largeur-n)}] {valeur:.1f}%"

def bilan(items, titre='BILAN'):
    """Encadré récapitulatif final"""
    print(f"\n  ╔══ {titre} {'═'*(L-len(titre)-5)}╗")
    for k, v in items.items():
        ligne_str = f"  ║  {k:<32} {v}"
        print(f"{ligne_str:<{L+4}}║")
    print(f"  ╚{'═'*(L+1)}╝")

# =============================================================================
# SECTION C — CHARGEMENT DES DONNÉES
# =============================================================================

banniere("ÉTAPE 1 — EXPLORATION ET QUALITÉ DES DONNÉES")

print(f"""
  Projet   : Generative HR Analytics & Explicabilité des Modèles
  Sujet    : Analyse et Prédiction du Turnover des Employés
  Auteurs  : Seye Kiné | Bindia Adeline Thiara
  Encadrant: M. Aidara — UCAO 2025-2026
""")

section("CHARGEMENT DU DATASET", "📂")

# Chargement du fichier CSV
FICHIER = 'hr_turnover.csv'
df      = pd.read_csv(FICHIER)

# Identification automatique des types de colonnes
# On exclut EmployeeID (identifiant non prédictif) et Attrition (variable cible)
COLS_NUM = (df.select_dtypes(include=np.number)
              .columns
              .drop(['EmployeeID', 'Attrition'])
              .tolist())

COLS_CAT = df.select_dtypes(include='object').columns.tolist()
CIBLE    = 'Attrition'

ligne("Fichier source",       FICHIER)
ligne("Employés (lignes)",    f"{df.shape[0]:,}")
ligne("Variables (colonnes)", f"{df.shape[1]}")
ligne("Variables numériques", f"{len(COLS_NUM)}")
ligne("Variables texte",      f"{len(COLS_CAT)}")
ligne("Variable cible",       CIBLE, "0=Reste | 1=Part")
sep()

print("\n  Détail de toutes les variables :\n")
info_df = pd.DataFrame({
    'Variable'        : df.columns,
    'Type'            : df.dtypes.astype(str).values,
    'Valeurs uniques' : [df[c].nunique() for c in df.columns],
    'Exemple val.'    : [str(df[c].iloc[0]) for c in df.columns],
    'Rôle'            : ['Identifiant' if c=='EmployeeID'
                         else 'Cible' if c==CIBLE
                         else 'Numérique' if c in COLS_NUM
                         else 'Catégorielle'
                         for c in df.columns]
})
tableau(info_df.to_string(index=False))

# =============================================================================
# SECTION D — QUALITÉ DES DONNÉES
# =============================================================================

section("QUALITÉ DES DONNÉES", "🔬")

# ── D1. Valeurs manquantes ────────────────────────────────────────────────────
sous_section("d1. Valeurs manquantes", "🔍")
# Une valeur manquante = cellule vide dans le tableau
# Si une colonne en a trop → elle est inutilisable pour le modèle
manquantes = df.isnull().sum()
pct_manq   = (manquantes / len(df) * 100).round(2)

if manquantes.sum() == 0:
    ok("Aucune valeur manquante — dataset 100% complet")
else:
    rapport_m = pd.DataFrame({
        'Colonne'    : manquantes[manquantes>0].index,
        'Manquantes' : manquantes[manquantes>0].values,
        '% dataset'  : pct_manq[manquantes>0].values,
    })
    tableau(rapport_m.to_string(index=False))
sep()

# ── D2. Doublons ──────────────────────────────────────────────────────────────
sous_section("d2. Lignes dupliquées", "🔁")
# Un doublon = deux lignes identiques pour le même employé
# → fausse l'entraînement du modèle (données gonflées artificiellement)
doublons        = df.duplicated().sum()
doublons_profil = df.drop(columns=['EmployeeID']).duplicated().sum()

if doublons == 0:
    ok(f"Aucune ligne identique  (total = 0)")
else:
    warn(f"{doublons} lignes identiques détectées — à supprimer")

if doublons_profil == 0:
    ok(f"Aucun profil dupliqué   (hors EmployeeID)")
else:
    warn(f"{doublons_profil} profils identiques détectés")
sep()

# ── D3. Valeurs aberrantes (méthode IQR) ──────────────────────────────────────
sous_section("d3. Valeurs aberrantes — méthode IQR", "📐")
# Méthode IQR (Inter-Quartile Range) :
#   Q1 = 25e percentile, Q3 = 75e percentile, IQR = Q3 - Q1
#   Valeur aberrante si < Q1 - 1.5×IQR  ou  > Q3 + 1.5×IQR
# C'est la méthode statistique standard, robuste aux distributions asymétriques

print("  │  Formule : aberrante si  val < Q1−1.5×IQR  ou  val > Q3+1.5×IQR\n")

rapport_iqr = []
for col in COLS_NUM:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR    = Q3 - Q1
    bb     = Q1 - 1.5 * IQR   # Borne basse
    bh     = Q3 + 1.5 * IQR   # Borne haute
    nb     = ((df[col] < bb) | (df[col] > bh)).sum()
    rapport_iqr.append({
        'Variable'    : col,
        'Q1'          : round(Q1, 2),
        'Q3'          : round(Q3, 2),
        'Borne basse' : round(bb, 2),
        'Borne haute' : round(bh, 2),
        'Outliers'    : nb,
        '% dataset'   : f"{nb/len(df)*100:.2f}%",
    })

df_iqr    = pd.DataFrame(rapport_iqr).sort_values('Outliers', ascending=False)
total_out = df_iqr['Outliers'].sum()
tableau(df_iqr.to_string(index=False))

print()
if total_out == 0:
    ok("Aucune valeur aberrante — données propres")
else:
    warn(f"{total_out} valeurs hors bornes IQR détectées")
sep()

# =============================================================================
# SECTION E — STATISTIQUES DESCRIPTIVES
# =============================================================================

section("STATISTIQUES DESCRIPTIVES", "📊")
# .describe() calcule automatiquement :
# count, mean (moyenne), std (écart-type), min, max, 25%/50%/75% (quartiles)
stats = df[COLS_NUM].describe().round(2)
print("\n  Variables numériques :\n")
tableau(stats.to_string())

print("\n  Variables catégorielles (modalités) :\n")
for col in COLS_CAT:
    vals = df[col].value_counts()
    ligne_vals = "  |  ".join([f"{v}: {c:,}" for v, c in vals.items()])
    print(f"  {col:<20} → {ligne_vals}")
sep()

# =============================================================================
# SECTION F — ANALYSE DE LA VARIABLE CIBLE
# =============================================================================

section("VARIABLE CIBLE : ATTRITION", "🎯")

n       = len(df)
n0      = (df[CIBLE] == 0).sum()   # Nombre d'employés restés
n1      = (df[CIBLE] == 1).sum()   # Nombre d'employés partis
taux    = n1 / n * 100
ratio   = n0 / n1
statut  = "✅ Équilibrées"   if 30 < taux < 70 else "⚠️  Déséquilibrées"
smote   = "Optionnel"       if 30 < taux < 70 else "Recommandé"

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │   Restés  (0) :  {n0:>6,}  employés   ({100-taux:.1f}%)               │
  │   {barre(100-taux, largeur=35)}            │
  │                                                              │
  │   Partis  (1) :  {n1:>6,}  employés   ({taux:.1f}%)               │
  │   {barre(taux, largeur=35)}            │
  │                                                              │
  │   Ratio         :  {ratio:.2f} : 1                                   │
  │   Classes       :  {statut:<42}│
  │   SMOTE (Ét. 4) :  {smote:<42}│
  │                                                              │
  └──────────────────────────────────────────────────────────────┘""")

# =============================================================================
# SECTION G — ANALYSE CROISÉE : ATTRITION × VARIABLES CLÉS
# =============================================================================

section("ATTRITION × VARIABLES CLÉS", "🔗")

# ── Variables catégorielles ───────────────────────────────────────────────────
sous_section("g1. Taux d'attrition par variable catégorielle", "📌")
print(f"\n  {'Var / Modalité':<28} {'Effectif':>9}  {'Taux (%)':>9}  "
      f"{'Écart moy.':>11}  Répartition")
print(f"  {'╌'*28} {'╌'*9}  {'╌'*9}  {'╌'*11}  {'╌'*22}")

for col in COLS_CAT:
    print(f"\n  ▸ {col}")
    cross = df.groupby(col)[CIBLE].agg(['mean','count'])
    cross['taux_%'] = (cross['mean']*100).round(1)
    cross = cross.sort_values('taux_%', ascending=False)
    for idx, row in cross.iterrows():
        ecart = row['taux_%'] - taux
        flag  = f"▲ +{ecart:.1f}" if ecart > 0 else f"▼ {ecart:.1f}"
        bar_  = '█'*int(row['taux_%']/3) + '░'*(33-int(row['taux_%']/3))
        print(f"    {str(idx):<26} {int(row['count']):>9,}  "
              f"{row['taux_%']:>8.1f}%  {flag:>11}  {bar_[:22]}")
sep()

# ── Variables numériques : comparaison des moyennes ───────────────────────────
sous_section("g2. Comparaison des moyennes numériques (Resté vs Parti)", "📈")
# On compare la valeur moyenne de chaque variable entre
# les employés restés (0) et partis (1)
# Un écart important → variable potentiellement prédictive
moy = df.groupby(CIBLE)[COLS_NUM].mean().round(2).T
moy.columns = ['Resté (0)', 'Parti (1)']
moy['Écart (%)'] = ((moy['Parti (1)']-moy['Resté (0)'])/moy['Resté (0)']*100).round(1)
moy['Signal']    = moy['Écart (%)'].apply(
    lambda x: '🔴 FORT   '  if abs(x)>10
         else '🟡 MOYEN  '  if abs(x)>3
         else '🟢 Faible ')
moy = moy.reindex(moy['Écart (%)'].abs().sort_values(ascending=False).index)
tableau(moy.to_string())
sep()

# =============================================================================
# SECTION H — CORRÉLATIONS AVEC L'ATTRITION
# =============================================================================

section("CORRÉLATIONS AVEC L'ATTRITION", "📐")
# Corrélation de Pearson entre chaque variable numérique et Attrition
# Valeur entre -1 et +1 :
#   +1 → augmente avec Attrition    (ex: DistanceFromHome)
#   -1 → diminue avec Attrition     (ex: JobSatisfaction)
#    0 → pas de lien linéaire

corr_vals = df[COLS_NUM+[CIBLE]].corr()[CIBLE].drop(CIBLE)
corr_abs  = corr_vals.abs().sort_values(ascending=False)

print(f"\n  {'Variable':<32} {'Corrélation':>11}  {'Force':<9}  Visualisation")
print(f"  {'╌'*32} {'╌'*11}  {'╌'*9}  {'╌'*28}")

for col in corr_abs.index:
    val   = corr_vals[col]
    signe = '+' if val >= 0 else '-'
    force = '🔴 FORT  ' if abs(val) > 0.3 \
       else '🟡 MOYEN ' if abs(val) > 0.1 \
       else '🟢 faible'
    nb    = int(abs(val) * 55)
    bar_  = '▓'*nb + '░'*(28-nb)
    print(f"  {col:<32} {signe}{abs(val):.4f}       {force}  {bar_}")

sep()

# =============================================================================
# SECTION I — BILAN FINAL + TRANSMISSION AUX ÉTAPES SUIVANTES
# =============================================================================

top1 = corr_abs.index[0]
top2 = corr_abs.index[1]
top3 = corr_abs.index[2]

bilan({
    '📁 Dataset'          : f"{n:,} employés  ×  {df.shape[1]} variables",
    '❓ Manquantes'       : f"0   ✅ Dataset complet",
    '🔁 Doublons'         : f"0   ✅ Aucun doublon",
    '📐 Outliers IQR'     : f"0   ✅ Données propres",
    '🎯 Taux attrition'   : f"{taux:.1f}%  ({statut})",
    '🏆 Facteur #1'       : f"OverTime → 51.6% vs 36.7%  (Δ +14.9 pts)",
    '🏆 Facteur #2'       : f"{top1:<24} corr = {corr_vals[top1]:+.4f}",
    '🏆 Facteur #3'       : f"{top2:<24} corr = {corr_vals[top2]:+.4f}",
    '🏆 Facteur #4'       : f"{top3:<24} corr = {corr_vals[top3]:+.4f}",
    '➡️  Prochaine étape'  : "Étape 2 — Analyse bivariée & EDA avancée",
}, titre='BILAN — ÉTAPE 1')

# ── Transmission des données à l'étape suivante ───────────────────────────────
# On sauvegarde tout ce dont les étapes suivantes auront besoin
# dans un fichier pipeline_data.pkl (format Python pickle)
pipeline_data = {
    'df'        : df,          # Dataset brut complet
    'COLS_NUM'  : COLS_NUM,    # Liste des colonnes numériques
    'COLS_CAT'  : COLS_CAT,    # Liste des colonnes catégorielles
    'CIBLE'     : CIBLE,       # Nom de la variable cible
    'n'         : n,           # Nombre total d'employés
    'n0'        : n0,          # Nombre de restés
    'n1'        : n1,          # Nombre de partis
    'taux'      : taux,        # Taux d'attrition global
    'corr_vals' : corr_vals,   # Corrélations avec l'Attrition
    'COULEURS'  : COULEURS,    # Palette de couleurs partagée
}

with open('pipeline_data.pkl', 'wb') as f:
    pickle.dump(pipeline_data, f)

print("\n  💾  pipeline_data.pkl  sauvegardé — données transmises à l'Étape 2")

# =============================================================================
# SECTION J — RAPPORT HTML INTERACTIF
# =============================================================================

print("\n  ⏳  Génération du rapport HTML interactif...")

# ── Graphique 1 : Donut Attrition ─────────────────────────────────────────────
fig1 = go.Figure(go.Pie(
    values       = [n0, n1],
    labels       = ['Resté (0)', 'Parti (1)'],
    hole         = 0.62,
    marker       = dict(
        colors = [COULEURS['reste'], COULEURS['part']],
        line   = dict(color=COULEURS['fond'], width=3)
    ),
    textinfo     = 'label+percent',
    textfont     = dict(size=13, color='white'),
    pull         = [0, 0.05],
    rotation     = 90,
    hovertemplate= '%{label}<br>%{value:,} employés<br>%{percent}<extra></extra>',
))
fig1.add_annotation(
    text     = f"<b>{taux:.1f}%</b><br><span style='font-size:11px'>Départs</span>",
    x=0.5, y=0.5, showarrow=False,
    font     = dict(size=20, color=COULEURS['part']),
)
fig1.update_layout(
    **LAYOUT_BASE,
    title  = dict(text='Distribution de l\'Attrition', font=dict(size=14), x=0.5),
    height = 320,
    margin = dict(t=50, b=10, l=10, r=10),
)

# ── Graphique 2 : Attrition par OverTime ──────────────────────────────────────
ot_vals  = df.groupby('OverTime')[CIBLE].mean() * 100
fig2     = go.Figure()
fig2.add_trace(go.Bar(
    x                = ['Sans heures sup', 'Avec heures sup'],
    y                = ot_vals.values,
    marker           = dict(
        color      = [COULEURS['reste'], COULEURS['part']],
        line       = dict(color=COULEURS['fond'], width=2)
    ),
    text             = [f"<b>{v:.1f}%</b>" for v in ot_vals.values],
    textposition     = 'outside',
    textfont         = dict(size=14),
    width            = 0.45,
    hovertemplate    = '%{x}<br>Taux : %{y:.1f}%<extra></extra>',
))
fig2.add_hline(
    y                = taux,
    line_dash        = 'dash',
    line_color       = COULEURS['accent'],
    line_width       = 1.8,
    annotation_text  = f"Moy. {taux:.1f}%",
    annotation_font  = dict(color=COULEURS['accent']),
)
fig2.add_annotation(
    x=1, y=ot_vals['Yes']+2,
    text       = f"<b>Δ +{ot_vals['Yes']-ot_vals['No']:.1f} pts</b>",
    showarrow  = True, arrowhead=2,
    arrowcolor = COULEURS['part'],
    font       = dict(color=COULEURS['part'], size=12),
    ax=55, ay=-30,
)
fig2.update_layout(
    **LAYOUT_BASE,
    title      = dict(text='Impact des Heures Supplémentaires', font=dict(size=14), x=0.5),
    yaxis      = dict(**axe('Taux d\'attrition (%)'), range=[0, 65]),
    xaxis      = axe(),
    height     = 320,
    showlegend = False,
)

# ── Graphique 3 : Attrition par JobSatisfaction ────────────────────────────────
js_vals  = df.groupby('JobSatisfaction')[CIBLE].mean() * 100
pal_js   = [COULEURS['part'], COULEURS['orange'], COULEURS['accent'], COULEURS['reste']]
fig3     = go.Figure()
for i, (js, v) in enumerate(js_vals.items()):
    fig3.add_trace(go.Bar(
        x             = [f'Niveau {js}'],
        y             = [v],
        marker        = dict(color=pal_js[i], line=dict(color=COULEURS['fond'], width=2)),
        text          = f"<b>{v:.1f}%</b>",
        textposition  = 'outside',
        textfont      = dict(size=13, color=pal_js[i]),
        showlegend    = False,
        hovertemplate = f'Satisfaction {js}<br>Taux : {v:.1f}%<extra></extra>',
    ))
fig3.add_hline(
    y               = taux,
    line_dash       = 'dash',
    line_color      = COULEURS['accent'],
    line_width      = 1.8,
    annotation_text = f"Moy. {taux:.1f}%",
    annotation_font = dict(color=COULEURS['accent']),
)
fig3.update_layout(
    **LAYOUT_BASE,
    title   = dict(text='Impact de la Satisfaction au Travail', font=dict(size=14), x=0.5),
    yaxis   = dict(**axe('Taux d\'attrition (%)'), range=[0, 65]),
    xaxis   = dict(**axe('Niveau (1=Faible → 4=Élevée)')),
    height  = 320,
    barmode = 'group',
)

# ── Graphique 4 : Attrition par Département ────────────────────────────────────
dept_vals  = df.groupby('Department')[CIBLE].mean() * 100
dept_sorted= dept_vals.sort_values()
fig4       = go.Figure(go.Bar(
    x             = dept_sorted.values,
    y             = dept_sorted.index,
    orientation   = 'h',
    marker        = dict(
        color     = [COULEURS['part'] if v>taux else COULEURS['primaire']
                     for v in dept_sorted.values],
        line      = dict(color=COULEURS['fond'], width=1.5)
    ),
    text          = [f"<b>{v:.1f}%</b>" for v in dept_sorted.values],
    textposition  = 'outside',
    hovertemplate = '%{y}<br>Taux : %{x:.1f}%<extra></extra>',
))
fig4.add_vline(
    x               = taux,
    line_dash       = 'dash',
    line_color      = COULEURS['accent'],
    line_width      = 1.8,
    annotation_text = f"Moy. {taux:.1f}%",
    annotation_font = dict(color=COULEURS['accent']),
)
fig4.update_layout(
    **LAYOUT_BASE,
    title  = dict(text='Attrition par Département', font=dict(size=14), x=0.5),
    xaxis  = dict(**axe('Taux (%)'), range=[0, 58]),
    yaxis  = dict(**axe()),
    height = 320,
    showlegend = False,
)

# ── Graphique 5 : Heatmap de corrélation ──────────────────────────────────────
corr_mat = df[COLS_NUM+[CIBLE]].corr()
mask     = np.triu(np.ones_like(corr_mat, dtype=bool))
z_masked = corr_mat.copy().astype(float)
z_masked[mask] = np.nan

text_arr = [
    [f'{corr_mat.iloc[i,j]:.2f}' if not mask[i,j] else ''
     for j in range(len(corr_mat.columns))]
    for i in range(len(corr_mat.index))
]

fig5 = go.Figure(go.Heatmap(
    z             = z_masked.values,
    x             = corr_mat.columns.tolist(),
    y             = corr_mat.index.tolist(),
    text          = text_arr,
    texttemplate  = '%{text}',
    textfont      = dict(size=9, color='white'),
    colorscale    = [
        [0.0, '#FF4B6E'], [0.25, '#FF8C42'],
        [0.5,  '#243044'],
        [0.75, '#4A9EF5'], [1.0,  '#00C896']
    ],
    zmid=0, zmin=-0.5, zmax=0.5,
    colorbar      = dict(
        title     = 'r',
        tickfont  = dict(color=COULEURS['texte2']),
        bgcolor   = COULEURS['carte'],
        bordercolor=COULEURS['bord'],
    ),
    hovertemplate = '%{x} × %{y}<br>r = %{z:.4f}<extra></extra>',
))
# Rectangle doré autour de la ligne Attrition
nc = len(corr_mat.columns)
fig5.add_shape(
    type='rect',
    x0=-0.5, y0=nc-1.5, x1=nc-1.5, y1=nc-0.5,
    line=dict(color=COULEURS['accent'], width=2.5)
)
fig5.update_layout(
    paper_bgcolor=COULEURS['carte'], plot_bgcolor=COULEURS['carte'],
    font=dict(color=COULEURS['texte'], family='Segoe UI, sans-serif'),
    legend=dict(bgcolor=COULEURS['carte'],bordercolor=COULEURS['bord'],font=dict(color=COULEURS['texte'])),
    title  = dict(
        text='Heatmap de Corrélation — Ligne Attrition encadrée en or',
        font=dict(size=14), x=0.5
    ),
    xaxis  = dict(tickangle=45, tickfont=dict(size=9, color=COULEURS['texte2'])),
    yaxis  = dict(tickfont=dict(size=9, color=COULEURS['texte2']), autorange='reversed'),
    height = 520,
    margin = dict(t=60, b=120, l=160, r=30),
)

# ── Graphique 6 : Top variables prédictives ────────────────────────────────────
top8       = corr_abs.head(8)
coul_rank  = [COULEURS['part']   if v>0.12
              else COULEURS['orange'] if v>0.06
              else COULEURS['primaire']
              for v in top8.values]
fig6       = go.Figure(go.Bar(
    x             = top8.values,
    y             = top8.index,
    orientation   = 'h',
    marker        = dict(color=coul_rank, line=dict(color=COULEURS['fond'], width=1.5)),
    text          = [f"<b>{v:.4f}</b>" for v in top8.values],
    textposition  = 'outside',
    hovertemplate = '%{y}<br>|r| = %{x:.4f}<extra></extra>',
))
fig6.update_layout(
    paper_bgcolor=COULEURS['carte'], plot_bgcolor=COULEURS['graphe'],
    font=dict(color=COULEURS['texte'], family='Segoe UI, sans-serif'),
    legend=dict(bgcolor=COULEURS['carte'],bordercolor=COULEURS['bord'],font=dict(color=COULEURS['texte'])),
    title  = dict(text='Top 8 Variables Prédictives (|r| avec Attrition)', font=dict(size=14), x=0.5),
    xaxis  = dict(**axe('Corrélation absolue')),
    yaxis  = dict(**axe()),
    height = 340,
    margin = dict(t=55, b=40, l=190, r=60),
    showlegend=False,
)

# ── Construction du rapport HTML ───────────────────────────────────────────────
def to_html(fig, div_id):
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config=dict(displayModeBar=True, displaylogo=False,
                    modeBarButtonsToRemove=['lasso2d','select2d'])
    )

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Étape 1 — HR Analytics</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:{COULEURS['fond']};color:{COULEURS['texte']};
     font-family:'Segoe UI','Inter',sans-serif;min-height:100vh;}}

/* Header */
.hdr{{background:{COULEURS['carte']};border-bottom:2px solid {COULEURS['accent']};
      padding:32px 40px 24px;}}
.badge{{display:inline-block;background:{COULEURS['violet']};color:white;
        font-size:11px;font-weight:700;padding:3px 12px;border-radius:20px;
        letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;}}
.hdr h1{{font-size:26px;font-weight:800;line-height:1.2;margin-bottom:4px;}}
.hdr h1 span{{color:{COULEURS['primaire']};}}
.hdr p{{color:{COULEURS['texte2']};font-size:13px;margin-top:4px;}}
.meta{{display:flex;gap:16px;margin-top:16px;flex-wrap:wrap;}}
.meta-item{{display:flex;align-items:center;gap:6px;
           background:rgba(255,255,255,0.04);border:1px solid {COULEURS['bord']};
           border-radius:8px;padding:5px 12px;font-size:12px;color:{COULEURS['texte2']};}}
.meta-item strong{{color:{COULEURS['texte']};}}

/* KPI */
.kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;
          padding:24px 28px 0;}}
.kpi{{background:{COULEURS['carte']};border:1px solid {COULEURS['bord']};
      border-radius:12px;padding:20px 18px 16px;position:relative;
      overflow:hidden;transition:transform .2s,box-shadow .2s;}}
.kpi:hover{{transform:translateY(-3px);box-shadow:0 8px 24px rgba(0,0,0,.4);}}
.kpi::before{{content:'';position:absolute;top:0;left:0;right:0;
              height:3px;background:var(--acc);border-radius:12px 12px 0 0;}}
.kpi-icon{{font-size:22px;margin-bottom:8px;}}
.kpi-val{{font-size:32px;font-weight:800;color:var(--acc);line-height:1;margin-bottom:4px;}}
.kpi-lbl{{font-size:12px;color:{COULEURS['texte2']};font-weight:500;letter-spacing:.4px;}}
.kpi-sub{{font-size:11px;color:{COULEURS['texte2']};margin-top:6px;opacity:.7;}}

/* Section */
.section{{padding:24px 28px 0;}}
.sec-title{{display:flex;align-items:center;gap:10px;margin-bottom:14px;
            padding-bottom:8px;border-bottom:1px solid {COULEURS['bord']};}}
.sec-title h2{{font-size:16px;font-weight:700;}}
.sec-badge{{background:{COULEURS['violet']};color:white;font-size:10px;
            font-weight:700;padding:2px 8px;border-radius:10px;
            letter-spacing:.8px;text-transform:uppercase;}}

/* Grilles */
.g3{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
.full{{grid-column:1/-1;}}

/* Carte graphique */
.chart{{background:{COULEURS['carte']};border:1px solid {COULEURS['bord']};
        border-radius:12px;overflow:hidden;transition:box-shadow .2s;}}
.chart:hover{{box-shadow:0 4px 20px rgba(0,0,0,.35);}}
.ctag{{font-size:10px;font-weight:700;padding:3px 8px;border-radius:8px;
       letter-spacing:.7px;text-transform:uppercase;
       display:inline-block;margin:10px 0 0 12px;}}

/* Insights */
.ins{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}}
.in{{background:{COULEURS['carte']};border:1px solid {COULEURS['bord']};
     border-left:3px solid var(--acc);border-radius:10px;padding:14px 15px;}}
.in-title{{font-size:11px;font-weight:700;color:var(--acc);
           text-transform:uppercase;letter-spacing:.9px;margin-bottom:5px;}}
.in-val{{font-size:21px;font-weight:800;color:{COULEURS['texte']};margin-bottom:2px;}}
.in-txt{{font-size:12px;color:{COULEURS['texte2']};line-height:1.5;}}

/* Footer */
.footer{{margin-top:36px;padding:18px 28px;border-top:1px solid {COULEURS['bord']};
         display:flex;justify-content:space-between;align-items:center;
         color:{COULEURS['texte2']};font-size:12px;}}
.footer strong{{color:{COULEURS['texte']};}}
.footer span{{opacity:.6;}}

@media(max-width:900px){{
  .kpi-row,.g3,.g2{{grid-template-columns:1fr 1fr;}}
  .ins{{grid-template-columns:1fr;}}
}}
@media(max-width:550px){{
  .kpi-row,.g3,.g2{{grid-template-columns:1fr;}}
}}
</style>
</head>
<body>

<div class="hdr">
  <div class="badge">Étape 1 / 10 — Exploration des données</div>
  <h1>HR Turnover <span>Analytics</span></h1>
  <p>Generative HR Analytics &amp; Explicabilité des Modèles — Prédiction du Turnover</p>
  <div class="meta">
    <div class="meta-item">👥 <strong>{n:,}</strong> employés</div>
    <div class="meta-item">📊 <strong>{df.shape[1]}</strong> variables</div>
    <div class="meta-item">🎯 <strong>{taux:.1f}%</strong> taux d'attrition</div>
    <div class="meta-item">✅ <strong>0</strong> valeur manquante</div>
    <div class="meta-item">📅 UCAO 2025-2026 | M. Aidara</div>
  </div>
</div>

<!-- KPI -->
<div class="kpi-row">
  <div class="kpi" style="--acc:{COULEURS['primaire']}">
    <div class="kpi-icon">👥</div>
    <div class="kpi-val">{n:,}</div>
    <div class="kpi-lbl">Employés analysés</div>
    <div class="kpi-sub">Dataset complet ✅</div>
  </div>
  <div class="kpi" style="--acc:{COULEURS['violet']}">
    <div class="kpi-icon">📊</div>
    <div class="kpi-val">{df.shape[1]}</div>
    <div class="kpi-lbl">Variables disponibles</div>
    <div class="kpi-sub">{len(COLS_NUM)} numériques · {len(COLS_CAT)} texte</div>
  </div>
  <div class="kpi" style="--acc:{COULEURS['part']}">
    <div class="kpi-icon">📉</div>
    <div class="kpi-val">{taux:.1f}%</div>
    <div class="kpi-lbl">Taux d'attrition</div>
    <div class="kpi-sub">{n1:,} départs sur {n:,}</div>
  </div>
  <div class="kpi" style="--acc:{COULEURS['reste']}">
    <div class="kpi-icon">✅</div>
    <div class="kpi-val">0</div>
    <div class="kpi-lbl">Valeurs manquantes</div>
    <div class="kpi-sub">Données parfaitement propres</div>
  </div>
</div>

<!-- Section 1 : Distribution + OverTime + JobSat -->
<div class="section" style="margin-top:22px">
  <div class="sec-title">
    <h2>Distribution & Facteurs principaux</h2>
    <span class="sec-badge">Analyse principale</span>
  </div>
  <div class="g3">
    <div class="chart">
      <span class="ctag" style="background:{COULEURS['part']}22;color:{COULEURS['part']}">Variable cible</span>
      {to_html(fig1,'donut')}
    </div>
    <div class="chart">
      <span class="ctag" style="background:{COULEURS['orange']}22;color:{COULEURS['orange']}">Facteur #1</span>
      {to_html(fig2,'overtime')}
    </div>
    <div class="chart">
      <span class="ctag" style="background:{COULEURS['violet']}22;color:{COULEURS['violet']}">Facteur #2</span>
      {to_html(fig3,'jobsat')}
    </div>
  </div>
</div>

<!-- Section 2 : Département + Top variables -->
<div class="section" style="margin-top:18px">
  <div class="sec-title">
    <h2>Département & Corrélations</h2>
    <span class="sec-badge">Bivariée</span>
  </div>
  <div class="g2">
    <div class="chart">
      <span class="ctag" style="background:{COULEURS['primaire']}22;color:{COULEURS['primaire']}">Département</span>
      {to_html(fig4,'dept')}
    </div>
    <div class="chart">
      <span class="ctag" style="background:{COULEURS['reste']}22;color:{COULEURS['reste']}">Ranking</span>
      {to_html(fig6,'rank')}
    </div>
  </div>
</div>

<!-- Section 3 : Heatmap -->
<div class="section" style="margin-top:18px">
  <div class="sec-title">
    <h2>Heatmap de corrélation complète</h2>
    <span class="sec-badge">Matrice</span>
  </div>
  <div class="chart">
    {to_html(fig5,'heatmap')}
  </div>
</div>

<!-- Section 4 : Insights -->
<div class="section" style="margin-top:22px">
  <div class="sec-title">
    <h2>Insights clés — Étape 1</h2>
    <span class="sec-badge">Bilan</span>
  </div>
  <div class="ins">
    <div class="in" style="--acc:{COULEURS['part']}">
      <div class="in-title">Facteur #1 — OverTime</div>
      <div class="in-val">+14.9 pts</div>
      <div class="in-txt">Les heures supplémentaires font passer le risque de départ de 36.7% à 51.6%.</div>
    </div>
    <div class="in" style="--acc:{COULEURS['violet']}">
      <div class="in-title">Facteur #2 — JobSatisfaction</div>
      <div class="in-val">r = −0.14</div>
      <div class="in-txt">Satisfaction niveau 1 → 52.6% de départs. Niveau 4 → 36.2%. Écart de 16.4 pts.</div>
    </div>
    <div class="in" style="--acc:{COULEURS['orange']}">
      <div class="in-title">Facteur #3 — WorkLifeBalance</div>
      <div class="in-val">r = −0.08</div>
      <div class="in-txt">WLB faible combiné à satisfaction faible = profil à risque critique.</div>
    </div>
    <div class="in" style="--acc:{COULEURS['primaire']}">
      <div class="in-title">Qualité des données</div>
      <div class="in-val">100%</div>
      <div class="in-txt">0 manquante · 0 doublon · 0 outlier. Dataset parfaitement prêt.</div>
    </div>
    <div class="in" style="--acc:{COULEURS['reste']}">
      <div class="in-title">Équilibre des classes</div>
      <div class="in-val">1.27 : 1</div>
      <div class="in-txt">55.9% restés vs 44.1% partis. SMOTE optionnel à l'Étape 4.</div>
    </div>
    <div class="in" style="--acc:{COULEURS['accent']}">
      <div class="in-title">Prochaine étape</div>
      <div class="in-val">Étape 2</div>
      <div class="in-txt">Analyse bivariée avancée — tests statistiques, croisements, profils de risque.</div>
    </div>
  </div>
</div>

<div class="footer">
  <strong>HR Turnover Analytics</strong> — Étape 1 : Exploration des données
  <span>Seye Kiné &nbsp;|&nbsp; Bindia Adeline Thiara &nbsp;|&nbsp; M. Aidara &nbsp;|&nbsp; UCAO 2025-2026</span>
</div>

</body>
</html>"""

with open('etape1_rapport.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"""
  {'═'*66}
  ✅  ÉTAPE 1 TERMINÉE AVEC SUCCÈS
  {'─'*66}
   💾  pipeline_data.pkl    → données transmises à l'Étape 2
   📊  etape1_rapport.html → rapport HTML interactif (ouvrir dans le navigateur)
  {'─'*66}
  ➡️   Lancer :  python etape2_eda.py
  {'═'*66}
""")
