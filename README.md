#  Generative HR Analytics & Explicabilité des Modèles

**Application de prédiction du turnover des employés, combinant Machine Learning, explicabilité et recommandations générées par IA.**

Ce projet, réalisé dans le cadre de mon mémoire de fin d'études (Licence Informatique de Gestion, UCAO Dakar), propose un outil complet d'analytique RH : prédire quels employés risquent de quitter l'entreprise, comprendre pourquoi, et générer des recommandations d'action concrètes.

##  Le problème résolu

Le turnover des employés coûte cher aux entreprises, mais les modèles prédictifs classiques restent souvent des "boîtes noires" difficiles à exploiter par les équipes RH. Ce projet combine performance prédictive et explicabilité, pour que les décideurs RH comprennent non seulement *qui* risque de partir, mais *pourquoi*, et *quoi faire*.

##  Comment ça fonctionne

1. Un modèle XGBoost prédit la probabilité de turnover pour chaque employé
2. SHAP (SHapley Additive exPlanations) décompose chaque prédiction pour identifier les facteurs déterminants
3. Mistral AI génère des recommandations RH personnalisées à partir de ces facteurs
4. Le tout est présenté dans un dashboard Streamlit interactif

##  Résultats

- **AUC : 0.8031** — bonne capacité de discrimination du modèle
- **Seuil de décision optimisé : 0.31**, ajusté pour la problématique métier
- Explicabilité complète par employé et par facteur via SHAP

##  Stack technique

- **Machine Learning** : XGBoost
- **Explicabilité** : SHAP
- **Génération de recommandations** : Mistral AI
- **Interface** : Streamlit (déployé sur Streamlit Cloud)

##  Contexte académique

Mémoire de fin d'études réalisé avec Adeline Thiara Bindia, sous la direction du Professeur Chamsedine Aidara — UCAO (Université Catholique de l'Afrique de l'Ouest), Dakar.

##  Auteure

**Kiné Seye** — Data Scientist & AI Developer, Dakar, Sénégal
[LinkedIn](https://www.linkedin.com/in/kine-seye-b513b13ba) · [GitHub](https://github.com/kine-seye)
