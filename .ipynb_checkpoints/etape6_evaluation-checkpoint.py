import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, classification_report
import plotly.graph_objects as go

# 1. Chargement du pipeline
try:
    with open('pipeline_data.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    
    modele = pipeline.get('modele_final', pipeline.get('meilleur_modele'))
    X_test = pipeline['X_test_proc']
    y_test = pipeline['y_test']

    # 2. Calcul du seuil optimal (Threshold Tuning)
    y_probs = modele.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    meilleur_seuil = thresholds[np.argmax(f1_scores)]

    # 3. Application du seuil
    y_pred_final = (y_probs >= meilleur_seuil).astype(int)

    print(f"✅ ÉTAPE 6 TERMINÉE AVEC SUCCÈS")
    print(f"──────────────────────────────────────────────────────────────────")
    print(f"🏆 Seuil optimal retenu : {meilleur_seuil:.2f}")
    print(f"📊 Rapport de performance :")
    print(classification_report(y_test, y_pred_final))

    # 4. Sauvegarde enrichie
    pipeline['y_probs'] = y_probs
    pipeline['meilleur_seuil'] = meilleur_seuil
    pipeline['y_pred_final'] = y_pred_final
    
    with open('pipeline_data.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"💾  pipeline_data.pkl    → enrichi et transmis à l'Étape 7")
    print(f"══════════════════════════════════════════════════════════════════")

except Exception as e:
    print(f"❌ Erreur : {e}")
    
