# Prédiction du CAC40

Ce projet utilise des données historiques pour prédire les mouvements futurs du CAC40 en utilisant des techniques d'apprentissage automatique.

## Installation

1. Créez un environnement virtuel Python :
```bash
python -m venv venv
```

2. Activez l'environnement virtuel :
```bash
# Windows
venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Pour exécuter le script de prédiction :
```bash
python cac40_prediction.py
```

Le script va :
1. Télécharger les données historiques du CAC40
2. Calculer des indicateurs techniques (SMA, RSI, Bandes de Bollinger)
3. Entraîner un modèle RandomForest
4. Générer des prédictions
5. Sauvegarder les résultats dans des graphiques

## Résultats

Le script génère deux graphiques :
- `predictions.png` : Comparaison entre les valeurs réelles et les prédictions
- `feature_importance.png` : Importance relative des différents indicateurs techniques
