# Prédiction CAC40

## Description du Projet

Ce projet utilise des techniques d'apprentissage automatique pour prédire les mouvements futurs du CAC40 (indice boursier français). Le modèle utilise une combinaison d'indicateurs techniques et d'un modèle RandomForest pour générer des prédictions précises.

## Indicateurs Techniques Utilisés

- **Moyennes Mobiles**
  - SMA 20 jours (Simple Moving Average)
  - SMA 50 jours
  - EMA 20 jours (Exponential Moving Average)

- **Bandes de Bollinger**
  - Bande supérieure (2 écarts-types au-dessus de SMA 20)
  - Bande inférieure (2 écarts-types en dessous de SMA 20)

- **MACD**
  - Utilisation de deux moyennes mobiles exponentielles (12 et 26 jours)

- **RSI**
  - Relative Strength Index sur 14 jours

## Architecture du Projet

```
CAC40_prediction/
├── cac40_prediction.py        # Script principal
├── requirements.txt          # Dépendances Python
├── README.md                # Documentation
├── feature_importance.html  # Importance des indicateurs
├── predictions.html         # Comparaison prédictions/réels
└── technical_indicators.html # Visualisation des indicateurs
```

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
1. Télécharger automatiquement les données historiques du CAC40 depuis Yahoo Finance
2. Calculer les indicateurs techniques
3. Préparer les données pour l'entraînement
4. Entraîner un modèle RandomForest avec recherche d'hyperparamètres
5. Générer des prédictions pour 5 jours
6. Sauvegarder les résultats dans des graphiques interactifs HTML

## Résultats

Le script génère trois graphiques interactifs HTML :

1. `predictions.html`
   - Comparaison entre les valeurs réelles et les prédictions
   - Métriques de performance (MSE, MAE, R² Score)

2. `feature_importance.html`
   - Importance relative des différents indicateurs techniques
   - Contribution de chaque indicateur à la prédiction

3. `technical_indicators.html`
   - Visualisation des indicateurs techniques sur le prix
   - Bandes de Bollinger, RSI, MACD

## Métriques de Performance

Le modèle atteint une performance moyenne de :
- MSE : ~0.000184
- MAE : ~0.010141
- R² Score : ~0.6575

## Configuration

Les paramètres du modèle sont configurables via le fichier `config.py` :
- Période de données
- Paramètres des indicateurs techniques
- Paramètres du modèle RandomForest

## Contributeurs

 Oumar Abdramane ALLAWAN

