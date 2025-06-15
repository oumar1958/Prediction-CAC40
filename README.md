# Projet de Prévision du Taux de Chômage en France

## Description du Projet

Ce projet vise à développer un modèle statistique de prévision du taux de chômage en France en utilisant des données macroéconomiques. Les modèles ARIMA, la régression multiple et les modèles VAR seront appliqués pour analyser les données et faire des prédictions.

## Structure du Projet

Le projet est organisé en trois composantes principales :

- **Chômage** : Contient les données et analyses spécifiques au taux de chômage
- **Inflation** : Données et analyses liées à l'inflation
- **PIB** : Données et analyses liées au Produit Intérieur Brut

## Données Utilisées

Les données principales sont stockées dans les fichiers suivants :

- `2025T1_sl_chomage.xlsx` : Données sur le taux de chômage
- `2025T1_sl_indicateurs.xlsx` : Données sur les indicateurs macroéconomiques

## Méthodologie

1. **Analyse Exploratoire**
   - Analyse des tendances temporelles
   - Tests de stationnarité
   - Analyse de corrélation entre variables

2. **Modélisation**
   - Modèles ARIMA pour les séries temporelles
   - Régression multiple pour identifier les facteurs explicatifs
   - Modèles VAR pour l'analyse des interactions entre variables

3. **Évaluation**
   - Validation croisée
   - Métriques de performance (RMSE, MAE, R²)
   - Tests statistiques de significativité

## Technologies Utilisées

- R pour le traitement et l'analyse des données
- Packages R spécifiques pour la modélisation statistique
- Excel pour le stockage initial des données

## Résultats Attendus

- Prédictions du taux de chômage à court et moyen terme
- Analyse des facteurs macroéconomiques influençant le chômage
- Interprétations économiques des résultats
- Comparaison des performances des différents modèles

## Prérequis

- R et RStudio installés
- Packages R nécessaires (stats, forecast, vars, etc.)
- Excel pour la manipulation des fichiers de données (.xlsx)

## Structure des Dossiers

```
Prédiction_chômage_France/
├── Chômage/
│   ├── 2025T1_sl_chomage.xlsx
│   └── 2025T1_sl_indicateurs.xlsx
├── Inflation/
└── PIB/
```

## Auteurs

Ce projet est réalisé dans le cadre du Master 1 Analyse et politique économique, parcours Statistique pour l'évaluation et la prévision.

## Licence

Ce projet est protégé par les droits d'auteur. Pour toute utilisation, veuillez contacter l'auteur.
