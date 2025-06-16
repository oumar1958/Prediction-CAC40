"""
Configuration pour le projet de prédiction du CAC40
"""

class Config:
    """Configuration principale du projet"""
    
    # Paramètres généraux
    random_state = 42  # Pour la reproductibilité
    
    # Paramètres de données
    symbol = "^FCHI"  # Symbole Yahoo Finance pour le CAC40
    data_period = "5y"  # Période de données historiques (5 ans)
    
    # Paramètres des indicateurs techniques
    sma_windows = [20, 50]  # Fenêtres pour les moyennes mobiles simples
    ema_window = 20  # Fenêtre pour la moyenne mobile exponentielle
    rsi_window = 14  # Fenêtre pour le RSI
    bollinger_window = 20  # Fenêtre pour les bandes de Bollinger
    bollinger_std = 2  # Nombre d'écarts-types pour les bandes de Bollinger
    
    # Paramètres du modèle RandomForest
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }
    
    # Paramètres pour la recherche d'hyperparamètres
    grid_search_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Paramètres de validation croisée
    n_splits = 5  # Nombre de plis pour la validation croisée temporelle
    
    # Paramètres de prédiction
    prediction_days = 5  # Nombre de jours à prédire
    test_size = 0.2  # Taille du jeu de test (20% des données)
    
    # Paramètres de visualisation
    plot_title = "Prédiction du CAC40"
    plot_width = 1200
    plot_height = 800
    
    # Chemins de sauvegarde
    output_dir = "results"
    predictions_file = "predictions.html"
    feature_importance_file = "feature_importance.html"
    technical_indicators_file = "technical_indicators.html"
