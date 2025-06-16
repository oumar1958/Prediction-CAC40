import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volume import VolumeWeightedAveragePrice
import plotly.graph_objects as go
from datetime import datetime
from config import Config

# Configuration utilisée depuis config.py
config = Config()

# Téléchargement des données
print("Téléchargement des données du CAC40...")
cac40 = yf.download(config.symbol, period=config.data_period)

# Création des indicateurs techniques
print("Création des indicateurs techniques...")

# Moyennes mobiles
cac40['SMA_20'] = cac40['Close'].rolling(window=20).mean()
cac40['SMA_50'] = cac40['Close'].rolling(window=50).mean()
cac40['EMA_20'] = cac40['Close'].ewm(span=20).mean()

# Bandes de Bollinger
mean = cac40['Close'].rolling(window=20).mean()
std = cac40['Close'].rolling(window=20).std()
cac40['BB_High'] = mean + 2 * std
cac40['BB_Low'] = mean - 2 * std

# MACD
exp1 = cac40['Close'].ewm(span=12, adjust=False).mean()
exp2 = cac40['Close'].ewm(span=26, adjust=False).mean()
cac40['MACD'] = exp1 - exp2
cac40['MACD_Signal'] = cac40['MACD'].ewm(span=9, adjust=False).mean()

# RSI et Stochastique
delta = cac40['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
cac40['RSI'] = 100 - (100 / (1 + gain / loss))

stoch_k = ((cac40['Close'] - cac40['Low'].rolling(window=14).min()) / 
           (cac40['High'].rolling(window=14).max() - cac40['Low'].rolling(window=14).min()) * 100)
cac40['Stochastic'] = stoch_k.rolling(window=3).mean()

# Volume
typical_price = (cac40['High'] + cac40['Low'] + cac40['Close']) / 3
cac40['VWAP'] = typical_price.rolling(window=20).sum() / cac40['Volume'].rolling(window=20).sum()

# Volatilité
cac40['ATR'] = (abs(cac40['High'] - cac40['Low']).rolling(window=14).mean() +
                abs(cac40['High'] - cac40['Close'].shift()).rolling(window=14).mean() +
                abs(cac40['Low'] - cac40['Close'].shift()).rolling(window=14).mean()) / 3

# Préparation des données
print("Préparation des données...")
print("Préparation des données...")
# Création de la variable cible (rendement futur)
cac40['Future_Return'] = cac40['Close'].pct_change(5).shift(-5)

# Sélection des features
features = [
    'Open', 'High', 'Low', 'Volume',
    'SMA_20', 'SMA_50', 'EMA_20',
    'BB_High', 'BB_Low',
    'MACD', 'MACD_Signal',
    'RSI', 'Stochastic',
    'VWAP', 'ATR'
]

# Remplacer les valeurs infinies par NaN
cac40 = cac40.replace([np.inf, -np.inf], np.nan)

# Alignement des données en gardant les mêmes indices
X = cac40[features]
y = cac40['Future_Return']

# Suppression des lignes avec des valeurs manquantes
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Validation croisée temporelle
tscv = TimeSeriesSplit(n_splits=config.n_splits)

# Recherche des meilleurs hyperparamètres avec GridSearchCV
print("Recherche des meilleurs hyperparamètres...")
model = RandomForestRegressor(random_state=config.random_state)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=config.grid_search_params,
    cv=tscv,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_scaled, y)

# Utilisation du meilleur modèle
best_model = grid_search.best_estimator_
print(f"Meilleurs paramètres: {grid_search.best_params_}")

# Division des données avec TimeSeriesSplit
print("Division des données avec TimeSeriesSplit...")
train_size = int(len(X) * (1 - config.test_size))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Prédiction
y_pred = best_model.predict(X_test)

# Évaluation détaillée du modèle
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nRésultats de l'évaluation:")
print(f"Mean Squared Error: {mse:.6f}")
print(f"Mean Absolute Error: {mae:.6f}")
print(f"R² Score: {r2:.4f}")

# Visualisation détaillée des résultats
print("\nCréation des visualisations...")

# Graphique des prédictions vs réels
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Valeurs réelles'))
fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, name='Prédictions', line=dict(dash='dot')))
fig.update_layout(
    title='Prédictions vs Valeurs Réelles du CAC40',
    xaxis_title='Date',
    yaxis_title='Rendement futur',
    template='plotly_dark'
)
fig.write_html('predictions.html')

# Graphique de l'importance des features
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

fig = go.Figure(
    go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h'
    )
)
fig.update_layout(
    title='Importance des Indicateurs Techniques',
    template='plotly_dark'
)
fig.write_html('feature_importance.html')

# Graphique des indicateurs techniques
fig = go.Figure()
fig.add_trace(go.Scatter(x=cac40.index, y=cac40['Close'], name='Prix'))
fig.add_trace(go.Scatter(x=cac40.index, y=cac40['SMA_20'], name='SMA 20'))
fig.add_trace(go.Scatter(x=cac40.index, y=cac40['SMA_50'], name='SMA 50'))
fig.add_trace(go.Scatter(x=cac40.index, y=cac40['EMA_20'], name='EMA 20'))
fig.add_trace(go.Scatter(x=cac40.index, y=cac40['BB_High'], name='BB High', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=cac40.index, y=cac40['BB_Low'], name='BB Low', line=dict(dash='dot')))
fig.update_layout(
    title='Indicateurs Techniques du CAC40',
    xaxis_title='Date',
    yaxis_title='Prix',
    template='plotly_dark'
)
fig.write_html('technical_indicators.html')

print("\nAnalyse terminée. Les résultats ont été sauvegardés dans les fichiers HTML")
