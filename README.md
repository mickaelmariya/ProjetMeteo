# ProjetMeteo

# 🌩️ WeatherForYnov — Analyse météo & prédiction des intempéries

Application de data storytelling et de machine learning pour analyser des données météorologiques historiques de plusieurs villes françaises et estimer les conditions favorables aux épisodes de grêle.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ML](https://img.shields.io/badge/Machine%20Learning-RandomForest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## Main features

- Analyse de données météorologiques historiques
- Détection des conditions favorables à la grêle
- Visualisations interactives avec Streamlit
- Modèle de machine learning pour estimer le risque
- Carte heatmap du risque par ville
- Simulation météo en temps réel
- Export des jours à risque

---

## Table of Contents

- [Project overview](#project-overview)
- [Dataset](#dataset)
- [Project pipeline](#project-pipeline)
- [Quick start](#quick-start)
- [Machine learning](#machine-learning)
- [Streamlit application](#streamlit-application)
- [Project structure](#project-structure)
- [Results](#results)
- [Installation](#installation)
- [License](#license)

---

## Project overview

L’objectif du projet est d’analyser des données météorologiques issues de plusieurs villes françaises afin d’identifier les conditions climatiques associées aux épisodes d’intempéries, notamment la grêle.

Le projet combine :

- acquisition et préparation de données,
- analyse exploratoire,
- création d’un indicateur `hail_risk`,
- modélisation prédictive,
- application interactive de visualisation.

---

## Dataset

Les données utilisées proviennent de l’API **Open-Meteo** et couvrent plusieurs villes françaises :

- Paris
- Lyon
- Marseille
- Lille
- Bordeaux

### Variables principales

| Column | Description |
|--------|-------------|
| `time` | Date de l’observation |
| `city` | Ville |
| `temperature_2m_max` | Température maximale |
| `temperature_2m_min` | Température minimale |
| `temperature_2m_mean` | Température moyenne |
| `precipitation_sum` | Précipitations totales |
| `rain_sum` | Pluie totale |
| `showers_sum` | Averses totales |
| `wind_speed_10m_max` | Vitesse max du vent |
| `wind_gusts_10m_max` | Rafales max |
| `relative_humidity_2m_mean` | Humidité relative moyenne |
| `cloud_cover_mean` | Couverture nuageuse moyenne |
| `sunshine_duration` | Durée d’ensoleillement |
| `shortwave_radiation_sum` | Rayonnement solaire |

### Variables créées dans le projet

| Column | Description |
|--------|-------------|
| `hail_risk` | Indicateur de conditions favorables à la grêle |
| `prediction` | Prédiction du modèle ML |
| `proba` | Probabilité estimée par le modèle |
| `month` | Mois extrait de la date |

---

## Project pipeline

### 1. Data acquisition and preparation
- Récupération des données météo via Open-Meteo
- Fusion des villes dans un dataset unique
- Nettoyage et structuration des données
- Création de `weather_clean.csv`

### 2. Exploratory data analysis
- Évolution des températures
- Analyse des précipitations
- Heatmap de corrélation
- Analyse temporelle des épisodes à risque

### 3. Hail risk detection
Création d’un indicateur métier :

```python
hail_risk = (
    (wind_gusts_10m_max > 60) &
    (precipitation_sum > 10)
).astype(int)
