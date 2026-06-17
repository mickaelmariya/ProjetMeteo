

# 🌩️ PROJET METEO — Analyse météo & prédiction des intempéries

Application de data storytelling et de machine learning permettant d’analyser des données météorologiques historiques de plusieurs villes françaises et d’identifier les conditions favorables aux épisodes de grêle et d’intempéries.

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
- [Data collection process](#data-collection-process)
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

- l’acquisition et la préparation des données,
- l’analyse exploratoire,
- la création d’un indicateur `hail_risk`,
- une modélisation prédictive,
- une application interactive de visualisation.

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
| `wind_speed_10m_max` | Vitesse maximale du vent |
| `wind_gusts_10m_max` | Rafales maximales |
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

## Data collection process

Les données brutes ne sont pas versionnées dans ce dépôt.  
Elles peuvent être régénérées à partir de l’API **Open-Meteo** en suivant le processus ci-dessous.

### Source
- API utilisée : **Open-Meteo Archive API**
- URL : `https://archive-api.open-meteo.com/v1/archive`
- Documentation Open-Meteo : `https://open-meteo.com/`

### Villes étudiées
Les données sont récupérées pour plusieurs villes françaises :
- Paris
- Lyon
- Marseille
- Lille
- Bordeaux

### Coordonnées utilisées
- Paris : `48.8566, 2.3522`
- Lyon : `45.7640, 4.8357`
- Marseille : `43.2965, 5.3698`
- Lille : `50.6292, 3.0573`
- Bordeaux : `44.8378, -0.5792`

### Variables collectées
Les variables demandées à l’API sont :
- `temperature_2m_max`
- `temperature_2m_min`
- `temperature_2m_mean`
- `precipitation_sum`
- `rain_sum`
- `showers_sum`
- `wind_speed_10m_max`
- `wind_gusts_10m_max`
- `relative_humidity_2m_mean`
- `cloud_cover_mean`
- `sunshine_duration`
- `shortwave_radiation_sum`

### Méthodologie
1. Définir la liste des villes avec leurs coordonnées géographiques.
2. Envoyer une requête à l’API Open-Meteo pour chaque ville.
3. Récupérer les données historiques au format JSON.
4. Convertir chaque réponse en DataFrame avec `pandas`.
5. Ajouter une colonne `city` pour identifier la ville.
6. Fusionner les DataFrames de toutes les villes en un seul dataset.
7. Sauvegarder les données brutes dans `data/raw/`.
8. Nettoyer et structurer les données pour générer `data/clean/weather_clean.csv`.

### Exemple de logique Python utilisée

```python
import pandas as pd
import requests

url = "https://archive-api.open-meteo.com/v1/archive"

cities = {
    "Paris": (48.8566, 2.3522),
    "Lyon": (45.7640, 4.8357),
    "Marseille": (43.2965, 5.3698),
    "Lille": (50.6292, 3.0573),
    "Bordeaux": (44.8378, -0.5792)
}

all_data = []

for city, coords in cities.items():
    params = {
        "latitude": coords[0],
        "longitude": coords[1],
        "start_date": "2014-01-01",
        "end_date": "2023-12-31",
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum,showers_sum,wind_speed_10m_max,wind_gusts_10m_max,relative_humidity_2m_mean,cloud_cover_mean,sunshine_duration,shortwave_radiation_sum",
        "timezone": "Europe/Paris"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "daily" in data:
        df = pd.DataFrame(data["daily"])
        df["city"] = city
        all_data.append(df)

weather_data = pd.concat(all_data, ignore_index=True)
weather_data.to_csv("data/raw/weather_france.csv", index=False)
