import streamlit as st
import pandas as pd
import os
import pydeck as pdk
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Dashboard Météo & Grêle", layout="wide")
st.title("🌩️ Dashboard météo & prédiction des intempéries")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(BASE_DIR, "data", "clean", "weather_clean.csv")

    df = pd.read_csv(path)

    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month

    df["hail_risk"] = (
        (df["wind_gusts_10m_max"] > 60) &
        (df["precipitation_sum"] > 10)
    ).astype(int)

    return df

df = load_data()

# ======================
# MODEL
# ======================
X = df.select_dtypes(include="number").drop(columns=["hail_risk"])
y = df["hail_risk"]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X, y)

df["prediction"] = model.predict(X)
df["proba"] = model.predict_proba(X)[:,1]

# ======================
# SIDEBAR
# ======================
st.sidebar.header("Filtres")
city = st.sidebar.selectbox("Ville", df["city"].unique())

df_city = df[df["city"] == city]

# ======================
# HEATMAP MAP 🗺️
# ======================
st.subheader("🗺️ Carte heatmap du risque de grêle")

coords = {
    "Paris": [48.85, 2.35],
    "Lyon": [45.75, 4.85],
    "Marseille": [43.29, 5.37],
    "Lille": [50.63, 3.06],
    "Bordeaux": [44.84, -0.58]
}

map_data = df.groupby("city")["proba"].mean().reset_index()
map_data["lat"] = map_data["city"].apply(lambda x: coords[x][0])
map_data["lon"] = map_data["city"].apply(lambda x: coords[x][1])

layer = pdk.Layer(
    "HeatmapLayer",
    data=map_data,
    get_position="[lon, lat]",
    get_weight="proba",
    radiusPixels=60
)

view = pdk.ViewState(latitude=46.5, longitude=2.5, zoom=5)

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view,
    tooltip={
        "html": "<b>{city}</b><br/>Risque: {proba:.2f}",
        "style": {"backgroundColor": "black", "color": "white"}
    }
))

# ======================
# GRAPHS INTERACTIFS 
# ======================
col1, col2 = st.columns(2)

# TEMP
with col1:
    st.subheader("🌡️ Température")

    fig = px.line(
        df_city,
        x="time",
        y="temperature_2m_mean",
        title="Température"
    )

    fig.update_traces(mode="lines+markers")

    st.plotly_chart(fig, use_container_width=True)

# HAIL
with col2:
    st.subheader("🌩️ Épisodes réels")

    hail_days = df_city[df_city["hail_risk"] == 1]

    fig = px.scatter(
        hail_days,
        x="time",
        y="wind_gusts_10m_max",
        size="precipitation_sum",
        color="precipitation_sum",
        hover_data=["temperature_2m_mean"],
        title="Conditions grêle"
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================
# PROBA TEMPS
# ======================
st.subheader("📈 Probabilité de grêle dans le temps")

fig = px.line(
    df_city,
    x="time",
    y="proba",
    title="Probabilité de grêle"
)

fig.update_traces(mode="lines+markers")

st.plotly_chart(fig, use_container_width=True)

# ======================
# SIMULATION LIVE 
# ======================
st.subheader("🤖 Simulation météo (temps réel)")

col1, col2, col3 = st.columns(3)

with col1:
    wind = st.slider("Vent (km/h)", 0, 120, 50)

with col2:
    rain = st.slider("Précipitations (mm)", 0, 50, 5)

with col3:
    temp = st.slider("Température (°C)", -10, 40, 15)

input_data = pd.DataFrame([{
    "temperature_2m_max": temp,
    "temperature_2m_min": temp - 5,
    "temperature_2m_mean": temp,
    "precipitation_sum": rain,
    "rain_sum": rain,
    "showers_sum": rain,
    "wind_speed_10m_max": wind,
    "wind_gusts_10m_max": wind,
    "relative_humidity_2m_mean": 70,
    "cloud_cover_mean": 50,
    "sunshine_duration": 5,
    "shortwave_radiation_sum": 200,
    "month": 6
}])

input_data = input_data.reindex(columns=X.columns, fill_value=0)

pred = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0][1]

st.progress(proba)

if pred == 1:
    st.error(f"⚠️ Risque de grêle : {proba*100:.1f}%")
else:
    st.success(f"✅ Faible risque : {proba*100:.1f}%")

# ======================
# STATS
# ======================
st.subheader("📊 Statistiques")

col1, col2 = st.columns(2)

col1.metric("Réel", int(df_city["hail_risk"].sum()))
col2.metric("Prédit", int(df_city["prediction"].sum()))

# ======================
# EXPORT
# ======================
st.subheader("💾 Export des jours à risque")

risk_days = df[df["hail_risk"] == 1]

st.download_button(
    label="Télécharger CSV",
    data=risk_days.to_csv(index=False),
    file_name="hail_risk_days.csv",
    mime="text/csv"
)