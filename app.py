import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib

# Cargar el modelo y los datos normalizados
pipeline = joblib.load('pipeline.joblib')
df_alojamientos = pd.read_csv('alojamientos_normalizados.csv')
df_bodegas = pd.read_csv('bodegas_normalizados.csv')

# Normalizar datos combinados para las gráficas
df_combined = pd.concat([df_alojamientos[['LATITUD', 'LONGITUD']].assign(tipo='alojamiento'),
                         df_bodegas[['LATITUD', 'LONGITUD']].assign(tipo='bodega')], ignore_index=True)
df_combined['cluster'] = pipeline.predict(df_combined[['LATITUD', 'LONGITUD']])

# Función para predecir clúster de nuevos datos
def predecir_cluster(nueva_latitud, nueva_longitud):
    nuevo_punto = np.array([[nueva_latitud, nueva_longitud]])
    cluster_predicho = pipeline.predict(nuevo_punto)
    return cluster_predicho[0]

# Definir la función haversine
def haversine(lat1, lon1, lat2, lon2):
    # Convertir grados a radianes
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Fórmula haversine
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c  # Radio de la Tierra en kilómetros
    return km

# Función para recomendar bodegas
def recomendar_bodegas_point(nueva_latitud, nueva_longitud):
    cluster_predicho = predecir_cluster(nueva_latitud, nueva_longitud)
    bodegas_cluster = df_bodegas[df_bodegas['cluster'] == cluster_predicho]

    point = pd.DataFrame([[nueva_latitud, nueva_longitud]], columns=['LATITUD', 'LONGITUD'])
    bodegas_cluster['DISTANCIA'] = bodegas_cluster.apply(lambda x: haversine(nueva_latitud, nueva_longitud, x['LATITUD'], x['LONGITUD']), axis=1)
    bodegas_cluster = bodegas_cluster.sort_values(by='DISTANCIA')
    return bodegas_cluster, cluster_predicho

# Función para crear una gráfica
def grafica_point_cluster(nueva_latitud, nueva_longitud, cluster_predicho):
    bodegas_cluster = df_bodegas[df_bodegas['cluster'] == cluster_predicho]
    plt.figure(figsize=(13, 10))
    for cluster in range(pipeline.n_clusters):
        plt.scatter(df_combined[df_combined['cluster'] == cluster]['LONGITUD'],
                    df_combined[df_combined['cluster'] == cluster]['LATITUD'],
                    label=f'Cluster {cluster}', alpha=0.5)

    plt.scatter(bodegas_cluster['LONGITUD'], bodegas_cluster['LATITUD'], color='blue', label='Bodegas Recomendadas', s=30)
    plt.scatter(nueva_longitud, nueva_latitud, color='pink', label='Nuevo Punto', s=150, marker='D')

    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.title('Clusters de Alojamiento y Bodegas con Nuevo Punto')
    plt.legend()
    st.pyplot(plt)

# Función para generar un mapa de la predicción
def crear_mapa_prediccion(nueva_latitud, nueva_longitud, bodegas_recomendadas_point):
    mendoza_model_map = folium.Map(location=[nueva_latitud, nueva_longitud], zoom_start=10)

    # Agregar marcador del nuevo punto
    folium.Marker(
        location=[nueva_latitud, nueva_longitud],
        popup="Nuevo Punto",
        icon=folium.Icon(color='pink')
    ).add_to(mendoza_model_map)

    # Agregar marcadores de bodegas recomendadas
    for index, row in bodegas_recomendadas_point.iterrows():
        folium.Marker(
            location=[row['LATITUD'], row['LONGITUD']],
            popup=f"{row['NOMBRE_BODEGA']} - {row['DISTANCIA']:.2f} km",
            icon=folium.Icon(color='blue')
        ).add_to(mendoza_model_map)

    return mendoza_model_map

# Función para verificar que las coordenadas enviadas por el usuario son validas
def validar_coordenadas(latitud, longitud):
    if -32.0150 <= latitud <= -3.5427 and -70.2640 <= longitud <= -66.5323:
        return True
    else:
        return False

# Streamlit App
st.title('El mejor recorrido para realizar enoturismo en Mendoza')
st.write('Esta aplicación está entrenada con un modelo de machine learning la cual toma una ubicación de la base de datos o cargada por el usuario manualmente y te brinda un tur más optimo para realizar enoturismo en la provincia de Mendoza.')
st.sidebar.header('Opciones de carga de datos')

# Seleccionar modo
modo = st.sidebar.selectbox("Seleccione el modo:", ["Seleccionar Alojamiento", "Ingresar Coordenadas"])

if modo == "Seleccionar Alojamiento":
    alojamiento = st.sidebar.selectbox("Seleccione un Alojamiento", df_alojamientos['NOMBRE_FANTASIA'].unique())
    if alojamiento:
        cluster_predicho = df_alojamientos[df_alojamientos['NOMBRE_FANTASIA'] == alojamiento]['cluster'].values[0]
        bodegas_recomendadas = df_bodegas[df_bodegas['cluster'] == cluster_predicho]
        alojamiento_coords = df_alojamientos[df_alojamientos['NOMBRE_FANTASIA'] == alojamiento][['LATITUD', 'LONGITUD']].values[0]
        
        lat_alojamiento, lon_alojamiento = alojamiento_coords
        bodegas_recomendadas.loc[:, 'DISTANCIA'] = bodegas_recomendadas.apply(lambda x: haversine(lat_alojamiento, lon_alojamiento, x['LATITUD'], x['LONGITUD']), axis=1)
        bodegas_recomendadas = bodegas_recomendadas.sort_values(by='DISTANCIA')

        st.subheader(f"Bodegas recomendadas de la más cercana a la más lejana:")
        st.write(f"alojamiento: {alojamiento}")
        st.dataframe(bodegas_recomendadas[['NOMBRE_BODEGA', 'DISTANCIA']])

        st.subheader('Mapa de Bodegas Recomendadas')
        map_recomendado = crear_mapa_prediccion(lat_alojamiento, lon_alojamiento, bodegas_recomendadas)
        folium_static(map_recomendado)

        st.subheader('Gráfica de Clusters realizado por el modelo.')
        grafica_point_cluster(lat_alojamiento, lon_alojamiento, cluster_predicho)

        st.sidebar.subheader("Proyecto realizado por Joaquín Rodríguez")
        st.sidebar.write("Presentación del proyecto")
        st.sidebar.write("- GitHub: [joarodev](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX)")
        st.sidebar.write("- Documentación: [TheOutlierMan](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX)")

elif modo == "Ingresar Coordenadas":
    nueva_latitud = st.sidebar.number_input("Latitud", min_value=-36.87000, max_value=-32.35000)
    nueva_longitud = st.sidebar.number_input("Longitud", min_value=-69.95000, max_value=-66.76000)
    if st.sidebar.button("Predecir Cluster"):
        bodegas_recomendadas, cluster_predicho = recomendar_bodegas_point(nueva_latitud, nueva_longitud)
        
        st.subheader(f"El nuevo punto pertenece al cluster: {cluster_predicho}")
        st.dataframe(bodegas_recomendadas[['NOMBRE_BODEGA', 'DISTANCIA']])
        
        st.subheader('Gráfica de Clusters')
        grafica_point_cluster(nueva_latitud, nueva_longitud, cluster_predicho)

        st.subheader('Mapa de Bodegas Recomendadas')
        map_recomendado = crear_mapa_prediccion(nueva_latitud, nueva_longitud, bodegas_recomendadas)
        folium_static(map_recomendado)

    st.sidebar.subheader("Proyecto realizado por Joaquín Rodríguez")
    st.sidebar.write("Presentación del proyecto")
    st.sidebar.write("- GitHub: [joarodev](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX)")
    st.sidebar.write("- Documentación: [TheOutlierMan](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX)")
