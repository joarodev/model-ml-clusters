# Análisis de datos + Modelo de Clustering Machine Learning 

#### ¿En que zona o departamento de la provincia de Mendoza Argentina me debería ubicar si deseo realizar optimamente enoturismo?

#### LINKS DE REFERENCIA

- [Presentación del proyecto](https://drive.google.com/file/d/1RAYVfwTHp9TcLFh0JrNJnSTRsQ0gB51G/view?usp=drive_link)

- [Repositorio Github](https://github.com/joarodev/Analisis-enoturismo-Mendoza)

- [Aplicación web](https://model-ml-enoturismo-mendoza.streamlit.app/)

### Motivación y audiencia

Lo que me motiva a realizar este proyecto es un problema muy común en mi provincia, los turistas que deciden visitarla tienen un objetivo muy claro, realizar desgustaciones de vinos, pero desconocen donde realizar esta actividad.

Por lo tanto decidí realizar un análisis, buscar el mejor alojamiento el cual me permita posicionarme estrategicamente para visitar la mayor cantidad de bodegas, y así poder concluir el objetivo de la visita.

Además voy a generar un modelo de Machine Learning, utilizare un algoritmo de clusterización, el mismo me permitirá pasarle los datos de mi localización para que realice una predicción de la zona en la que me encuentro y marque en un mapa las bodegas dentro de mi misma zona para poder realizar enoturismo.

### El análisis descarta y valida las siguientes hipótesis.

#### Hipótesis

* La zona donde podemos encontrar más cantidad de cabañas para hospedarnos en la provincia es la zona oeste (Lujan de cuyo, Tunuyán, Tupungato, San Carlos) debido a que es un área montañosa porque atravieza la cordillera de Los Andes. -
* Valle De Uco, conformado por (Tunuyán, Tupungato y San Carlos) es donde podemos encontrar más fincas con viñedos. Por lo tanto es donde se ubica mayor cantidad de bodegas.
* El centro de Mendoza es el departamento donde podemos encontrar más cantidad de hoteles ya que se ubica cercano a la terminal de ómnibus y el aeropuerto Internacional Francisco Gabrielli.
#### Preguntas de interés

* ¿Donde ubicarse estrategicamente para visitar la mayor cantidad de bodegas?
* ¿La ciudad de Mendoza es el departamento con más cantidad de ubicaciones para hospedarse?
* ¿Es mucho más facil encontrar un Hotel que cualquier establecimiento?
### Información y datos técnicos.

Los datos fueron extraidos de la página oficial "Datos Abiertos Mendoza", para posteriormente procesarlos y generar mediante la extensión **Geocode** los datos de Latitud y longitud de enoturismo y alojamientos.

* https://datosabiertos.mendoza.gov.ar/

#### Datasets y Geojson.

* Dataset de alojamientos en Mendoza: https://raw.githubusercontent.com/TheOutlierMan/public-datasets/main/mendoza-datasets/alojamientos-mendoza.csv

* Dataset de enoturismo en Mendoza: https://raw.githubusercontent.com/TheOutlierMan/public-datasets/main/mendoza-datasets/enoturismo_mendoza.csv

* GeoJson de los departamentos de Mendoza: https://raw.githubusercontent.com/TheOutlierMan/cartografia/main/departamentos-mendoza.json

#### Tecnologías utilizadas.

* Python: Lenguae de programación que aplique para el análisis.
* Google Colaboratory: IDE en la nube para realizar el script.
* Geocode: extensión de google spreadsheet para localizar direcciones.
* Google spreadsheet: Cargar datasets para generar nuevas categorías

#### Librerías.
* numpy 
* pandas 
* matplotlib
* seaborn 
* plotly
* geopandas
* folium
* MarkerCluster
* requests
* json
* sklearn
* Joblib



## Script base

https://colab.research.google.com/drive/1OauszM6ZThCg0u0zzWXpmoJQZXq9LDli




#### 🚀 Realizado por Joaquín Rodríguez

* [linkedin] *https://www.linkedin.com/in/joaquinrodriguez-dev/*
* [kaggle] *https://www.linkedin.com/in/joaquinrodriguez-dev/*



