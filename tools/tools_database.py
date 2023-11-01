"""
Python module reading and visualizing the data.

Functions
---------

read_data
    Read the data from a csv file's path.

load_coords
    Load the dictionary where we stored the spherical coordinates of each station.

build_network
    Construct the adjacency matrix of all stations in our dataset and the existing trajects.   

display_map
    Construct the initial map centred around France. 

add_map_markers
    Add a layer of markers (in the specified coordinates) on the given map in the parameters.   

add_map_routes
    Add a layer of lines for each traject in the network on the given map in the parameters.    

display_network
    Display the network of all the trajects specified in the dataset on a map   

display_map_delays
    Display the means of one column with respect to each station on a map   

box_plot_months
    The delay's (or any other numerical feature) box plot on one traject with respect to months 

histograms
    Display the histograms of one or multiple numerical features

remove_outliers
    Remove the oultiers rows with respect to the difference between arival and departure delays

last_month_column
	Add a column for the arrival delays of the last month to each row

"""

###############
### Imports ###
###############

### Python imports ###

import pandas as pd
import numpy as np
import pickle
import folium
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#################
### Functions ###
#################

def read_data(path):
    """
    Read the data from a csv file's path.

    Parameters
    ----------
    path : str 
        Path to the dataset.

    Returns
    -------
    dataset : Pandas dataframe. 
        loaded dataset.
    """
    dataset = pd.read_csv(path, delimiter=';')

    dataset['date'] = pd.to_datetime(dataset['date'])

    # last_month_delay = []

    # for j in range(len(dataset['date'])) :
    #     try :
    #         Row = dataset.iloc[j]
    #         Frame = dataset[dataset['gare_depart']==Row['gare_depart'] & dataset['gare_arrivee']==Row['gare_arrivee'] & dataset['date'].dt.month==(Row['date'].dt.month-1)]
    #         pass
    #     except :
    #         pass
    return dataset


def load_coords(path='./Data/Coords.pickle'):
    """
    Load the dictionary where we stored the spherical coordinates of each station.

    Parameters
    ----------
    path : str, optional (default is './Data/Coords.pickle')
        Path to the pickle file where the dictionary is saved.

    Returns
    -------
    station_coordinates : python dict 
        In the format {key->station_name : value->coordinates, }.
    """
    with open(path, 'rb') as file:
        station_coordinates = pickle.load(file)

    return station_coordinates


def build_network(station_coordinates, dataset, list_stations_name):
    """
    Construct the adjacency matrix of all stations in our dataset and the existing trajects.

    Parameters
    ----------
    station_cooridinates : python dict
        dictionary with the coordinates of each station.
    dataset : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.
    list_stations_name : list 
        Stations names list.

    Returns
    -------
    network : numpy.ndrray 
        Adjacency matrix of all the stations network.
    """
    network = np.zeros((len(station_coordinates), len(station_coordinates)))

    for i in range(len(network)):
        for j in range(i):
            if len(dataset[dataset['gare_depart'] == list_stations_name[i]][dataset['gare_arrivee'] == list_stations_name[j]]) or len(dataset[dataset['gare_depart'] == list_stations_name[j]][dataset['gare_arrivee'] == list_stations_name[i]]):
                network[i, j], network[j, i] = 1, 1

    return network


def display_map(zoom=5):
    """
    Construct the initial map centred around France.

    Parameters
    ----------
    zoom : int, optional (default is 5)
        Initial map zoom.

    Returns
    -------
    france_map
        The map. 
    """

    france_map = folium.Map(location=[46.603354, 1.888334], zoom_start=zoom)

    return france_map


def add_map_markers(map, markers_coords):
    """
    Add a layer of markers (in the specified coordinates) on the given map in the parameters.

    Parameters
    ----------
    map : html èvjlv
        Initial empty map centred around France.
    markers_coords : python dict 
        Dictionary with the name of each marker (or station) and its coordinates

    Returns
    -------
    None
    """

    for station, coordinates in markers_coords.items():
        folium.Marker(location=coordinates, popup=station).add_to(map)


def add_map_routes(map, network, station_coordinates, list_stations_name):
    """
    Add a layer of lines for each traject in the network on the given map in the parameters.

    Parameters
    ----------
    map : html èvjlv
        Initial empty map centred around France.
    network : numpy.ndarray 
        Adjacency matrix for our network 
    station_coordinates : numpy dict 
        Stations coordinates dictionary.
    list_stations_name : list
        List of the stations names.

    Returns
    -------
    None
    """

    for i in range(len(list_stations_name)):
        for j in range(i):
            if network[i, j]:
                coordinates = [station_coordinates[list_stations_name[i]],
                               station_coordinates[list_stations_name[j]]]
                folium.PolyLine(locations=coordinates,
                                color='black', weight=1).add_to(map)


def display_network(dataset):
    """
    Display the network of all the trajects specified in the dataset on a map.

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.

    Returns
    -------
    france_map
        The map.
    """

    station_coordinates = load_coords()
    list_stations_name = list(station_coordinates.keys())

    network = build_network(station_coordinates, dataset, list_stations_name)

    france_map = display_map(5)

    add_map_markers(france_map, station_coordinates)

    add_map_routes(france_map, network, station_coordinates, list_stations_name)

    return france_map


def display_map_delays(dataset, column='delay'):
    """
    Display the means of one column with respect to each station on a map.

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.
    column : str, optional (default is "delay")
        The numeric feature to display

    Returns
    -------
    france_map
        The map.
    """

    station_coordinates = load_coords()
    s = list(station_coordinates.keys())

    france_map = display_map(5)

    delays = []

    for station in s:

        frame = dataset[dataset['gare_arrivee'] == station]

        if column == 'delay':
            delays.append(
                np.mean(frame['retard_moyen_arrivee']/frame['duree_moyenne']))
        else:
            delays.append(np.mean(frame[column]))

    scaler = MinMaxScaler(feature_range=(0, 1))
    delays = scaler.fit_transform(np.array(delays).reshape(-1, 1))

    radius = 70000*delays.flatten()

    for i in range(len(s)):
        folium.Circle(
            location=station_coordinates[s[i]],
            radius=radius[i],
            color='blue',  # Circle border color
            weight=1,
            fill=True,
            fill_color='blue',  # Circle fill color
            fill_opacity=0.1,   # Opacity of the fill color
        ).add_to(france_map)

    return france_map


def box_plot_months(dataset, gare_dep, gare_arr, column):
    """
    The delay's (or any other numerical feature) box plot on one traject with respect to months.

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.
    gare_dep : str 
        Name of departure train station.
    gare_arr : str 
        Name of arriving train station.
    column : str 
        Numerical feature to plot.

    Returns
    -------
    None
    """

    data = []
    months = np.unique(dataset['date'].dt.month)

    for month in months:

        frame = dataset[dataset['gare_depart'] == gare_dep]
        frame = frame[frame['gare_arrivee'] == gare_arr]
        frame = frame[frame['date'].dt.month == month]

        data.append(np.array(frame[column]))

    plt.boxplot(data, labels=months)
    plt.xlabel('Months')
    plt.ylabel(column)
    plt.title(f'Traject {gare_dep} / {gare_arr}')
    plt.show()


def histograms(dataset, columns):
    """
    Display the histograms of one or multiple numerical features

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.
    columns : list 
        List of the numerical features to plot their histograms.

    Returns
    -------
    None
    """

    dataset[columns].hist(figsize=(20, 20), bins=100)
    plt.show()


def remove_outliers(dataset, threshold):
    """
    Remove the oultiers rows with respect to the difference between arrival and departure delays.

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.
    threshold : float 
        Z score threshold to eliminate the outliers.

    Returns
    -------
    dataset : pandas.core.frame.DataFrame
        Dataset with the outliers removed.
    """

    col = dataset['retard_moyen_arrivee']-dataset['retard_moyen_depart']
    mean = np.mean(np.array(col))
    std = np.std(np.array(col))

    Z_score = abs(np.array((col-mean)/std))
    dataset = dataset[Z_score < threshold]
    return dataset


def last_month_column(dataset_i):
    """
    Add a column to the dataset for the arrival delays of the last month to each row.

    Parameters
    ----------
    dataset_i : pandas.core.frame.DataFrame
        Dataset that contains all the possible trajects.

    Returns
    -------
    dataset : pandas.core.frame.DataFrame
        Dataset with the added column.
    """

    print("=========================")
    print("Adding last month column")
    print("=========================")

    last_month_delay = []
    dataset = pd.DataFrame.copy(dataset_i)

    for j in range(len(dataset['date'])):
        row = dataset.iloc[j]
        frame = dataset[dataset['gare_depart'] == row['gare_depart']]
        frame = frame[frame['gare_arrivee'] == row['gare_arrivee']]
        index = np.where(frame["date"] == row['date'])[0][0]
        if index > 0:
            last_month_delay.append(
                frame.iloc[index-1]['retard_moyen_arrivee'])
        else:
            last_month_delay.append(0)

    dataset['retard_mois_prec'] = last_month_delay
    dataset  = dataset[dataset['retard_mois_prec']!=0]
    # dataset  = dataset[dataset['retard_moyen_arrivee']!=0]

    return dataset
