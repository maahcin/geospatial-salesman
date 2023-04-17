import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd


def homogeneous_poisson_on_rectangle(intensity, x_lim, y_lim):
    x_len = x_lim[1] - x_lim[0]
    y_len = y_lim[1] - y_lim[0]
    exp = intensity * x_len * y_len

    n = np.random.poisson(exp)

    df = pd.DataFrame()
    df["X"] = np.random.rand(n) * (x_lim[1] - x_lim[0]) + x_lim[0]
    df["Y"] = np.random.rand(n) * (y_lim[1] - y_lim[0]) + y_lim[0]

    return df


def generate_points_on_circle(radius, num_points, center):
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    x_temp = radius * np.cos(angles) + center[0]
    y_temp = radius * np.sin(angles) + center[1]
    x = [el + np.random.uniform(-radius / 50, radius / 50) for el in x_temp]
    y = [el + np.random.uniform(-radius / 50, radius / 50) for el in y_temp]

    df = pd.DataFrame()
    df["X"] = x
    df["Y"] = y
    return df


def points_on_circle_polygon(num_points, polygon, crs):
    bounds = polygon.bounds
    radius = np.min([(bounds.maxx - bounds.minx)[0], (bounds.maxy - bounds.miny)[0]]) / 3
    centroid = polygon.centroid
    df_1 = generate_points_on_circle(radius, num_points, [centroid.x[0], centroid.y[0]])

    temp = gpd.GeoDataFrame()
    temp['geometry'] = gpd.points_from_xy(df_1['X'], df_1['Y'], crs=crs)
    mult_polygon = gpd.GeoDataFrame()
    for i in range(len(temp)):
        mult_polygon = pd.concat(
            [mult_polygon, gpd.GeoDataFrame({'Nazwa': 'wojewodztwo', 'geometry': polygon.geometry})]).reset_index(
            drop=True)
    temp['is_in'] = temp.within(mult_polygon)

    df = gpd.GeoDataFrame({'geometry': temp[temp['is_in'] == True].geometry}).reset_index(drop=True)

    return df


def homogeneous_poisson_on_polygon(intensity, polygon, crs):
    bounds = polygon.bounds
    df_poisson = homogeneous_poisson_on_rectangle(intensity, [float(bounds.minx[0]), float(bounds.maxx[0])],
                                                  [float(bounds.miny[0]), float(bounds.maxy[0])])
    temp = gpd.GeoDataFrame()
    temp['geometry'] = gpd.points_from_xy(df_poisson['X'], df_poisson['Y'], crs=crs)
    mult_polygon = gpd.GeoDataFrame()
    for i in range(len(temp)):
        mult_polygon = pd.concat(
            [mult_polygon, gpd.GeoDataFrame({'Nazwa': 'wojewodztwo', 'geometry': polygon.geometry})]).reset_index(
            drop=True)
    temp['is_in'] = temp.within(mult_polygon)

    df = gpd.GeoDataFrame({'geometry': temp[temp['is_in'] == True].geometry}).reset_index(drop=True)

    return df


if __name__ == '__main__':
    country = gpd.read_file("Polska.zip")
    country = country.to_crs(2180)

    intensity = 50 / country.geometry[0].area
    points_1 = homogeneous_poisson_on_polygon(intensity, country.geometry, country.geometry.crs)
    points_2 = points_on_circle_polygon(40, country.geometry, country.geometry.crs)

    fig, ax = plt.subplots(1, 2, figsize=[20, 10])

    country.boundary.plot(ax=ax[0], color="red")
    points_1.plot(ax=ax[0], color="black")
    ax[0].set_title("Jednorodny proces punktowy Poissona na obszarze Polski")

    country.boundary.plot(ax=ax[1], color="red")
    points_2.plot(ax=ax[1], color="black")
    ax[1].set_title("Proces punktowy na okręgu na obszarze Polski")

    plt.show()
