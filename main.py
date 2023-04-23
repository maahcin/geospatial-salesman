import numpy as np
import pandas as pd
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd


def homogeneous_poisson_on_rectangle(intensity, x_lim, y_lim):
    """
    Function that generates a random Poisson point process on a rectangle with a given intensity.

    :param intensity: The intensity of the Poisson point process.
    :param x_lim: A tuple or list specifying the lower and upper limits of the x-axis of the rectangle.
    :param y_lim: A tuple or list specifying the lower and upper limits of the y-axis of the rectangle.
    :return: A Pandas DataFrame with generated points as two columns: "X" for x-coordinates and "Y" for y-coordinates.
    """
    # Calculate the expected number of points based on the intensity and rectangle dimensions
    x_len = x_lim[1] - x_lim[0]
    y_len = y_lim[1] - y_lim[0]
    exp = intensity * x_len * y_len

    # Generate the number of points from a Poisson distribution
    n = np.random.poisson(exp)

    # Generate random x and y coordinates within the rectangle
    df = pd.DataFrame()
    df["X"] = np.random.rand(n) * (x_lim[1] - x_lim[0]) + x_lim[0]
    df["Y"] = np.random.rand(n) * (y_lim[1] - y_lim[0]) + y_lim[0]

    return df


def generate_points_on_circle(radius, num_points, center):
    """
    Function that generates a random point process on a circle with a given radius.

    :param radius: The radius of the circle.
    :param num_points: The number of points to be generated on the circle.
    :param center: A tuple or list specifying the center coordinates (x, y) of the circle.
    :return: A Pandas DataFrame with generated points as two columns: "X" for x-coordinates and "Y" for y-coordinates.
    """
    # Generate random angles from 0 to 2 * pi
    angles = np.random.uniform(0, 2 * np.pi, num_points)

    # Convert polar coordinates (radius, angle) to Cartesian coordinates (x, y)
    x_temp = radius * np.cos(angles) + center[0]
    y_temp = radius * np.sin(angles) + center[1]

    # Add small random perturbations to x and y coordinates
    x = [el + np.random.uniform(-radius / 50, radius / 50) for el in x_temp]
    y = [el + np.random.uniform(-radius / 50, radius / 50) for el in y_temp]

    # Create a Pandas DataFrame with the generated points
    df = pd.DataFrame()
    df["X"] = x
    df["Y"] = y

    return df


def points_on_circle_polygon(num_points, polygon, crs):
    """
    Generates points on the circumference of a circle centered at the centroid of the input polygon geometry.

    :param num_points: Number of points to generate on the circle.
    :param polygon: A polygon geometry from the geopandas library.
    :param crs: The coordinate reference system (CRS) for the generated points.
    :return: A geopandas GeoDataFrame containing the generated points that fall within the input polygon.
    """
    # Calculate the bounds of the input polygon
    bounds = polygon.bounds
    # Determine the radius of the circle as one-third of the minimum of the x and y bounds
    radius = np.min([(bounds.maxx - bounds.minx)[0], (bounds.maxy - bounds.miny)[0]]) / 3
    # Calculate the centroid of the polygon
    centroid = polygon.centroid
    # Generate points on the circle using a previously defined function (generate_points_on_circle)
    df_1 = generate_points_on_circle(radius, num_points, [centroid.x[0], centroid.y[0]])

    # Create a temporary GeoDataFrame to store the generated points
    temp = gpd.GeoDataFrame()
    # Convert the generated points to GeoPoints and add them to the temporary GeoDataFrame
    temp['geometry'] = gpd.points_from_xy(df_1['X'], df_1['Y'], crs=crs)

    # Create a temporary GeoDataFrame to store the multipolygon (same as the input polygon)
    mult_polygon = gpd.GeoDataFrame()
    # Loop through each point in the temporary GeoDataFrame
    for i in range(len(temp)):
        # Concatenate the multipolygon with a new GeoDataFrame containing the same polygon geometry
        mult_polygon = pd.concat(
            [mult_polygon, gpd.GeoDataFrame({'Nazwa': 'wojewodztwo', 'geometry': polygon.geometry})]).reset_index(
            drop=True)

    # Check if each point is within the multipolygon (i.e., the input polygon)
    temp['is_in'] = temp.within(mult_polygon)

    # Create a final GeoDataFrame containing only the points that fall within the input polygon
    df = gpd.GeoDataFrame({'geometry': temp[temp['is_in'] == True].geometry}).reset_index(drop=True)

    # Return the GeoDataFrame with the generated points within the input polygon
    return df


def homogeneous_poisson_on_polygon(intensity, polygon, crs):
    """
    Function that generates a homogeneous Poisson point process on a polygonal geometry using a previously defined function
    for generating Poisson points on a rectangle.

    :param intensity: The intensity or average density of points in the Poisson process.
    :param polygon: A polygon geometry from the geopandas library on which the Poisson points will be generated.
    :param crs: The coordinate reference system (CRS) for the generated points.
    :return: A geopandas GeoDataFrame containing the generated points that fall within the input polygon.
    """
    # Calculate the bounding box of the input polygon
    bounds = polygon.bounds

    # Generate Poisson points on a rectangle that bounds the polygon
    df_poisson = homogeneous_poisson_on_rectangle(intensity, [float(bounds.minx[0]), float(bounds.maxx[0])],
                                                  [float(bounds.miny[0]), float(bounds.maxy[0])])

    # Create a geopandas GeoDataFrame to store the generated points
    temp = gpd.GeoDataFrame()
    temp['geometry'] = gpd.points_from_xy(df_poisson['X'], df_poisson['Y'], crs=crs)

    # Check if each point falls within the polygon
    mult_polygon = gpd.GeoDataFrame()
    for i in range(len(temp)):
        mult_polygon = pd.concat(
            [mult_polygon, gpd.GeoDataFrame({'Nazwa': 'wojewodztwo', 'geometry': polygon.geometry})]).reset_index(
            drop=True)
    temp['is_in'] = temp.within(mult_polygon)

    # Create a new GeoDataFrame containing only the points that fall within the polygon
    df = gpd.GeoDataFrame({'geometry': temp[temp['is_in'] == True].geometry}).reset_index(drop=True)

    return df


def generate_permutations_with_fixed_element(n, population_size, first=0):
    """
    Function that generates m permutations of an n-element set, where
    the specified element always occupies the first position.

    :param n: Length of set which will be permutated.
    :param population_size: Number of permutations to generate.
    :param first: Element that should be in the first position of permutations.
    :return: List of m permutations of an n-element set with the element in the first position.
    """

    # Generate list from 0 to n
    iterable = list(range(n))

    # Check if the element is present in the iterable
    if first not in iterable:
        raise ValueError("Element must be present in the iterable set.")

    # Remove the element from the iterable set
    iterable.remove(first)

    # Generate permutations using np.random.permutation
    # and add the element to the first position in each permutation
    permutations = []
    for _ in range(population_size):
        perm = np.random.permutation(iterable).tolist()
        perm.insert(0, first)
        permutations.append(perm)

    return np.array(permutations)


def calculate_distances(points, perms):
    """
    Function that calculates distances between consecutive points in a geodataframe.
    Order is arranged by permutations.

    :param points: A geodataframe containing point data with x and y coordinates.
    :param perms: An array representing permutations of indices that define the order of points.
    :return: An array containing the sum of distances for each permutation of points provided in perms.
    """

    # Extract x and y coordinates from the geodataframe
    x = np.array(points.geometry.x)
    y = np.array(points.geometry.y)

    # Create a shifted version of perms
    perms_shift = np.roll(perms, -1)

    # Create arrays for x and y coordinates of points in specified order and shifted order
    x_new = x[perms]
    y_new = y[perms]
    x_shifted = x[perms_shift]
    y_shifted = y[perms_shift]

    # Calculate distances using Euclidean distance formula
    distances = np.sqrt((x_shifted - x_new) ** 2 + (y_shifted - y_new) ** 2)

    # Return the sum of distances for each permutation along axis 1
    return np.sum(distances, axis=1)


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
