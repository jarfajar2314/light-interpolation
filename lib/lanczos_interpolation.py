import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin


def lanczos_kernel(x, a):
    if (x == 0):
        return 1
    elif (x >= (a * -1) and x <= a):
        return (a * math.sin(math.pi*x) * math.sin(math.pi * x / a)) / ((math.pi**2) * (x**2))
    else:
        return 0


def clamp(x, b):
    if (x < 0):
        return 0
    elif (x > b):
        return b
    else:
        return x


def lanczos_interpolation(x, s, a):
    result = 0
    for i in range(math.floor(x) - a + 1, min(math.floor(x) + a + 1, len(s))):
        l_i = s[clamp(i, len(s)-1)] * lanczos_kernel((x - i), a)
        result += l_i
    result = round(result, 10)
    return result


def resample_array(array, new_size):
    resampled_array = []
    original_size = len(array)

    # Iterate over the new array indices
    for i in range(new_size):
        # Calculate the corresponding position in the original array
        position = (i / (new_size - 1)) * (original_size - 1)

        interpolated_value = lanczos_interpolation(position, array, 2)

        # Append the interpolated value to the resampled array
        resampled_array.append(interpolated_value)
    return resampled_array


def row_interpolation(array, new_length):
    new_array = []
    array = np.array(array)
    for hrow in tqdm(range(len(array))):
        interpolated_array = resample_array(array[hrow], new_length)
        new_array.append(interpolated_array)

    new_array = np.array(new_array)
    return new_array


def image_interpolation(array, new_x, new_y):
    result = []
    print("array:", array.shape)
    result = row_interpolation(array, new_y)
    print("result:", result.shape)
    transposed_result = np.transpose(result)
    print("transposed:", transposed_result.shape)
    final_result = row_interpolation(transposed_result, new_x)
    final_result = np.transpose(final_result)
    print("result:", final_result.shape)
    return final_result


def export_to_image(array, save_filename):
    # Set the color map to 'viridis'
    cmap = plt.get_cmap('viridis')

    # Create the plot without any legend
    plt.imshow(array, cmap=cmap)

    # # Hide the axes
    # plt.axis('off')

    # Invert the y-axis
    plt.gca().invert_yaxis()

    # Save the plot as a PNG file with a DPI of 300
    plt.savefig(save_filename + '.png', dpi=300,
                bbox_inches='tight', pad_inches=0)


def convert_to_df(df, array, multiplier, save_csv=False, save_filename=""):
    top_left_lat = df['y'].max()
    top_left_lon = df['x'].min()
    x_unique = df['x'].unique()
    y_unique = df['y'].unique()
    res_lat = round(y_unique[0] - y_unique[1], 7) / multiplier
    res_lon = round(x_unique[1] - x_unique[0], 7) / multiplier

    # Get the dimensions of the array
    rows, cols = array.shape

    # Generate new coordinates
    latitudes = top_left_lat - np.repeat(np.arange(rows), cols) * res_lat
    longitudes = top_left_lon + np.tile(np.arange(cols), rows) * res_lon

    latitudes = np.flip(latitudes)

    # Flatten the array into 1D for the 'val' column
    vals = array.flatten()

    # Create a pandas dataframe from the 'x', 'y' and 'val' arrays
    df = pd.DataFrame({
        'x': longitudes,
        'y': latitudes,
        'val': vals
    })

    # Save the dataframe to a CSV file
    if (save_csv == True):
        df.to_csv('output/' + save_filename + '.csv', index=False)
    return df


def save_to_tif(array, df, multiplier, filename):
    top_left_lat = df['y'].max()
    top_left_lon = df['x'].min()
    x_unique = df['x'].unique()
    y_unique = df['y'].unique()
    res_lat = round(y_unique[0] - y_unique[1], 7) / multiplier
    res_lon = round(x_unique[1] - x_unique[0], 7) / multiplier

    # Create the affine transform
    transform = from_origin(top_left_lon, top_left_lat, res_lon, res_lat)

    # Define the coordinate reference system (CRS)
    crs = "EPSG:4326"  # WGS84, you may need to adjust this

    # Save the array as a GeoTIFF
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array.astype(rasterio.float32), 1)


def pixel2coord(col, row, transform):
    lon, lat = transform * (col, row)
    return lon, lat


def tif_to_csv(filename):
    # Open the GeoTIFF file
    with rasterio.open("data/" + filename) as src:
        # Get the GeoTIFF transform parameters
        transform = src.transform

        # Read the GeoTIFF data as a 2D array
        data = src.read(1)

    # Get the rows, cols and vals (ignoring NoData values)
    rows, cols = np.where(data != src.nodata)
    vals = np.extract(data != src.nodata, data)

    # Convert pixel positions to longitude, latitude
    coords = [pixel2coord(col, row, transform) for col, row in zip(cols, rows)]
    lons, lats = zip(*coords)

    # Create a pandas dataframe from the longitude, latitude and value arrays
    df = pd.DataFrame({
        'x': lons,
        'y': lats,
        'val': vals
    })

    # Save dataframe to CSV
    new_filename = filename.split('.')[0] + ".csv"
    df.to_csv("data/" + new_filename, index=False)
    print("Saved to data/" + new_filename)
    return new_filename
