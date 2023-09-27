from lib.ordinary_kriging import OrdinaryKriging
import lib.lanczos_interpolation as lanczos

import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def interpolation_lanczos(filename, multiplier):
    if not filename.endswith('.csv'):
        csv_file = lanczos.tif_to_csv(filename)

    df = pd.read_csv('data/' + csv_file, delimiter=',', encoding='utf-8')
    array = df.pivot(index='y', columns='x', values='val')

    # Calculate the shape (length) of "x" and "y" columns
    x_length = df['x'].nunique()
    y_length = df['y'].nunique()

    # Print the shape
    print("x length:", x_length)
    print("y length:", y_length)

    # Interpolate the array
    new_x = x_length * multiplier
    new_y = y_length * multiplier
    new_array = lanczos.image_interpolation(array, new_x, new_y)

    new_df = lanczos.convert_to_df(df, new_array, multiplier)

    if not os.path.exists('output'):
        os.makedirs('output')
    # Save the array as a GeoTIFF
    lanczos.save_to_tif(new_array, df, multiplier,
                        "output/" + csv_file.split(".")[0] + "_interpolated.tif")
    print("Saved to output/" + csv_file.split(".")[0] + "_interpolated.tif")

    # Plot the interpolated array
    x_min = new_df['x'].min()
    x_max = new_df['x'].max()
    y_min = new_df['y'].min()
    y_max = new_df['y'].max()
    plt.imshow(new_array, cmap='viridis', extent=[
        x_min, x_max, y_min, y_max], origin='lower')
    plt.xticks(np.arange(x_min, x_max, 0.1))
    plt.colorbar(label='Brightness (nW/cm2/sr)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Lanczos Interpolation')
    # plt.savefig("visualization/" + save_filename + '-df', dpi=300)
    plt.show()


def interpolation_kriging(filename):
    df = pd.read_csv('data/' + filename, delimiter=',', encoding='utf-8')

    X = df['x'].values
    Y = df['y'].values
    Z = df['val'].values

    # Create the Ordinary Kriging object
    OK = OrdinaryKriging(
        X, Y, Z
    )

    # Setup Interpolation Grid
    threshold = 0.01
    size = 100
    x_range = [X.min() - threshold, X.max() + threshold]
    y_range = [Y.min() - threshold, Y.max() + threshold]
    gridx = np.linspace(x_range[0], x_range[1], size)
    gridy = np.linspace(y_range[0], y_range[1], size)
    zgrid, ss = OK.predict(gridx, gridy)

    # Visualization
    # Create the contour plot for interpolated values
    plt.contourf(gridx, gridy, zgrid, cmap='viridis_r')
    plt.colorbar(label="mag/arcsec^2")
    plt.scatter(X, Y, c='r')  # Overlay the original points
    plt.xlabel('x')  # Set labels and title
    plt.ylabel('y')
    plt.title('Ordinary Kriging')
    plt.show()


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Light pollution map interpolation using Ordinary Kriging and Lanczos.')

   # Add arguments
    parser.add_argument('-m', '--method', type=str,
                        help='Method to use "kriging" or "lanczos"')
    parser.add_argument('-f', '--file', type=str,
                        help='Input data file (e.g., "sqm-data.csv") (must be placed on the "data" folder)')
    parser.add_argument('-z', '--zoom', type=str,
                        help='Zoom multiplier, default value are 10x (only for "lanczos" method)')

    # Parse the command line arguments
    args = parser.parse_args()

    # Check if required arguments are provided
    if not args.method or not args.file:
        parser.error('Both --method and --file arguments are required.')

    if args.method == 'kriging':
        interpolation_kriging(args.file)
    elif args.method == 'lanczos':
        if not args.zoom:
            args.zoom = 10
        interpolation_lanczos(args.file, args.zoom)
    else:
        parser.error(f"Method not '{args.method}' supported.")


if __name__ == "__main__":
    main()
