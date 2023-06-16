import numpy as np
from scipy.interpolate import griddata
from scipy.integrate import simps
import pandas as pd
from multiprocessing import Pool
import itertools as it


def fix_complex(filename, output):
    """
    Function for changing i to j in Comsol data to match Python conventions for complex numbers
    :param filename: name of file to be corrected
    :param output:   name of file to output to
    :return:
    """
    dict = {}
    df = pd.read_csv(filename, sep=',', header=8)
    cols = df.columns
    print(cols)
    for col in cols:
        temp = df[col].to_numpy()
        if isinstance(temp[0], str):
            temp = np.array([temp[n].replace('i', 'j') for n in range(temp.size)])
        dict[col] = temp

    outfile = pd.DataFrame(dict, columns=cols)
    outfile.to_csv(output)
    return


def interpolate_data(args):
    field_df = args[0]
    component = args[1]

    # Create meshgrid
    xmin = field_df.X.min()
    xmax = field_df.X.max()
    ymin = field_df.Y.min()
    ymax = field_df.Y.max()
    x, y = np.linspace(xmin, xmax, 10000), np.linspace(ymin, ymax, 10000)
    X, Y = np.meshgrid(x, y)

    # Now load in field data
    E = field_df[component].to_numpy(dtype=complex)

    # Interpolate
    interp_E = griddata((field_df.X.to_numpy(), field_df.Y.to_numpy()), E, (X, Y), method='nearest')

    return interp_E, x, y


def calc_transmittance(ex, ey, ez, x):
    trans_int = ex[:, -1] * ex[:, -1].conj() + ey[:, -1] * ey[:, -1].conj() \
    + ez[:, -1] * ez[:, -1].conj()
    print(trans_int.shape)
    trans_pow = 0.5 * 3 * 8.854e-4 * simps(trans_int, x*1e-3)
    return trans_pow


# Fix complex number issue in data
beam_filenames = ["f=0.52THz_field.csv", "f=0.63THz_field.csv", "f=0.72THz_field.csv", "f=0.82THz_field.csv"]
out_filenames = ["f=0.52THz_field_fixed.csv", "f=0.63THz_field_fixed.csv",
                 "f=0.72THz_field_fixed.csv", "f=0.82THz_field_fixed.csv"]
for _ in zip(beam_filenames, out_filenames):
    fix_complex(_[0], _[1])

# Read in data
cols = ['X', 'Y', 'Ex', 'Ey', 'Ez']
beam_dfs = [pd.read_csv(_, sep=',', header=None, skiprows=1, names=cols) for _ in out_filenames]

# Interpolate fields
components = ['Ex', 'Ey', 'Ez']
args = list(it.product(beam_dfs, components))
with Pool() as p:
    results = list(p.map(interpolate_data, args))

interp_files = ["f=0.52THz_interp.npz", "f=0.63THz_interp.npz", "f=0.72THz_interp.npz", "f=0.82THz_interp.npz"]
count = 0
for _ in interp_files:
    np.savez(
        _,
        X=results[count][1],
        Y=results[count][2],
        Ex=results[count][0],
        Ey=results[count + 1][0],
        Ez=results[count + 2][0]
    )
    count += 3

