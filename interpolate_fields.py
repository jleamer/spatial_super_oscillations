import numpy as np
from scipy.interpolate import griddata
from scipy.integrate import simps
import pandas as pd


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


def interpolate_data(field_df):
    # Create meshgrid
    xmin = field_df.X.min()
    xmax = field_df.X.max()
    ymin = field_df.Y.min()
    ymax = field_df.Y.max()
    x, y = np.linspace(xmin, xmax, 10000), np.linspace(ymin, ymax, 10000)
    X, Y = np.meshgrid(x, y)

    # Now load in field data
    Ex = field_df['Ex'].to_numpy(dtype=complex)
    Ey = field_df['Ey'].to_numpy(dtype=complex)
    Ez = field_df['Ez'].to_numpy(dtype=complex)

    # Interpolate
    interp_Ex = griddata((field_df.X.to_numpy(), field_df.Y.to_numpy()), Ex, (X, Y), method='nearest')
    interp_Ey = griddata((field_df.X.to_numpy(), field_df.Y.to_numpy()), Ey, (X, Y), method='nearest')
    interp_Ez = griddata((field_df.X.to_numpy(), field_df.Y.to_numpy()), Ez, (X, Y), method='nearest')

    return interp_Ex, interp_Ey, interp_Ez, x, y


def calc_transmittance(ex, ey, ez, pos):
    trans_int = ex[:][-1] @ ex[:][-1].conj() + ey[:][-1] @ ey[:][-1].conj() + ez[:][-1] @ ez[:][-1].conj()
    trans_pow = 0.5 * 3 * 8.854e-4 * simps(simps(trans_int, pos[0]*1e-3), pos[1]*1e-3)
    return trans_pow


# Fix complex number issue in data
beam_filename = "f=0.63THz_field.csv"
out_filename = "f=0.63THz_field_fixed.csv"
#fix_complex(beam_filename, out_filename)

# Read in data
cols = ['X', 'Y', 'Ex', 'Ey', 'Ez']
beam_df = pd.read_csv(out_filename, sep=',', header=None, skiprows=1, names=cols)
print(beam_df.columns)

# Interpolate fields
Ex, Ey, Ez, x, y = interpolate_data(beam_df)

# Calculate transmitted intensity
output_power = calc_transmittance(Ex, Ey, Ez, (x, y))
print(output_power)
