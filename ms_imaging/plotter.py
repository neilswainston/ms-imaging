'''
SYNBIOCHEM (c) University of Manchester 2018

SYNBIOCHEM is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
from collections import Counter
import sys

import matplotlib.pyplot as plt
from ms_imaging import reader


def plot(df, target_mz):
    '''Plot data.'''
    extent = [df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()]

    matrix, max_i = _get_matrix(df, target_mz)
    plt.clf()
    plt.imshow(matrix, cmap='hot', extent=extent, vmin=0, vmax=max_i)
    plt.show()


def _get_matrix(df, target_mz):
    '''Get matrix for plotting.'''
    df['x_scale'] = _scale(df['x'])
    df['y_scale'] = _scale(df['y'])
    target_df = df[df['mz'] == target_mz]

    return target_df.pivot('y_scale', 'x_scale', 'i').fillna(0).values, \
        target_df['i'].max()


def _scale(col):
    '''Scale column.'''
    return [int(round(val)) for val in (col - col.min()) / _get_diff(col)]


def _get_diff(col, precision=8):
    '''Get difference.'''
    unique = col.unique()
    counter = Counter([round(j - i, precision)
                       for i, j in zip(unique[:-1], unique[1:])])
    return counter.most_common(1)[0][0]


def main(args):
    '''main method.'''
    df = reader.read(args[0])
    plot(df, float(args[1]))


if __name__ == '__main__':
    main(sys.argv[1:])
