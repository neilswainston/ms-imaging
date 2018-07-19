'''
SYNBIOCHEM (c) University of Manchester 2018

SYNBIOCHEM is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
from collections import Counter
from itertools import product
import sys

import matplotlib.pyplot as plt
from ms_imaging import reader
import numpy as np
import pandas as pd


def plot(df, target_mz):
    '''Plot data.'''
    extent = [df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()]

    matrix, max_i = _get_matrix(df, target_mz)
    plt.clf()
    plt.imshow(matrix, cmap='hot', extent=extent, vmin=0, vmax=max_i)
    plt.show()


def _get_matrix(df, target_mz):
    '''Get matrix for plotting.'''
    matrix = list(product(_get_range(df['x']), _get_range(df['y'])))
    matrix_df = pd.DataFrame(matrix, columns=['x', 'y'])

    # Add x,y scale:
    matrix_df['x_scale'] = _scale(matrix_df['x'])
    matrix_df['y_scale'] = _scale(matrix_df['y'])

    target_df = df[df['mz'] == target_mz]

    matrix_df = _merge_float(matrix_df, target_df, ['x', 'y'])

    return matrix_df.pivot('y_scale', 'x_scale', 'i').fillna(0).values, \
        matrix_df['i'].max()


def _get_range(col):
    '''Get range.'''
    return np.arange(col.min(), col.max(), _get_diff(col))


def _scale(col):
    '''Scale column.'''
    return ((col - col.min()) / _get_diff(col)).apply(lambda x: int(round(x)))


def _merge_float(left, right, on):
    '''Merge dataframes by float column(s).'''
    for df, col in product([left, right], on):
        df.loc[:, col] = _multiply(df[col])

    left = pd.merge(left, right, how='left', on=on)

    for df, col in product([left, right], on):
        df.loc[:, col] = _divide(df[col])

    return left


def _get_diff(col, precision=8):
    '''Get difference.'''
    unique = col.unique()
    counter = Counter([round(j - i, precision)
                       for i, j in zip(unique[:-1], unique[1:])])
    return counter.most_common(1)[0][0]


def _multiply(col, precision=8):
    '''Multiple column, for float->int conversion.'''
    return np.round(col * 10 ** precision).astype(int)


def _divide(col, precision=8):
    '''Divide column, for int->float conversion.'''
    return np.round(col / (10 ** precision))


def main(args):
    '''main method.'''
    df = reader.read(args[0])
    df.to_csv(args[2], index=False)
    plot(df, float(args[1]))


if __name__ == '__main__':
    main(sys.argv[1:])
