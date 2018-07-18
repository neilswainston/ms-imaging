'''
SYNBIOCHEM (c) University of Manchester 2018

SYNBIOCHEM is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
import sys

from sklearn.cluster import MiniBatchKMeans

import numpy as np
import pandas as pd


def read(filename):
    '''Read 'raw' data.'''
    data = []

    with open(filename) as fle:
        for line_num, line in enumerate(fle):
            if line_num == 3:
                mzs = [float(mz) for mz in line.split()]
            elif line_num > 3:
                scan = line.split()

                # Add data as x, y, mz, intensity:
                data.extend([(float(scan[1]), float(scan[2]), mz, i)
                             for mz, i in zip(mzs,
                                              [float(val)
                                               for val in scan[3:-2]])
                             if i > 0])

    return pd.DataFrame(data, columns=['x', 'y', 'mz', 'i'])


def cluster(df, n_clusters=384, verbose=0):
    '''Cluster.'''
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             init_size=n_clusters + 1,
                             verbose=verbose).fit(df[['x', 'y']])

    df['cluster'] = kmeans.labels_

    centres = zip(*kmeans.cluster_centers_)
    df['cluster_centre_x'] = df['cluster'].apply(lambda x: centres[0][x])
    df['cluster_centre_y'] = df['cluster'].apply(lambda y: centres[1][y])
    df['distance_from_centre'] = \
        np.linalg.norm(df[['x', 'y']].values -
                       df[['cluster_centre_x', 'cluster_centre_y']].values,
                       axis=1)

    return df


def main(args):
    '''main method.'''
    df = read(args[0])
    df = cluster(df, verbose=1)
    df.to_csv(args[1], index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
