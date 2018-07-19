'''
SYNBIOCHEM (c) University of Manchester 2018

SYNBIOCHEM is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
from sklearn.cluster import MiniBatchKMeans

import numpy as np
import pandas as pd


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


def standardise(df, mz_stnd):
    '''Apply internal standard.'''
    groups = []

    for _, group in df.groupby(['x', 'y']):
        i_stnd = group[np.isclose(group['mz'], mz_stnd, 0.000001)]['i']

        if not i_stnd.empty:
            group['i_stnd'] = group['i'] / i_stnd.iloc[0]
            groups.append(group)

    return pd.concat(groups)
