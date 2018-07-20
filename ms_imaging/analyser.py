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


def standardise(df, mz_stnd, tol=20):
    '''Apply internal standard.'''
    err = mz_stnd * tol * 10 ** -6
    stnd_df = df[(df['mz'] > mz_stnd - err) & (df['mz'] < mz_stnd + err)]
    stnd_df = pd.merge(df, stnd_df,
                       on=['x', 'y'],
                       suffixes=['_raw', '_stnd'])

    stnd_df.rename(columns={'mz_raw': 'mz'}, inplace=True)
    stnd_df['i'] = stnd_df['i_raw'] / stnd_df['i_stnd']

    return stnd_df
