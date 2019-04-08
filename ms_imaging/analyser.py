'''
SYNBIOCHEM (c) University of Manchester 2018

SYNBIOCHEM is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
import math

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd


def background_subtract(df):
    '''Perform background subtract.'''
    mz_data = []
    num_spectra = len(df[['x', 'y']].drop_duplicates())

    for mz, group in df.groupby('mz'):
        mz_data.append([mz, len(group) / num_spectra])

    mz_df = pd.DataFrame(mz_data, columns=['mz', 'frequency'])

    return df.join(mz_df.set_index('mz'), on='mz')


def cluster(df, n_clusters=384, verbose=0):
    '''Cluster.'''
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             init_size=n_clusters + 1,
                             verbose=verbose).fit(df[['x', 'y']])

    df['cluster'] = kmeans.labels_

    centres = list(zip(*kmeans.cluster_centers_))
    df['cluster_centre_x'] = df['cluster'].apply(lambda x: centres[0][x])
    df['cluster_centre_y'] = df['cluster'].apply(lambda y: centres[1][y])
    df['distance_from_centre'] = \
        np.linalg.norm(df[['x', 'y']].values -
                       df[['cluster_centre_x', 'cluster_centre_y']].values,
                       axis=1)

    return df


def filter_bkg(df, threshold=0.05):
    '''Apply background filter.'''
    filtered = []

    for _, group in df.groupby(['x', 'y']):
        filtered.append(group[(group['i'] / group['i'].max()) > threshold])

    return pd.concat(filtered)


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


def quantify(df, mz_target, tol=20):
    '''Quantify drops.'''
    quantified = []

    scaler = MinMaxScaler()
    df['norm_distance_from_centre'] = \
        scaler.fit_transform(df['distance_from_centre'].values.reshape(-1, 1))

    for _, group in df.groupby('cluster'):
        err = mz_target * tol * 10 ** -6
        quant_df = group[(group['mz'] > mz_target - err) &
                         (group['mz'] < mz_target + err)]

        if not quant_df.empty:
            weights = 1 - quant_df['norm_distance_from_centre']
            mean_i = np.average(quant_df['i'], weights=weights)
            std_i = math.sqrt(np.average((quant_df['i'] - mean_i)**2,
                                         weights=weights))

            group['mean_i'] = mean_i
            group['std_i'] = std_i
            quantified.append(group)

    return pd.concat(quantified)
