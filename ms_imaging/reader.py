'''
SYNBIOCHEM (c) University of Manchester 2018

SYNBIOCHEM is licensed under the MIT License.

To view a copy of this license, visit <http://opensource.org/licenses/MIT/>.

@author:  neilswainston
'''
# pylint: disable=invalid-name
import sys


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


def main(args):
    '''main method.'''
    df = read(args[0])
    df.to_csv(args[1], index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
