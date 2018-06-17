"""helper.py: 

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2017-, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import pandas as pd
import numpy as np
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    plt.style.use('classic' )
except Exception as e:
    pass


def lines_to_dataframe( lines ):
    cols = 't1,t2,s1,a,b,c,d,e,f,g,h,status,sig1,sig2'.split(',')
    d = pd.read_csv( io.StringIO(lines), sep = ',', names = cols
            , parse_dates = [ 't2', 't1'] )
    # Drop invlid lines.
    d = d.dropna()
    return d 

def get_time_slice( df, status ):
    f = df[df['status'] == status]['t1'].values
    if len(f) > 2:
        return f[0], f[-1]
    return 0, 0

def _max(a, b):
    if a is None:
        return b
    if b is None:
        return a 
    return max(a,b)

def _min(a, b):
    if a is None:
        return b
    if b is None:
        return a 
    return min(a,b)

def _interp( x, x0, y0 ):
    return np.interp(x, x0, y0)


def pad_frame( f, pad, color):
    return np.pad( f, pad_width=pad, mode='constant', constant_values=color)

def pad_frames( frames, pad = 20, color = 255 ):
    return [ pad_frame(f, pad, color) for f in frames ]

def save_frames( frames, outfile ):
    padded = [ pad_frame(f) for f in frames ]
    if len(frames) == 3:
        save_frame( np.dstack( frames ), outfile )
    else:
        save_frame( np.hstack( frames ), outfile )

def create_grid( frame, step ):
    r, c = frame.shape 
    for i in range(0, r, step):
        frame[i,:] = 50
    for i in range(0, c, step):
        frame[:,i] = 50
    return frame

def save_frames( frames, outfile ):

    plt.figure()
    for i, frame in enumerate(frames):
        ax = plt.subplot( 1, len(frames), i+1 )
        ax.imshow( frame, interpolation = 'none', aspect = 'auto' )

    plt.tight_layout( )
    plt.savefig( outfile )
    plt.close( )
    
