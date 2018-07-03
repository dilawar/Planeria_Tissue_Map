#!/usr/bin/env python
"""gen_heatmap.py: 

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2017-, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import cv2
import helper
import pickle

def hashof_file( infiles ):
    import hashlib
    return hashlib.md5(''.join(infiles).encode('utf-8')).hexdigest()

def rescale_width( frame, width ):
    print( frame.shape, frame.max() )
    return helper.scale_frame_width( frame, int(width))

def preprocess( fs, outfile = None ):
    signal = fs[:,:,:,0]
    outline = fs[:,:,:,1]
    meanF = np.uint8( np.mean( signal, axis=0 ))
    meanO = np.uint8( np.mean( outline, axis=0))
    return meanF, meanO

def generate_heatmap( data, prefix ):
    plt.figure( figsize=(8,12) )
    ax1 = plt.subplot( 121 )

    meanFs, meanOs = zip(*data)
    nAnimals = len(meanFs)
    print( "[INFO ] There are %d animals" % nAnimals )
    

    signal, outline = np.mean(meanFs, axis=0), np.mean(meanOs, axis=0)

    # Plot these values.
    im = ax1.imshow( signal + 0.1*outline, interpolation='none' )
    plt.colorbar( im, ax=ax1, orientation = 'horizontal' )
    ax1.set_title( 'Overlap' )

    # now heatmap of probability.
    heatmapImg = np.zeros_like( signal )

    nonZeroPS = np.where( signal > 0)

    for x, y in zip(*nonZeroPS):
        count = 0
        for f in meanFs:
            if f[x,y] > 0:
                count += 1
        heatmapImg[x,y] = count / nAnimals

    ax2 = plt.subplot( 122 )
    im =ax2.imshow( heatmapImg + 0.1*outline/outline.max(), interpolation = 'none' 
            , vmin=0, vmax=1
            )
    ax2.set_title( 'Probability' )
    plt.colorbar( im, ax=ax2, orientation='horizontal' )
    plt.tight_layout()
    plt.suptitle( 'Total animals=%d' % nAnimals)
    outfile = '%s_heatmap.png' % prefix 
    plt.savefig( outfile )
    print( '--> Generated heatmap to %s' % outfile )
    # save data as pickle.
    with open( '%s.pickle' % prefix, 'wb' ) as f:
        pickle.dump( (heatmapImg, signal, outline), f) 

def main( ):
    infiles = sys.argv[1:]
    outfileprefix = hashof_file(infiles)

    print( "[INFO ] Got following files: \n\t%s" % '\n\t'.join(infiles) )
    data = [ tifffile.imread(f) for f in infiles ]

    plt.figure( figsize=(len(infiles)*3,6) )
    final = []
    for i, fs in enumerate( data ):
        ax = plt.subplot( 2, len(data), i+1)
        meanF, outline = preprocess( fs )
        final.append( (meanF, outline) )
        im = ax.imshow( meanF + outline, interpolation = 'none' ) #, aspect='auto')
        plt.colorbar( im, ax=ax )
        ax.axis('off')

    # Now analyze the final results.
    # a) Find the mean width.
    widths = [ x[0].shape[1] for x in final ]
    meanW = int(np.mean( widths ))
    normalizedData = []
    with open( '%s_summary.pickle' % outfileprefix, 'wb') as f:
        pickle.dump( final, f )

    for i, (meanF, meanO) in enumerate(final):
        ax = plt.subplot( 2, len(data), len(data)+i+1)
        normalizedData.append( 
                (rescale_width(meanF, meanW), rescale_width(outline, meanW))
                )
        im = ax.imshow( np.sum(normalizedData[-1], axis=0), interpolation='none')
        plt.colorbar( im, ax=ax )
        ax.axis('off')
        ax.set_title( 'Normalized' )

    plt.tight_layout( )
    plt.savefig( '%s_summary.png' % outfileprefix )
    plt.close()

    generate_heatmap( normalizedData, outfileprefix )
    print( 'All done')
    

if __name__ == '__main__':
    main()
