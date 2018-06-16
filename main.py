#!/usr/bin/env python

import os
import sys
from PIL import Image, ImageSequence
import numpy as np
import itertools
import pickle
import helper
import cv2

cap_ = None

resdir_name_ = '_results'
datadir_     = None
infile_      = None
frames_      = []

brightFieldFrame_   = []
tissueFrames_       = []

def show_frame( frame, delay = 1 ):
    cv2.imshow( "Planaria", frame )
    cv2.waitKey( delay )

def save_frame( frame, outfile = 'a.png' ):
    cv2.imwrite( outfile, frame )
    print( '[INFO] Wrote frame to %s' % outfile )

def read_frames( infile ):
    global frames_
    with Image.open( infile ) as f:
        for i, page in enumerate(ImageSequence.Iterator(f)):
            frame = np.array( page )
            frame = np.array( frame // 2**8, dtype = np.uint8 )
            frames_.append( frame )
            if i % 2 == 1:
                brightFieldFrame_.append(frame)
            else:
                tissueFrames_.append(frame)
    print( '[INFO] Total %s frames read' % len(frames_))

def find_markers( frames ):
    f = np.mean( frames, axis = 0 )
    u, s = np.mean(f), np.std(f)

    # Threshold.
    f[f < u + 2*s ] = 0
    f[ f > u + s ] = 255

    f = np.uint8( f )

    # Loose few pixels.
    kernel = np.ones( (7,7), np.uint8 )
    f = cv2.morphologyEx( f, cv2.MORPH_OPEN, kernel )
    return f

def open_morph(f, times = 1):
    kernel = np.ones( (5,5), np.uint8 )
    for i in range(times):
      f = cv2.morphologyEx( f, cv2.MORPH_OPEN, kernel )
    return f

def find_orientation( frame ):
    #  pts = np.where( frame > 1 )
    #  pts = np.array( [ list(x) for x in list(zip(*pts))] )
    #  ellipse = cv2.fitEllipse( pts )
    #  cv2.ellipse(frame, ellipse, 255, 2)
    #  print( ellipse )

    frame = open_morph( frame )

    m, u = np.mean(frame), np.std(frame)
    frame[ frame > m + 0.5  ] = 0
    img, cnts, h = cv2.findContours( frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

    f = np.zeros_like(frame)
    goodCnts = []
    largestCnts = sorted( [ (cv2.contourArea(c), c) for c in cnts ] )[-1][1]

    cv2.drawContours( f, [largestCnts], -1, 255, 3 )
    save_frame( f, "temp_cnts.png" )


    outlineFile = '%s.outline.dat' % infile_ 
    with open(  outlineFile, 'w' ) as f:
        for x in largestCnts:
            x = x[0]
            f.write( '%d %d\n' % (x[0], x[1] ) )
    print( '[INFO] Wrote outline of animal to %s' % outlineFile )

    return 0

def find_animal_shape( frames ):
    f = np.mean( brightFieldFrame_, axis = 0 )
    f = f / f.max()
    f = f - f.mean()
    f[ f < 0 ] = 0
    f = 255 * f / f.max()
    #for i in range(1):
    #    f = cv2.morphologyEx( f, cv2.MORPH_OPEN, kernel )

    f = cv2.blur(f, (21,21) )

    save_frame( f, "aniaml_shape.png" )
    return np.uint8( f )


def run( infile, ignore_pickle = False ):
    global datadir, infile_
    global frames_
    infile_ = infile
    read_frames( infile )

    #  meanMarkerFrame = find_markers( tissueFrames_ )
    f = find_animal_shape( brightFieldFrame_ )
    theta = find_orientation( f )
    print( "[INFO ] Theta is %g" % theta )
    

    
def main():
    global infile_
    infile_ = sys.argv[1]
    run( infile_ )

if __name__ == '__main__':
    main()
