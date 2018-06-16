#!/usr/bin/env python

import os
import sys
from PIL import Image, ImageSequence
import numpy as np
import itertools
import pickle
import math
import helper
import cv2

cap_ = None

resdir_name_ = '_results'
datadir_     = None
infile_      = None
frames_      = []

if not os.path.isdir( resdir_name_ ):
    os.makedirs( resdir_name_ )

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

def ignore_neighbours( vec, min_distance = 10 ):
    yvec = list(vec[:])
    newvec = [yvec.pop()]
    while yvec:
        y = yvec.pop()
        if abs(y-newvec[-1]) < min_distance:
            continue
        newvec.append(y)
    return newvec


def find_outline( frame ):
    frame = open_morph( frame )
    m, u = np.mean(frame), np.std(frame)
    frame[ frame > m + 0.5  ] = 0
    img, cnts, h = cv2.findContours( frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

    f = np.zeros_like(frame)
    goodCnts = []
    largestCnts = sorted( [ (cv2.contourArea(c), c) for c in cnts ] )[-1][1]

    cv2.drawContours( f, [largestCnts], -1, 255, 1 )
    save_frame( f, "temp_cnts.png" )

    outlineFile = '%s.outline.dat' % infile_ 
    with open(  outlineFile, 'w' ) as h:
        for x in largestCnts:
            x = x[0]
            h.write( '%d %d\n' % (x[0], x[1] ) )
    print( '[INFO] Wrote outline of animal to %s' % outlineFile )

    return f

def find_animal( frames ):

    fsum = np.sum( frames, axis = 0 )
    fsum = 255 * fsum / fsum.max()
    f0 = np.mean( frames, axis = 0 )
    f0 = f0 / f0.max()
    f0 = f0 - f0.mean()
    f0[ f0 < 0 ] = 0
    f0 = 255 * f0 / f0.max()

    f = cv2.blur(f0, (11,11) )

    save_frame( np.hstack((fsum,f)), "aniaml_shape.png" )
    return np.uint8( f )

def rotate_by_theta( img, theta ):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def compute_midline_and_rotation( outline ):
    xvec, midpoints = [], []
    for i, row in enumerate(outline):
        pts = np.where( row==255 )[0]
        if len(pts) == 0:
            continue
        pts = ignore_neighbours( pts )
        if len(pts) == 0:
            continue

        midP = int(np.mean( pts ))
        xvec.append(i)
        midpoints.append( midP )
        #  outline[i, midP] = 100

    m, c = np.polyfit( xvec, midpoints, 1 )
    theta = - 180*math.atan(m)/math.pi
    print( "[INFO ] Rotate by m=%f. Rotate by %f deg" % (m, theta))
    
    for x, y in zip(xvec, midpoints):
        y = int(m*x + c)
        outline[x, y] = 200

    rotated = rotate_by_theta( outline, theta )
    save_frame( np.hstack((outline,rotated)), "outline+midline.png" )
    return rotated, theta

def lame_function( outlineMidline, theta ):
    global tissueFrames_
    for i, f in enumerate(tissueFrames_):
        f = np.uint8( 255 * f / f.max() )
        print( f.min(), f.max(), f.mean() )
        newF = rotate_by_theta( f, theta )
        save_frame( newF
                , os.path.join( resdir_name_, "f%03d.png" % i )
                )

def run( infile, ignore_pickle = False ):
    global datadir, infile_
    global frames_
    infile_ = infile
    read_frames( infile )

    f = find_animal( brightFieldFrame_ )
    outline = find_outline( f )
    outlineMidline, theta = compute_midline_and_rotation( outline )

    lame_function( outlineMidline, theta )
    
    
def main():
    global infile_
    infile_ = sys.argv[1]
    run( infile_ )

if __name__ == '__main__':
    main()
