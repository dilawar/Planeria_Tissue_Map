#!/usr/bin/env python

import os
import sys
from PIL import Image, ImageSequence
import numpy as np
import scipy.ndimage 
import itertools
import pickle
import math
import helper
import cv2
import tifffile

cap_ = None
resdir_name_ = '_results'
datadir_     = None
infile_      = None
frames_      = []
midline_     = None


if not os.path.isdir( resdir_name_ ):
    os.makedirs( resdir_name_ )

dapiFrames_         = []
eyeFrame_           = []
brainFrame_         = []
brightFieldFrame_   = []

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
            # Its 16 bit data. Make it 8 bit.
            frame = np.array( page )
            frame = np.uint8( np.array(frame // 2**3))
            frames_.append( frame )
            if i%4 == 0:
                dapiFrames_.append( frame )
            elif i%4 == 1:
                eyeFrame_.append(frame)
            elif i%4 == 2:
                brainFrame_.append(frame)
            else:
                brightFieldFrame_.append(frame)
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

def open_morph(f, times = 1, N = 11):
    kernel = np.ones( (N,N), np.uint8 )
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
    frame[ frame < 150  ] = 0

    f = np.zeros_like(frame)
    img, cnts, h = cv2.findContours( frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

    if len(cnts) < 1:
        print( "[WARN ] No contours found. Not doing anything else." )
        return f

    goodCnts = [ (cv2.contourArea(c), c) for c in cnts ]
    largestCntsWithArea = sorted( goodCnts, key = lambda x: x[0] )[-1]
    largestCnts = largestCntsWithArea[1]

    cv2.drawContours( f, cnts, -1, 255, 1 )

    save_frame( f, "temp_cnts.png" )

    outlineFile = '%s.outline.dat' % infile_ 
    with open(  outlineFile, 'w' ) as h:
        for x in largestCnts:
            x = x[0]
            h.write( '%d %d\n' % (x[0], x[1] ) )
    print( '[INFO] Wrote outline of animal to %s' % outlineFile )
    return f

def find_animal( frames ):
    # In this method, we threhold the image and use open_morph action to get the
    # outline. Probably contour deetection will also work. But this is good
    # function. One can compare with find_outline function as well.
    # Not sure why we using this. NOTES are no longer available.
    f = np.mean( frames, axis = 0)
    f = 255* f / f.max()
    f = np.uint8( f )
    #  f = cv2.equalizeHist( fm )
    #  f = cv2.blur( fm, (13,13) )
    save_frame( f, 'mean.png' )
    print( f.mean(), f.std() )

    f[f > f.mean()] = 255
    f[ f != 255 ] = 0

    f = open_morph( f, 5, 51)
    save_frame( f, "aniaml_shape.png" )
    return f

def rotate_by_theta( img, theta ):
    # make sure NOT TO interpolate data using higher order function else rest of
    # the algorithm will break. We use pixel values in many frame to locate
    # coordinates such as midpoint, outline etc.
    #  return scipy.ndimage.rotate( img, theta, order=0, reshape = False )
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    return cv2.warpAffine(img,M,(cols,rows), flags=cv2.INTER_NEAREST)

def compute_midline_and_rotation( outline ):
    # Given outline, compute the midline. Fit it with a line and rotate the
    # frame by line angle. After rotation, we must have a straight vertical line
    # (using np.polyfit) as midline along with the midline before linear fit.
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
        outline[i, midP] = helper.midline_val_

    # save the computed midline in global.
    m, c = np.polyfit( xvec, midpoints, 1 )
    for x, y in zip(xvec, midpoints):
        y = int(m*x+c)
        #  outline[x:x+3, y:y+3] = midline_straight_val_
        outline[x, y] = helper.midline_straight_val_

    theta = - 180*math.atan(m)/math.pi
    print( "[INFO ] Rotate by m=%f. Rotate by %f deg" % (m, theta))
    
    rotated = rotate_by_theta( outline, theta )
    save_frame( np.hstack((outline,rotated)), "outline+midline+rotated.png" )

    return theta

def shift_to_align( frames, midline ):
    # Make sure to rotate outline as well. After rotation both lines may get
    # distorted and any algorithm using straight midline and midline diff
    # may not work. Do row to row comparision.
    res = []
    for frame in frames:
        newframe = helper.straighten_frame(frame, midline)
        res.append(newframe)
    assert len(res) == len(frames)
    return res


def lame_function( outline, theta, frames):
    # We are given outline and angle to rotate. Outline has not been rotated
    # yet.
    grid = helper.create_grid( np.zeros_like(outline), 50 )
    tiff = []
    for i, f in enumerate(frames):
        f = cv2.equalizeHist( f )
        # CRITICAL: remove some noise. 
        f = open_morph(f, 2, 7)

        original = [ f, outline ]

        finalFs = [rotate_by_theta( x, theta ) for x in original] 

        finalFs = shift_to_align(finalFs, rotate_by_theta(outline,theta) )

        finalFs = helper.crop_these_frames( finalFs )
        finalFs = helper.rescale( finalFs, nrows = 1000 )

        toPlot = [ np.dstack( original + [grid]) 
                , np.dstack( finalFs + [np.zeros_like(finalFs[0])] )
                ]

        helper.save_frames(toPlot
                , outfile = os.path.join( resdir_name_, "f%03d.png" % i )
                )

        finalFs.append( np.zeros_like( finalFs[0] ) )
        tiff.append(np.dstack(finalFs) )

    return np.array(tiff, dtype=np.uint8)

def run( infile, ignore_pickle = False ):
    global datadir, infile_
    global frames_
    infile_ = infile
    read_frames( infile )

    f = find_animal( brightFieldFrame_ )
    outline = find_outline( f )
    theta = compute_midline_and_rotation( outline )

    eye = lame_function( outline, theta, eyeFrame_ )
    outfile = '%s.processed.eye.tif' % infile 
    tifffile.imsave( outfile, eye )
    print( 'Done saving result. %s' % outfile )

    brain = lame_function( outline, theta, brainFrame_ )
    outfile = '%s.processed.brain.tif' % infile 
    tifffile.imsave( outfile, eye )
    print( 'Done saving result. %s' % outfile )
    
    
    
def main():
    global infile_
    infile_ = sys.argv[1]
    run( infile_ )

if __name__ == '__main__':
    main()
