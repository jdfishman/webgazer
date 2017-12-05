#
# CSCI 1430 Webgazer Project
# james_tompkin@brown.edu
#

import os
import csv
import cv2

dirToView = "./P_1/1491423217564_2_-study-dot_test_instructions_frames/"
dataFile = "gazePredictions.csv"

with open( dirToView + dataFile ) as f:
    readCSV = csv.reader(f, delimiter=',')
    for row in readCSV:

        frameFilename = row[0]
        frameTimestamp = row[1]
        # Tobii has been calibrated such that 0,0 is top left and 1,1 is bottom right on the display.
        tobiiLeftEyeGazeX = float( row[2] )
        tobiiLeftEyeGazeY = float( row[3] )
        tobiiRightEyeGazeX = float( row[4] )
        tobiiRightEyeGazeY = float( row[5] )
        webgazerX = float( row[6] )
        webgazerY = float( row[7] )
        clmTracker = row[8:len(row)-1]
        clmTracker = [float(i) for i in clmTracker]
        clmTrackerInt = [int(i) for i in clmTracker]

        tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
        tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2

        img = cv2.imread( frameFilename )
        
        print( "WebGazer: {:6.4f} {:6.4f}  |  Tobii: {:6.4f} {:6.4f}".format( webgazerX, webgazerY, tobiiEyeGazeX, tobiiEyeGazeY ) )

        # Draw clmTracker points
        # [James] I'm sure clmTracker has documentation for this, but I just randomly poked until it was roughly right to give you an idea.

        # Jaw
        for i in range(0,28,2):
            cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)
        
        # Right eyebrow
        for i in range(30,36,2):
            cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

        # Left eyebrow
        for i in range(38,44,2):
            cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

        # Upper left eye
        for i in range(46,50,2):
            cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

        # Middle of left eye
        for i in range(54,56,2):
            cv2.circle(img, (clmTrackerInt[i],clmTrackerInt[i+1]), 4, (255,0,0), -4 )        

        # Upper right eye
        for i in range(56,60,2):
            cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

        # Middle of right eye
        for i in range(64,66,2):
            cv2.circle(img, (clmTrackerInt[i],clmTrackerInt[i+1]), 4, (255,0,0), -4 ) 
        
        # Nose
        for i in range(68,80,2):
            cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

        # Upper lip
        for i in range(88,100,2):
            cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

        # Lower lip
        for i in range(102,110,2):
            cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

        # All points
        for i in range(0,len(clmTrackerInt)-1,2):
            cv2.circle(img, (clmTrackerInt[i],clmTrackerInt[i+1]), 2, (0,0,255), -2 )

        cv2.imshow( 'viewer.py', img )
        cv2.waitKey(0)
