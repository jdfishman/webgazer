#
# CSCI 1430 Webgazer Project
# james_tompkin@brown.edu
#

import os, glob, random

#
# Split files into training and test.
#
# We're going to randomly select some participants.
probParticipant = 0.86
# Then, for each participant, we're going to randomly select some task folders.
probTask = 0.86
# Within each task folder, it's probably save to ignore randomization
# Further, if anyone wanted to try and exploit temporal consistency, then you could go back and try to extract every frame sequentially and exploit this.

# That's it.

# We'll also count the number of files in each set just to make sure it looks about right
nTrainImages = 0
nTestImages = 0

trainFile = "train.txt" 
testFile = "test.txt"

trf = open( trainFile, "w", newline='')
tef = open( testFile, "w", newline='')

# Gather all P_x directories
participantDirs = glob.glob("./*/")

for f in participantDirs:
    # Gather all frame folders within P_x directory
    pDir = glob.glob( "./" + f + "/*/")
    
    r = random.uniform(0,1)

    # Whole participant is in test
    if r > probParticipant:
        print( "Whole participant in test: " + f )
        for f2 in pDir:
            tef.write( os.path.normpath(f2) + '\n')
            nTestImages = nTestImages + len(glob.glob( "./" + f2 + "/*.png"))
    # Only parts of participant are in test
    else:
        for f2 in pDir:
            r = random.uniform(0,1)

            # Specific task is in test set
            if r > probTask:
                tef.write( os.path.normpath(f2) + '\n')
                nTestImages = nTestImages + len(glob.glob( "./" + f2 + "/*.png"))
            else:
                trf.write( os.path.normpath(f2) + '\n')
                nTrainImages = nTrainImages + len(glob.glob( "./" + f2 + "/*.png"))

trf.close()
tef.close()

print( "Num train images: " + str(nTrainImages) + " | Num test images: " + str(nTestImages) )