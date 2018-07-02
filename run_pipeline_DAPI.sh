#!/usr/bin/env bash
DIRNAME=$1

FILES=$(find $DIRNAME -type f -name "*.tif" -not -name "*processed*.tif")
for f in $FILES; do
    echo "\n\n"
    echo "== Analyzing $f"
    python3 ./gen_eye_brain_normalized.py $f
done

echo "\n\nDone processing raw TIFF files"

EYEFILES=$(find $DIRNAME -type f -name "*processed.eye.tif")
echo "== Ananlyzing eyes $EYEFILES"
python3 ./gen_heatmap.py $EYEFILES

BRAINFILES=$(find $DIRNAME -type f -name "*processed.brain.tif")
echo "== Ananlyzing brain $BRAINFILES"
python3 ./gen_heatmap.py $BRAINFILES

