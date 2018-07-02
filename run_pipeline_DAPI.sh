#!/usr/bin/env bash
DIRNAME=$1

FILES=$(find $DIRNAME -type f -name "*.tif" -not -name "*processed*.tif")
for f in $FILES; do
    echo "\n\n"
    echo "== Analyzing $f"
    python3 ./gen_eye_brain_normalized.py $f
done

echo "\n\nDone processing raw TIFF files"

echo "== Ananlyzing eyes"
EYEFILES=$(find . -type f -name "*processed*eye*.tif")
python3 ./gen_heatmap.py $EYEFILES

echo "== Ananlyzing brain"
EYEFILES=$(find . -type f -name "*processed*brain*.tif")
python3 ./gen_heatmap.py $EYEFILES

