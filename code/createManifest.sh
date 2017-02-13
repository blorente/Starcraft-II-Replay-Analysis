#!/bin/bash
DEST="$1/manifest.m"
echo $DEST
rm $DEST

for file in $1/*.m
do
  echo "source('$file');" >> $1/manifest.m
done
