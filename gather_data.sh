#!/bin/bash

echo 'Source data directory' $1

echo 'Attempring to create'  $2
mkdir -p $2

collector='/feature_collector/collector'
if [ ! -f $FILE ]; then
   make collector
fi

uuid=$(uuidgen)
temporary_file_path='data/'$uuid/
echo 'create dir at' $temporary_file_path
mkdir $temporary_file_path

python data_manupulations/size_unification.py $1 $temporary_file_path

shopt -s nullglob
csvfiles=($temporary_file_path/*.csv)

echo ./feature_collector/collector $temporary_file_path $2 ${#csvfiles[@]}

echo $temporary_file_path 'removed'
rm -rf $temporary_file_path


