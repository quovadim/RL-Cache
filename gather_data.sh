#!/bin/bash

echo 'Source data directory' $1

echo 'Attempring to create'  $2
mkdir -p $2

uuid=$(uuidgen)
temporary_file_path=data/$uuid/
echo $temporary_file_path created
mkdir $temporary_file_path

shopt -s nullglob
csvfiles=($1/*.csv)

echo './feature_collector/collector' $1 $temporary_file_path ${#csvfiles[@]}
./feature_collector/collector $1 $temporary_file_path ${#csvfiles[@]}

echo './reward_collector/collector' $temporary_file_path $2 ${#csvfiles[@]}
./reward_collector/collector $temporary_file_path $2 ${#csvfiles[@]}

echo $temporary_file_path 'removed'
rm -rf $temporary_file_path


