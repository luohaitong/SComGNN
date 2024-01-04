#!/bin/bash


dataset=Appliances
mkdir ../euler_data
mkdir ../euler_data/$dataset

mkdir stats 
mkdir data 
mkdir tmp
mkdir embs
mkdir imgs

echo "---------------- step 1: feature filter ----------------"
python 1_feature_filter.py $dataset
echo "--------------------------------------------------------"

echo "---------------- step 2: edge extraction ---------------"
python 2_edge_extractor.py $dataset
echo "--------------------------------------------------------"

echo "---------------- step 3: edge filter -------------------"
python 3_edge_filter.py $dataset
echo "--------------------------------------------------------"

echo "---------------- step 4: data formulation --------------"
python 4_data_formulator.py $dataset
echo "--------------------------------------------------------"

echo "---------------- step 5: category embedding --------------"
python 5_category_embs.py $dataset
echo "--------------------------------------------------------"
