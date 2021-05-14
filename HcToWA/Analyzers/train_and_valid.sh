#!/bin/sh
signals=( "MHc70_MA15" "MHc70_MA40" "MHc70_MA65"
		"MHc100_MA15" "MHc100_MA25" "MHc100_MA60" "MHc100_MA95" 
		"MHc130_MA15" "MHc130_MA45" "MHc130_MA55" "MHc130_MA90" "MHc130_MA125" 
		"MHc160_MA15" "MHc160_MA45" "MHc160_MA75" "MHc160_MA85" "MHc160_MA120" "MHc160_MA155")

for sig in ${signals[@]}
do
	echo [INFO] start evaluating ${sig}...
	python ResNet.py --sig ${sig} --train >> logs/ResNet_${sig}_train.log
	python ResNet.py --sig ${sig} >> logs/ResNet_${sig}_valid.log
done
