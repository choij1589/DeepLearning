#!/bin/sh
signals=( "MHc70_MA15" "MHc70_MA40" "MHc70_MA65"
        "MHc100_MA15" "MHc100_MA25" "MHc100_MA60" "MHc100_MA95"
        "MHc130_MA15" "MHc130_MA45" "MHc130_MA55" "MHc130_MA90" "MHc130_MA125"
        "MHc160_MA15" "MHc160_MA45" "MHc160_MA75" "MHc160_MA85" "MHc160_MA120" "MHc160_MA155")
means=(1.50E+01 4.00E+01 6.50E+01
	1.50E+01 2.50E+01 6.00E+01 9.50E+01
	1.50E+01 4.50E+01 5.50E+01 9.00E+01 1.25E+02
	1.50E+01 4.50E+01 7.50E+01 8.50E+01 1.20E+02 1.55E+02)
sigmas=(1.04E-01 3.14E-01 5.75E-01
	5.56E-02 1.35E-01 4.70E-01 9.14E-01
	1.02E-01 3.29E-01 4.19E-01 7.62E-01 1.19E+00
	1.31E-01 1.76E-01 5.90E-01 7.17E-01 1.12E+00 1.59E+00)

for i in {0..17}
do
	python choosePick.py --sig ${signals[$i]} --mean ${means[$i]} --sigma ${sigmas[$i]}
done

