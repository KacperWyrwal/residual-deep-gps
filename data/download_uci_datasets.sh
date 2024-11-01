#!/bin/bash


wget -O yacht.zip https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip
unzip yacht.zip
rm yacht.zip
mv yacht_hydrodynamics.data yacht.data


wget -O energy.zip https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip
unzip energy.zip
rm energy.zip
mv ENB2012_data.xlsx energy.xlsx


wget -O concrete.zip https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip
unzip concrete.zip
rm concrete.zip Concrete_Readme.txt
mv Concrete_Data.xls concrete.xls


wget -O kin8nm.arff https://www.openml.org/data/download/3626/dataset_2175_kin8nm.arff


wget -O power.zip https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip
unzip power.zip
rm power.zip
mv CCPP/Folds5x2_pp.xlsx power.xlsx
rm -r CCPP