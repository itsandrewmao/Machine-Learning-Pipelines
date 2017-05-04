#!/bin/bash
source config.sh;
results_folder="/$run_name/";
mkdir ~/results/$run_name;
python code/run.py data/$train_data data/$test_data --label $label --run_name $run_name --iterations $iterations --models $models --thresholds $thresholds --project_folder $results_folder >> $results_folder/$run_name.log 2>&1;
