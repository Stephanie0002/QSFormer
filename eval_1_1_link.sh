#!/bin/bash

# Define the data lists for each method
# "wikipedia" "uci" "reddit" "Contacts" "askUbuntu" "mooc" "lastfm" "SocialEvo" "Flights" "myket" "enron"
declare -a data_list_method1=("wikipedia" "uci")
declare -a methods_list=("JODIE" "DyRep" "TGAT" "TGN" "TCL" "GraphMixer" "RepeatMixer" "DyGFormer" "QSFormer" "FFNFormer")
# declare -a data_dyg=("CanParl" "USLegis" "UNtrade" "UNvote" "Contacts")  # List of data values for DyGFormer
declare -a negative_sample_strategy=("historical" "inductive" "myket")
# Read the gpu and method from command-line arguments
gpu=$1
method=$2

# Select the appropriate data list based on the method
data_list=("${data_list_method1[@]}")

# if [ "$method" == "DyGFormer" ]; then
#     data_list=("${data_dyg[@]}")
# fi

if [ $method == "EdgeBank" ]; then
    # Iterate over the data list
    for data in "${data_list[@]}"; do
        for neg in "${negative_sample_strategy[@]}"; do
            dir_to_check="logs/$method/$data/${neg}_negative_sampling_${method}_seed0"
            if ! find $dir_to_check -type f -name "*.1" -print -quit | grep -q '.'; then
                # Run the command with the provided gpu, method, and current data
                cmd="python evaluate_link_prediction.py --gpu $gpu --num_runs 1 --num_epochs 5 --dataset_name $data --model_name $method --negative_sample_strategy $neg --load_best_configs --val_neg_size 1 --test_neg_size 1"
                echo "$cmd"
                eval "$cmd"
                if [ $? -ne 0 ]; then
                    find logs/$method/$data/${neg}_negative_sampling_${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.1' \;
                    find $dir_to_check -type f -name "*.1" -delete
                else
                    find logs/$method/$data/${neg}_negative_sampling_${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.1' \;
                fi
            fi
        done
    done
else
    # Iterate over the data list
    for data in "${data_list[@]}"; do
        for neg in "${negative_sample_strategy[@]}"; do
            dir_to_check="logs/$method/$data/${neg}_negative_sampling_${method}_seed0"
            if ! find $dir_to_check -type f -name "*.1" -print -quit | grep -q '.'; then
                # Run the command with the provided gpu, method, and current data
                cmd="python evaluate_link_prediction.py --gpu $gpu --num_runs 1 --num_epochs 5 --dataset_name $data --model_name $method --negative_sample_strategy $neg --load_best_configs"
                echo "$cmd"
                eval "$cmd"
                if [ $? -ne 0 ]; then
                    find logs/$method/$data/${neg}_negative_sampling_${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.1' \;
                    find $dir_to_check -type f -name "*.1" -delete
                else
                    find logs/$method/$data/${neg}_negative_sampling_${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.1' \;
                fi
            fi
        done
    done
fi