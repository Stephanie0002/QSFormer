# !/bin/bash

# Define the data lists for each method
declare -a data_list_method1=("wikipedia" "reddit" "mooc" "lastfm" "myket" "enron" "SocialEvo" "uci" "Flights" "CanParl")
# declare -a data_dyg=("CanParl" "USLegis" "UNtrade" "UNvote" "Contacts")  # List of data values for DyGFormer
declare -a negative_sample_strategy=("random" "historical" "inductive")
# Read the gpu and method from command-line arguments
gpu=$1
method=$2

# Select the appropriate data list based on the method
data_list=("${data_list_method1[@]}")

# if [ "$method" == "DyGFormer" ]; then
#     data_list=("${data_dyg[@]}")
# fi

if [ $method == "EdgeBank" ]; then
    dir_to_check="logs/$method/$data/${negative_sample_strategy}_negative_sampling_${method}_seed0"
    if [ ! -d "$dir_to_check" ]; then
        # Iterate over the data list
        for data in "${data_list[@]}"; do
            for neg in "${negative_sample_strategy[@]}"; do
                # Run the command with the provided gpu, method, and current data
                cmd="CUDA_VISIBLE_DEVICES=$gpu python evaluate_link_prediction.py --num_runs 1 --num_epochs 5 --dataset_name $data --model_name $method --negative_sample_strategy $neg"
                echo "$cmd"
                eval "$cmd"
            done
        done
    fi
else
    dir_to_check="logs/$method/$data/${negative_sample_strategy}_negative_sampling_${method}_seed0"
    if [ ! -d "$dir_to_check" ]; then
        # Iterate over the data list
        for data in "${data_list[@]}"; do
            for neg in "${negative_sample_strategy[@]}"; do
                # Run the command with the provided gpu, method, and current data
                cmd="CUDA_VISIBLE_DEVICES=$gpu python evaluate_link_prediction.py --num_runs 1 --num_epochs 5 --dataset_name $data --model_name $method --negative_sample_strategy $neg"
                echo "$cmd"
                eval "$cmd"
            done
        done
    fi
fi