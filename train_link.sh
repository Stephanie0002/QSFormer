# !/bin/bash

# Define the data lists for each method
# declare -a data_list_method1=("wikipedia" "uci" "reddit" "Contacts" "askUbuntu" "mooc" "lastfm" "SocialEvo" "Flights" "myket" "enron")
# declare -a data_dyg=("USLegis" "UNtrade" "UNvote" "Contacts")  # List of data values for DyGFormer

# Read the gpu and method from command-line arguments
gpu=$1
method=$2
data=$3

# Select the appropriate data list based on the method
# data_list=("${data_list_method1[@]}")

# if [ "$method" == "DyGFormer" ]; then
#     data_list=("${data_dyg[@]}")
# fi

# Iterate over the data list
# for data in "${data_list[@]}"; do
dir_to_check="logs/$method/$data/${method}_seed0"
if [ ! -d "$dir_to_check" ]; then
    # Run the command with the provided gpu, method, and current data
    cmd="python train_link_prediction.py --gpu $gpu --num_runs 1 --dataset_name $data --model_name $method --load_best_configs --test_interval_epochs 120"
    echo "$cmd"
    eval "$cmd"
fi
# done
