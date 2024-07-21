# !/bin/bash

# Define the data lists for each method
declare -a data_list_method1=("wikipedia" "uci" "reddit" "Contacts" "askUbuntu" "mooc" "lastfm" "SocialEvo" "Flights" "myket" "enron")
# declare -a data_dyg=("CanParl" "USLegis" "UNtrade" "UNvote" "Contacts")  # List of data values for DyGFormer

# Read the gpu and method from command-line arguments
gpu=$1
method=$2

# Select the appropriate data list based on the method
data_list=("${data_list_method1[@]}")

# if [ "$method" == "DyGFormer" ]; then
#     data_list=("${data_dyg[@]}")
# fi

# Iterate over the data list
for data in "${data_list[@]}"; do
    dir_to_check="logs/$method/$data/${method}_seed0"
    if ! find $dir_to_check -type f -name "*.log.efficiency" -print -quit | grep -q '.'; then
        # Run the command with the provided gpu, method, and current data
        cmd="CUDA_VISIBLE_DEVICES=$gpu python train_link_prediction.py --num_runs 1 --dataset_name $data --model_name $method --load_best_configs  --num_epochs 10  --val_neg_size 1 --test_neg_size 1 --test_interval_epochs 20"
        echo "$cmd"
        eval "$cmd"
        if [ $? -ne 0 ]; then
            find logs/$method/$data/${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.efficiency' \;
            find $dir_to_check -type f -name "*.log.efficiency" -delete
        else
            find logs/$method/$data/${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.efficiency' \;
        fi
    fi
done
