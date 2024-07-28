#Basic QSFormer
# python train_link_prediction.py --dataset_name uci --gpu 0 --order gradient-0.08-3 --model_name QSFormer --patch_size 4 --max_input_sequence_length 128 --num_runs 1 --num_hops 2 --dropout 0.15 --num_hops 2 --num_high_order_neighbors 3  > gg2-gpu0-log.txt 2>&1 &

gpu=$1
data=$2

declare -a methods_list=("QSFormer")
# declare -a data_list=("uci" "mooc" "myket")


# for data in "${data_list[@]}"; do
for method in "${methods_list[@]}"; do
    # 去除自适应小批次生成
    cmd="python train_link_prediction.py --dataset_name $data --gpu $gpu --order chorno --model_name $method --num_runs 1 --num_hops 2 --num_layers 2 --load_best_configs --ablation --test_interval_epochs 120"
    echo "去除自适应小批次生成"
    echo "$cmd"
    eval "$cmd"
    cmd2="find logs/$method/$data/${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.adapt' \;"
    echo "$cmd2"
    eval "$cmd2"

    # 去除二跳邻居
    cmd="python train_link_prediction.py --dataset_name $data --gpu $gpu --order gradient-0.08-3 --model_name $method --num_runs 1 --num_layers 2 --load_best_configs --ablation --no_high_order--test_interval_epochs 120"
    echo "去除二跳邻居"
    echo "$cmd"
    eval "$cmd"
    cmd2="find logs/$method/$data/${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.onehop' \;"
    echo "$cmd2"
    eval "$cmd2"

    # 去除长序列
    cmd="python train_link_prediction.py --dataset_name $data --gpu $gpu --order gradient-0.08-3 --model_name $method --num_runs 1 --num_layers 2 --load_best_configs --ablation --no_long_sequence --test_interval_epochs 120"
    echo "去除长序列"
    echo "$cmd"
    eval "$cmd"
    cmd2="find logs/$method/$data/${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.onehop' \;"
    echo "$cmd2"
    eval "$cmd2"

    # 去除身份编码
    cmd="python train_link_prediction.py --dataset_name $data --gpu $gpu --order gradient-0.08-3 --model_name $method --num_runs 1 --num_hops 2 --num_layers 2 --no_id_encode --load_best_configs --ablation --test_interval_epochs 120"
    echo "去除身份编码"
    echo "$cmd"
    eval "$cmd"
    cmd2="find logs/$method/$data/${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.norole' \;"
    echo "$cmd2"
    eval "$cmd2"
done
# done