#Basic QSFormer
# python ../FastDygForm/train_link_prediction.py --dataset_name uci --gpu 0 --order gradient-0.08-3 --model_name QSFormer --patch_size 4 --max_input_sequence_length 128 --num_runs 1 --num_hops 2 --dropout 0.15 --num_hops 2 --num_high_order_neighbors 3  > ../FastDygForm/gg2-gpu0-log.txt 2>&1 &

gpu=$1

declare -a methods_list=("QSFormer")
declare -a data_list=("uci" "mooc" "myket")


for data in "${data_list[@]}"; do
    for method in "${methods_list[@]}"; do
        # 去除自适应小批次生成
        cmd="python ../FastDygForm/train_link_prediction.py --dataset_name $data --gpu $gpu --order chorno --model_name $method --num_runs 1 --num_hops 2 --num_layers 2 --num_high_order_neighbors 3 --load_best_configs --ablation --test_interval_epochs 120"
        echo "去除自适应小批次生成"
        echo "$cmd"
        eval "$cmd"
        find logs/$method/$data/${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.adapt' \;

        # 去除二跳邻居
        cmd="python ../FastDygForm/train_link_prediction.py --dataset_name $data --gpu $gpu --order gradient-0.08-3 --model_name $method --num_runs 1 --num_hops 1 --num_layers 2 --load_best_configs --ablation --test_interval_epochs 120"
        echo "去除二跳邻居"
        echo "$cmd"
        eval "$cmd"
        find logs/$method/$data/${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.onehop' \;

        # 去除身份编码
        cmd="python ../FastDygForm/train_link_prediction.py --dataset_name $data --gpu $gpu --order gradient-0.08-3 --model_name $method --num_runs 1 --num_hops 2 --num_layers 2 --num_high_order_neighbors 3 --no_id_encode --load_best_configs --ablation --test_interval_epochs 120"
        echo "去除身份编码"
        echo "$cmd"
        eval "$cmd"
        find logs/$method/$data/${method}_seed0 -type f -regex '.*/[0-9]+\.[0-9]+\.log' -exec bash -c 'mv {} {}.norole' \;
    done
done