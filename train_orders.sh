# chorno
python ../FastDygForm/train_link_prediction.py --dataset_name CanParl --gpu 0 --order chorno --model_name DyGFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gg2-gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name UNtrade --gpu 1 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --num_hops 1 --dropout 0.0  > ../FastDygForm/gg2-gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name mooc --gpu 2 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gg2-gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name enron --gpu 3 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --num_hops 1 --dropout 0.0  > ../FastDygForm/gg2-gpu3-log.txt 2>&1 &

python ../FastDygForm/train_link_prediction.py --dataset_name wikipedia --gpu 0 --order chorno --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --num_hops 1 --dropout 0.1 > ../FastDygForm/gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name reddit --gpu 1 --order chorno --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --num_runs 1 --num_hops 1 --dropout 0.1 > ../FastDygForm/gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name uci --gpu 2 --order chorno --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --num_hops 1 --dropout 0.1 > ../FastDygForm/gpu2-log.txt 2>&1 &
# python ../FastDygForm/train_link_prediction.py --dataset_name USLegis --gpu 3 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --num_hops 1 --dropout 0.0 > ../FastDygForm/gpu3-log.txt 2>&1 &

python ../FastDygForm/train_link_prediction.py --dataset_name SocialEvo --gpu 0 --order chorno --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name Flights --gpu 1 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name lastfm --gpu 2 --order chorno --model_name DyGFormer --patch_size 16 --max_input_sequence_length 512 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name myket --gpu 3 --order chorno --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --num_runs 1 --num_hops 1 --dropout 0.1 > ../FastDygForm/gpu3-log.txt 2>&1 &

# FFNFormer  gradient-0.08-3 + identical encode + two-hop neighbor + mixer
python ../FastDygForm/train_link_prediction.py --dataset_name CanParl --gpu 0 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gg2-gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name UNtrade --gpu 1 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.0  > ../FastDygForm/gg2-gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name mooc --gpu 2 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gg2-gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name enron --gpu 3 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.0  > ../FastDygForm/gg2-gpu3-log.txt 2>&1 &

python ../FastDygForm/train_link_prediction.py --dataset_name wikipedia --gpu 0 --order gradient-0.08-3 --model_name FFNFormer --patch_size 4 --max_input_sequence_length 128 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name reddit --gpu 1 --order gradient-0.08-3 --model_name FFNFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name uci --gpu 2 --order gradient-0.08-3 --model_name FFNFormer --patch_size 4 --max_input_sequence_length 128 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gpu2-log.txt 2>&1 &
# python ../FastDygForm/train_link_prediction.py --dataset_name USLegis --gpu 3 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.0 > ../FastDygForm/gpu3-log.txt 2>&1 &

python ../FastDygForm/train_link_prediction.py --dataset_name SocialEvo --gpu 0 --order gradient-0.08-3 --model_name FFNFormer --patch_size 4 --max_input_sequence_length 128 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name Flights --gpu 1 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name lastfm --gpu 2 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name myket --gpu 3 --order gradient-0.08-3 --model_name FFNFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --num_hops 2 --dropout 0.1 > ../FastDygForm/gpu3-log.txt 2>&1 &

# gradient-0.08-3 + identical encode + one-hop neighbor + mixer
python ../FastDygForm/train_link_prediction.py --dataset_name CanParl --gpu 0 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gg2-gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name UNtrade --gpu 1 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 1 --dropout 0.0  > ../FastDygForm/gg2-gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name mooc --gpu 2 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gg2-gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name enron --gpu 3 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 1 --dropout 0.0  > ../FastDygForm/gg2-gpu3-log.txt 2>&1 &

python ../FastDygForm/train_link_prediction.py --dataset_name wikipedia --gpu 3 --order gradient-0.08-3 --model_name FFNFormer --patch_size 4 --max_input_sequence_length 128 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gpu3-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name reddit --gpu 1 --order gradient-0.08-3 --model_name FFNFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name uci --gpu 2 --order gradient-0.08-3 --model_name FFNFormer --patch_size 4 --max_input_sequence_length 128 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gpu2-log.txt 2>&1 &
# python ../FastDygForm/train_link_prediction.py --dataset_name USLegis --gpu 3 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 1 --dropout 0.0 > ../FastDygForm/gpu3-log.txt 2>&1 &

python ../FastDygForm/train_link_prediction.py --dataset_name SocialEvo --gpu 0 --order gradient-0.08-3 --model_name FFNFormer --patch_size 4 --max_input_sequence_length 128 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name Flights --gpu 1 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name lastfm --gpu 2 --order gradient-0.08-3 --model_name FFNFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 1 --dropout 0.1  > ../FastDygForm/gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name myket --gpu 3 --order gradient-0.08-3 --model_name FFNFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --num_hops 1 --dropout 0.1 > ../FastDygForm/gpu3-log.txt 2>&1 &

# QSFormer gradient-0.08-3 + identical encode + tow-hop neighbor
python ../FastDygForm/train_link_prediction.py --dataset_name CanParl --gpu 0 --order gradient-0.08-3 --model_name QSFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gg2-gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name UNtrade --gpu 1 --order gradient-0.08-3 --model_name QSFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.0  > ../FastDygForm/gg2-gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name mooc --gpu 2 --order gradient-0.08-3 --model_name QSFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gg2-gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name enron --gpu 3 --order gradient-0.08-3 --model_name QSFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.0  > ../FastDygForm/gg2-gpu3-log.txt 2>&1 &

python ../FastDygForm/train_link_prediction.py --dataset_name wikipedia --gpu 0 --order gradient-0.08-3 --model_name QSFormer --patch_size 4 --max_input_sequence_length 128 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name reddit --gpu 1 --order gradient-0.08-3 --model_name QSFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name uci --gpu 2 --order gradient-0.08-3 --model_name QSFormer --patch_size 4 --max_input_sequence_length 128 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gpu2-log.txt 2>&1 &
# python ../FastDygForm/train_link_prediction.py --dataset_name USLegis --gpu 3 --order gradient-0.08-3 --model_name QSFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.0 > ../FastDygForm/gpu3-log.txt 2>&1 &

python ../FastDygForm/train_link_prediction.py --dataset_name SocialEvo --gpu 0 --order gradient-0.08-3 --model_name QSFormer --patch_size 4 --max_input_sequence_length 128 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name Flights --gpu 1 --order gradient-0.08-3 --model_name QSFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name lastfm --gpu 2 --order gradient-0.08-3 --model_name QSFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --num_hops 2 --dropout 0.1  > ../FastDygForm/gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name myket --gpu 3 --order gradient-0.08-3 --model_name QSFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --num_hops 2 --dropout 0.1 > ../FastDygForm/gpu3-log.txt 2>&1 &



# gradient-0.08-3
python ../FastDygForm/train_link_prediction.py --dataset_name CanParl --gpu 0 --order gradient-0.08-3 --model_name DyGFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --dropout 0.1  > ../FastDygForm/gg2-gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name UNtrade --gpu 1 --order gradient-0.08-3 --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.0  > ../FastDygForm/gg2-gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name mooc --gpu 2 --order gradient-0.08-3 --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.1  > ../FastDygForm/gg2-gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name enron --gpu 3 --order gradient-0.08-3 --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.0  > ../FastDygForm/gg2-gpu3-log.txt 2>&1 &

python ../FastDygForm/train_link_prediction.py --dataset_name wikipedia --gpu 0 --order gradient-0.08-3 --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --dropout 0.1  > ../FastDygForm/gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name reddit --gpu 1 --order gradient-0.08-3 --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --num_runs 1 --dropout 0.1  > ../FastDygForm/gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name uci --gpu 2 --order gradient-0.08-3 --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --dropout 0.1  > ../FastDygForm/gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name USLegis --gpu 3 --order gradient-0.08-3 --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.0 > ../FastDygForm/gpu3-log.txt 2>&1 &

# gradient-0.08-3 + identical encode 
python ../FastDygForm/train_link_prediction.py --dataset_name CanParl --gpu 0 --order gradient-0.08-3 --model_name EnFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --dropout 0.1  > ../FastDygForm/gg2-gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name UNtrade --gpu 1 --order gradient-0.08-3 --model_name EnFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.0  > ../FastDygForm/gg2-gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name mooc --gpu 2 --order gradient-0.08-3 --model_name EnFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.1  > ../FastDygForm/gg2-gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name enron --gpu 3 --order gradient-0.08-3 --model_name EnFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.0  > ../FastDygForm/gg2-gpu3-log.txt 2>&1 &

python ../FastDygForm/train_link_prediction.py --dataset_name wikipedia --gpu 0 --order gradient-0.08-3 --model_name EnFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --dropout 0.1  > ../FastDygForm/gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name reddit --gpu 1 --order gradient-0.08-3 --model_name EnFormer --patch_size 2 --max_input_sequence_length 64 --num_runs 1 --dropout 0.1  > ../FastDygForm/gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name uci --gpu 2 --order gradient-0.08-3 --model_name EnFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --dropout 0.1  > ../FastDygForm/gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name USLegis --gpu 3 --order gradient-0.08-3 --model_name EnFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.0 > ../FastDygForm/gpu3-log.txt 2>&1 &


# time interval
python ../FastDygForm/train_link_prediction.py --dataset_name CanParl --gpu 0 --order chorno --model_name DyGFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --dropout 0.1 --sample_neighbor_strategy time_interval_aware  > ../FastDygForm/gg2-gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name UNtrade --gpu 1 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.0 --sample_neighbor_strategy time_interval_aware  > ../FastDygForm/gg2-gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name mooc --gpu 2 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.1 --sample_neighbor_strategy time_interval_aware  > ../FastDygForm/gg2-gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name enron --gpu 3 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.0 --sample_neighbor_strategy time_interval_aware  > ../FastDygForm/gg2-gpu3-log.txt 2>&1 &

python ../FastDygForm/train_link_prediction.py --dataset_name wikipedia --gpu 0 --order chorno --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --dropout 0.1 --sample_neighbor_strategy time_interval_aware  > ../FastDygForm/gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name reddit --gpu 1 --order chorno --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --num_runs 1 --dropout 0.1  --sample_neighbor_strategy time_interval_aware > ../FastDygForm/gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name uci --gpu 2 --order chorno --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --dropout 0.1 --sample_neighbor_strategy time_interval_aware  > ../FastDygForm/gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name USLegis --gpu 3 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.0 --sample_neighbor_strategy time_interval_aware > ../FastDygForm/gpu3-log.txt 2>&1 &

# uniform
python ../FastDygForm/train_link_prediction.py --dataset_name CanParl --gpu 0 --order chorno --model_name DyGFormer --patch_size 32 --max_input_sequence_length 1024 --num_runs 1 --dropout 0.1 --sample_neighbor_strategy uniform  > ../FastDygForm/gg2-gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name UNtrade --gpu 1 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.0 --sample_neighbor_strategy uniform  > ../FastDygForm/gg2-gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name mooc --gpu 2 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.1 --sample_neighbor_strategy uniform  > ../FastDygForm/gg2-gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name enron --gpu 3 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.0 --sample_neighbor_strategy uniform  > ../FastDygForm/gg2-gpu3-log.txt 2>&1 &

python ../FastDygForm/train_link_prediction.py --dataset_name wikipedia --gpu 0 --order chorno --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --dropout 0.1 --sample_neighbor_strategy uniform  > ../FastDygForm/gpu0-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name reddit --gpu 1 --order chorno --model_name DyGFormer --patch_size 2 --max_input_sequence_length 64 --num_runs 1 --dropout 0.1  --sample_neighbor_strategy uniform > ../FastDygForm/gpu1-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name uci --gpu 2 --order chorno --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --dropout 0.1 --sample_neighbor_strategy uniform  > ../FastDygForm/gpu2-log.txt 2>&1 &
python ../FastDygForm/train_link_prediction.py --dataset_name USLegis --gpu 3 --order chorno --model_name DyGFormer --patch_size 8 --max_input_sequence_length 256 --num_runs 1 --dropout 0.0 --sample_neighbor_strategy uniform > ../FastDygForm/gpu3-log.txt 2>&1 &