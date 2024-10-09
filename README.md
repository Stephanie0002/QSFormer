# Quick Sense Temporal Graph Transformer with Efficient Representation Augmentation

This repository is built for the paper [Quick Sense Temporal Graph Transformer with Efficient Representation Augmentation](https://).

🔔 If you have any questions or suggestions, please feel free to let us know.
You can directly email Ziqi Huang using the email address ziqi@zju.edu.cn.

## Benchmark Datasets and Preprocessing

Eight datasets are used in QSFormer, including Wikipedia, Reddit, MOOC, LastFM, Myket, UCI, Flights and Contact.
The first five datasets are bipartite, and the others only contain nodes with a single type.

Most of the used original dynamic graph datasets come from [Towards Better Evaluation for Dynamic Link Prediction](https://openreview.net/forum?id=1GVpwr2Tfdg),
which can be downloaded [here](https://zenodo.org/record/7213796#.Y1cO6y8r30o).
Please download them and put them in ``DG_data`` folder.
The Myket dataset comes from [Effect of Choosing Loss Function when Using T-batching for Representation Learning on Dynamic Networks](https://arxiv.org/abs/2308.06862) and
can be accessed from [here](https://github.com/erfanloghmani/myket-android-application-market-dataset).
The original and preprocessed files for Myket dataset are included in this repository.

We can run ``preprocess_data/preprocess_data.py`` for pre-processing the datasets.
For example, to preprocess the *Wikipedia* dataset, we can run the following commands:

```{bash}
cd preprocess_data/
python preprocess_data.py  --dataset_name wikipedia
```

We can also run the following commands to preprocess all the original datasets at once:

```{bash}
cd preprocess_data/
python preprocess_all_data.py
```

## Dynamic Graph Learning BaseLines

Nine popular continuous-time dynamic graph learning baselines are included, including
[JODIE](https://dl.acm.org/doi/10.1145/3292500.3330895),
[DyRep](https://openreview.net/forum?id=HyePrhR5KX),
[TGAT](https://openreview.net/forum?id=rJeW1yHYwH),
[TGN](https://arxiv.org/abs/2006.10637),
[CAWN](https://openreview.net/forum?id=KYPz4YsCPj),
[EdgeBank](https://openreview.net/forum?id=1GVpwr2Tfdg),
[TCL](https://arxiv.org/abs/2105.07944),
[GraphMixer](https://openreview.net/forum?id=ayPPc0SyLv1), and
[DyGFormer](https://arxiv.org/abs/2303.13047).

## Evaluation Tasks

Dynamic link prediction under both transductive and inductive settings and two (i.e., random, historical-random) negative sampling strategies,
using 1:49 test, 1:9 validation, 1:1 train negative sampling ratio.

## Executing Scripts

### Preparation

Prepare a machine with gcc and python installed, and prepare the conda environment according to requirement.txt or environment.yml.
Compile the c++ sampler library we need according to the following command before running the script.

```bash
cd utils/cpp/
python setup.py build_ext --inplace
```

### Scripts for Dynamic Link Prediction

Dynamic link prediction could be performed on all the thirteen datasets.
If you want to load the best model configurations determined by the grid search, please set the *load_best_configs* argument to True.

#### Model Training

* If you want to use the best model configurations to train *QSFormer* on *Wikipedia* dataset, run

```{bash}
python train_link_prediction.py --dataset_name wikipedia --model_name QSFormer --load_best_configs --gpu 0
```

or

```{bash}
bash train_link.sh 0 QSFormer wikipedia
(format: bash train_link.sh $gpu $model $dataset)
```

#### Model Evaluation

Two (i.e., random, historical-random) negative sampling strategies can be used for model evaluation, using 1:49 test, 1:9 validation, 1:1 train negative sampling ratio.

* If you want to use the best model configurations to evaluate *QSFormer* with *random* negative sampling strategy on *Wikipedia* dataset, run

```{bash}
python evaluate_link_prediction.py --dataset_name wikipedia --model_name QSFormer --negative_sample_strategy random --load_best_configs --gpu 0
```

* If you want to use the best model configurations to evaluate *QSFormer* with *random* or *historical-random* negative sampling strategy on *Wikipedia* dataset, run

```{bash}
bash eval_link.sh 0 QSFormer wikipedia
(format: bash eval_link.sh $gpu $model $dataset)
```

## Citation

Please consider citing our paper when using this project.

```
To be released.
```