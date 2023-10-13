# Quantifying &amp; Modeling Multimodal Interactions

This repository contains the code for the following papers on quantifying and modeling multimodal interactions

[**Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework**](https://arxiv.org/abs/2302.12247)<br>
Paul Pu Liang, Yun Cheng, Xiang Fan, Chun Kai Ling, Suzanne Nie, Richard Chen, Zihao Deng, Nicholas Allen, Randy Auerbach, Faisal Mahmood, Ruslan Salakhutdinov, Louis-Philippe Morency<br>
NeurIPS 2023.

[**Multimodal Fusion Interactions: A Study of Human and Automatic Quantification**](https://arxiv.org/abs/2306.04125)<br> 
Paul Pu Liang, Yun Cheng, Ruslan Salakhutdinov, Louis-Philippe Morency<br>
ICMI 2023.

If you find this repository useful, please cite the relevant works:
```
@inproceedings{liang2023quantifying,
  title={Quantifying \& Modeling Multimodal Interactions: An Information Decomposition Framework},
  author={Liang, Paul Pu and Cheng, Yun and Fan, Xiang and Ling, Chun Kai and Nie, Suzanne and Chen, Richard and Deng, Zihao and Mahmood, Faisal and Salakhutdinov, Ruslan and Morency, Louis-Philippe},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
```
@inproceedings{liang2023multimodal,
  author = {Liang, Paul Pu and Cheng, Yun and Salakhutdinov, Ruslan and Morency, Louis-Philippe},
  title = {Multimodal Fusion Interactions: A Study of Human and Automatic Quantification},
  year = {2023},
  isbn = {9798400700552},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3577190.3614151},
  doi = {10.1145/3577190.3614151},
  booktitle = {Proceedings of the 25th International Conference on Multimodal Interaction},
  pages = {425–435},
  numpages = {11},
  keywords = {Multimodal fusion, Affective computing, Multimodal interactions},
  location = {Paris, France},
  series = {ICMI '23}
}
```

## Usage
### Environment Setup Using Conda
To install the repository, first clone the repository via Git, and then install the prerequisite packages through Conda (Linux/MacOS):
```
conda env create [-n ENVNAME] -f environment.yml
```
### Experiments
#### Synthetic Data
The code for synthetic data experiments in the paper *Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework* is located under `synthetic/`.

To generate synthetic data, run
```
python synthetic/generate_data.py --num-data 20000 --setting redundancy --out-path synthetic/experiments
```

To generate the suite of synthetic data, which includes 4 specialized datasets, 6 mixed datasets, and 5 random datasets for evaluating model selection, run
```
bash synthetic/experiments/generate_data.sh
```

Example command to train a multimodal model on synthetic data
```
python synthetic/experiments/base.py --data-path synthetic/experiments/DATA_redundancy.pickle --keys 0 1 label --bs 256 --input-dim 200 --hidden-dim 512 --n-latent 600 --num-classes 2 --saved-model synthetic/experiments/redundancy/redundancy_base_best.pt --modalities 1 1
```

To train the suite of multimodal models on all synthetic data, run
```
bash synthetic/experiments/experiments.sh
```

To train an unimodal version of a multimodal model (which are used in analyzing connections between PID and model robustness in the presence of missing modalities), run
```
bash synthetic/experiments/unimodal_experiments.sh
```

#### Real-World Data
We rely on the original [MultiBench](https://github.com/pliang279/MultiBench) implementations for real-world data experiments. Please refer to the repo for more details on training and evaluation.

### PID Estimator
To run the CVXPY PID estimator on a multimodal dataset, we first need to obtain a discrete clustered version of the original dataset using existing clustering algorithms such as K-Means. After that, call the `convert_data_to_distribution` function in `synthetic/rus.py` on the clustered version of the dataset and then `get_measure` function to get the PID estimates. More details of data clustering and example usage of the PID estimator can be found in the `synthetic/experiments/clustering.ipynb` notebook.

### Analysis and Model Selection
Examples of computing the PID agreement metric and analyzing the correlation between agreement and model performance can be found in the `synthetic/experiments/analysis.ipynb` notebook. Examples of applying PID in multimodal model selection can be found in the `synthetic/model_selection/analysis.ipynb` notebook.

### Estimating Synergy
Code for computing the lower and upper bound for synergy can be found in `bounds/bounds.py`. Examples of estimating synergy for synthetic bitwise datasets (including data generation) can be found in the `bounds/bounds_synthetic.ipynb` notebook and examples of estimating synergy for real-world multimodal datasets can be found in the `bounds/bounds_real.ipynb` notebook. Details on pre-processing real-world datasets can be found in the `bounds/clustering.ipynb` notebook.
