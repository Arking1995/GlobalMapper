# GlobalMapper
Official Pytorch Implementation of "GlobalMapper: Arbitrary-Shaped Urban Layout Generation"

[arXiv](https://arxiv.org/abs/2307.09693) | [BibTeX](#bibtex) | [Project Page](https://arking1995.github.io/GlobalMapper/)

This repo contains codes for single GPU training for 
[GlobalMapper: Arbitrary-Shaped Urban Layout Generation](https://arxiv.org/pdf/2307.09693.pdf)

**Note that this repo is lack of code comments.**


## Environment
We provide required environments in "environment.yml". But practially we suggest to use below commands for crucial dependencies:
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric
```
Then you may install other dependencies like: matplotlib, yaml, pickle, etc.

## Dataset
Unzip the dataset.tar.gz. "processed" folder contains 10K preprocessed graph-represented city blocks. You may read them by "networkx.read_gpickle()". "raw_geo" contains 10K corresponding original building and block polygons (shapely.polygon format) of each city block (coordinates in UTM Zone projection). You may read it by "pickle.load()". Those original building polygons are directly acquired by "osmnx.geometries module" from [osmnx](https://osmnx.readthedocs.io/en/stable/user-reference.html).

Our canonical spatial transformation converts the original building polygons to the canonical version. After simple normalization by mean substraction and std dividing, coordinates and location information are encoded as node attributes in 2D grid graphs, then saved in "processed". Since the raw dataset is public accessible, we encourage users to implement their own preprocessing of original building polygons. It may facilitate better performance.


## How to train your model
After set up your training parameters in "train_gnn.yaml". Simply run
```
python train.py
```


## How to test your model
After you setup desired "dataset_path" and "epoch_name". Simply run
```
python test.py
```


## BibTeX

If you use this code, please cite
```text
@inproceedings{he2023globalmapper,
  title={GlobalMapper: Arbitrary-Shaped Urban Layout Generation},
  author={He, Liu and Aliaga, Daniel},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

