# Universal_Embeddings
This repository implements universal embeddings for KGs.

## Installation

Clone this repository:
```
git clone https://github.com/dice-group/Universal_Embeddings.git

``` 
Also clone dice-embeddings into Universal_Embeddings:

```
cd Universal_Embeddings

git clone https://github.com/dice-group/dice-embeddings.git

```

Create a working environment and activate it:

```
conda create -n unikge python=3.9.12

conda activate unikge
 
```

Install all dependencies in requirements.txt

## Evaluation datasets

Download evaluation datasets from [hobbit](https://hobbitdata.informatik.uni-leipzig.de/KGE/splits.zip) and extract the zip file 

## Reproducing the reported results
Enter in dice-embeddings: `cd dice-embeddings`

Run the following to start training embeddings:

`` python main.py --path_dataset_folder path_to_kg_folder --model model_name --batch_size 8192 --embedding_dim 32 --eval train_test --num_epochs 500 ``
