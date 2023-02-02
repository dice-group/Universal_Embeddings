# Universal_Embeddings
This repository implements universal embeddings for KGs.

## Description

Clone this repository:

```
git clone https://github.com/dice-group/Universal_Embeddings.git

``` 

There are two sub-repositories in this repository: `dice-embeddings` and `embeddings.cc`

### dice-embeddings

The source repository is at [dice-embeddings](https://github.com/dice-group/dice-embeddings.git). It is used to compute our universal knowledge graph embeddings. It implements several knowledge graph embedding models, including ConEx, QMult, ComplEx, DistMult, etc.


### embeddings.cc

This repository implements the API ([embeddings.cc](https://embeddings.cc/)) for providing our universal embeddings. Additional information can be found on the source [github](https://github.com/dice-group/embeddings.cc) repository. At [embeddings.cc](https://embeddings.cc/) and [zenodo](https://zenodo.org/record/7566020#.Y9vKk9LMJH6), we provide over 170 million entity emebddings from the merge of [DBpedia](https://www.dbpedia.org/blog/dbpedia-snapshot-2022-09-release/) and [Wikidata](https://dumps.wikimedia.org/wikidatawiki/). These embeddings are computed using the ConEx embedding model which was chosen based on its performance on our experiments in Table 3. We will add more datasets in our future releases.


## Installation

Make sure Ananconda is install on your machine. 

1. Create a working environment and activate it:

```
conda create -n unikge python=3.9.12

conda activate unikge
 
```

2. Install all dependencies in requirements.txt:

`` pip install -r requirements.txt ``


## Training universal embeddings


### Reproducing the reported results (Table 3)

1. Download evaluation datasets from [hobbit](https://hobbitdata.informatik.uni-leipzig.de/UniKGE/splits.zip) and extract the zip file.

2. Make sure `dice-embeddings` is cloned and switch to the `CLF` branch using `git checkout CLF` then `git checkout .`.

3. To reproduce evaluation results (Table 3) in the paper, enter in the dice-embeddings repository with `cd dice-embeddings` and run the following command:

`` python main.py --path_dataset_folder {path_to_kg_folder} --model {model_name} --batch_size 8192 --embedding_dim 32 --eval train_test --num_epochs 500 ``

Here, {path_to_kg_folder} is the path to DBpedia, DBpedia+, Wikidata, or Wikidata+ which are downloaded above. {model_name} is the embedding model name, i.e., ConEx, DistMult, ComplEx, or QMult.


### Computing universal embeddings on other knowledge graphs

1. Merge your knowledge graphs using Algorithm 1 in the paper (page 7). For large knowledge graphs, we recommend saving the merged KG in parquet format to save memory and enable fast reading and processing with libraries such as [polars](https://pypi.org/project/polars/).

2. Run `` python main.py --path_dataset_folder {path_to_kg_folder} --model {model_name} --batch_size {e.g. 1024} --embedding_dim {e.g. 32} --eval train_test --num_epochs {e.g. 200} ``

3. Call .get_embeddings() on the trained model to obtain entity and relation embeddings





