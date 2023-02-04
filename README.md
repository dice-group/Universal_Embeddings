# Universal Embeddings for Knowledge Graphs

This repository contains code related to the scientific article *Universal Knowledge Graph Embeddings*.  
The article is currently under review.

## Code structure

The code of the Universal Embeddings approach comprises this main repository *Universal_Embeddings* and two sub-repositories: *dice-embeddings* and *embeddings.cc*. They are briefly introduced in the following and documented in the repositories themselves.

* **Main repository: Universal_Embeddings**  
This repository contains two notebooks: merge.ipynb and Visualize_Embeddings.ipynb. The first notebook implements the merge (fusion) of DBpedia and Wikidata via sameAs links. The second notebook visualizes the embeddings of the same entities across different knowledge graphs.
The computation of universal embeddings is based on the [dice-embeddings](https://github.com/dice-group/dice-embeddings) framework. We also develop the [embeddings.cc repository](https://github.com/dice-group/embeddings.cc) to provide universal embeddings as a service. 
* **Sub-repository: dice-embeddings**  
We use the [dice-embeddings repository](https://github.com/dice-group/dice-embeddings) to compute universal knowledge graph embeddings. It implements several knowledge graph embedding models, including ConEx, QMult, ComplEx and DistMult.
* **Sub-repository: embeddings.cc**  
The [embeddings.cc repository](https://github.com/dice-group/embeddings.cc) was implemented on top of our computed Universal Embeddings for the artivle *Universal Knowledge Graph Embeddings*. It implements the online API for providing the embeddings data. The API is deployed at [https://embeddings.cc](https://embeddings.cc/). 

## Resulting data

On [https://embeddings.cc](https://embeddings.cc/) and also in a dump file on Zenodo ([DOI: 10.5281/zenodo.7566020](https://doi.org/10.5281/zenodo.7566020)) we provide roughly 180 million entity emebddings from the merge of DBpedia ([DBpedia Snapshot 2022-09](https://www.dbpedia.org/blog/dbpedia-snapshot-2022-09-release/)) and Wikidata ([Wikidata dumps](https://dumps.wikimedia.org/wikidatawiki/)). These embeddings are computed using the ConEx embedding model which was chosen based on its performance on our experiments in article table 3.

## Installation

Start with cloning this repository:

```bash
git clone https://github.com/dice-group/Universal_Embeddings.git
```

Make sure [Ananconda](https://www.anaconda.com/) is install on your machine.  
Create a working environment and activate it:
```bash
conda create -n unikge python=3.9.12
conda activate unikge
```

Install all dependencies in requirements.txt:

```bash
pip install -r requirements.txt
```


## Training Universal Embeddings

### Reproducing the reported paper results (Table 3 in the article)

1. Download the [evaluation datasets zip file](https://hobbitdata.informatik.uni-leipzig.de/UniKGE/splits.zip)  and extract it.

2. Make sure `dice-embeddings` is cloned and switch to the `CLF` branch using `git checkout CLF` then `git checkout .`

3. To reproduce evaluation results (Table 3) in the paper, enter in the dice-embeddings repository with `cd dice-embeddings` and run the following command:
`` python main.py --path_dataset_folder {path_to_kg_folder} --model {model_name} --batch_size 8192 --embedding_dim 32 --eval train_test --num_epochs 500 ``  
Inside the command, ``{path_to_kg_folder}`` is the path to DBpedia, DBpedia+, Wikidata, or Wikidata+ which are downloaded above. ``{model_name}`` is the embedding model name, i.e., ConEx, DistMult, ComplEx, or QMult.


### Computing universal embeddings on other knowledge graphs

1. Merge your knowledge graphs using Algorithm 1 in the paper (page 7). For large knowledge graphs, we recommend saving the merged KG in parquet format to save memory and enable fast reading and processing with libraries such as [polars](https://pypi.org/project/polars/).

2. Run `` python main.py --path_dataset_folder {path_to_kg_folder} --model {model_name} --batch_size {e.g. 1024} --embedding_dim {e.g. 32} --eval train_test --num_epochs {e.g. 200} ``

3. Call .get_embeddings() on the trained model to obtain entity and relation embeddings.





