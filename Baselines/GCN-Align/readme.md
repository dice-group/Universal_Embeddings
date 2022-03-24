## Baseline Instructions 

---
### GCN-Align: 

#### Overview: 
  * The GCN-Align approach is designed to align related entities from multilingual graphs using graph convolutional network (GCN). 
  * Entity alignments are discovered based on the distances between entties in the embedding space. 
  * The authors proposed embedding from structural information via GCN, and attribute information of entities, the final alignments are computed based on the combination of both embeddings. 
#### Installation
* The required libraries for installing GCN-Align can be found in : `GCN-Align/requirements.txt` --> `pip install -r requirements.txt` or `python setup.py install` 

#### Dataset: 
* We carried out our experiments on the OpenEA benchmark dataset (v1.1) `DBpedia15k_FR_EN, and DBpedia100k_FR_EN` 
  * The datasets can be download from [Dropbox link](https://www.dropbox.com/s/nzjxbam47f9yk3d/OpenEA_dataset_v1.1.zip?dl=0).
  * Please create a folder named `datasets`, and add the corresponding datasets into i.e., `EN_FR_15k_V1 , EN_FR_15k_V2 , EN_FR_100k_V1 , EN_FR_100k_V2`

#### How to run: 
* You can simply run the GCN-Align experiments using the script files in `GCN-Align/run/run_15k.sh` and `GCN-Align/run/run_100k.sh` 
* If you want to change the hyper-parameters, you can find the configuration files in `GCN-Align/run/args` 

#### Results: 
  
  The results will be saved in `Baseline/results`. The output files are the evaluation of alignents, and the learned embeddings.
  We used a greed strategy to align entities, as follows :  
  `greedy_alignment(embed1, embed2, top_k, nums_threads, metric, normalize, csls_k, accurate)`  

* Parameters:
    * `embed1` : matrix_like
    An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    * `embed2` : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    * `top_k` : list of integers
        Hits@k metrics for evaluating results.
    * `nums_threads` : int
        The number of threads used to search alignment.
    * `metric` : string
        The distance metric to use. It can be 'cosine', 'euclidean' or 'inner'.
    *  `normalize` : bool, true or false.
        Whether to normalize the input embeddings.
    *  `csls_k` : int
        K value for csls. If k > 0, enhance the similarity by csls.

* Returns
    * `alignment_rest` :  list, pairs of aligned entities
    * `hits1` : float, hits@1 values for alignment results
    * `mr` : float, MR values for alignment results
    * `mrr` : float, MRR values for alignment results

* More details about the evaluation metrics can be found in `src/openea/modules/finding/alignment.py` and `src/openea/modules/finding/evaluation.py`
  
#### Contact
If you have any question or suggestion related to the GCN-Align experiment, please contact `hamada.zahera@upb.de` 
  