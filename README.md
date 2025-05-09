# KPEval: SemR-p 🛠️ 
## (Fork for Semantic R-precision Implementation)

**Note:** This repository is a fork of the official [KPEval toolkit](https://github.com/uclanlp/KPEval) by Wu, Yin, and Chang. The primary purpose of this fork is the development and testing of a new keyphrase evaluation metric, Semantic R-Precision (SemR-p).

**Semantic R-precision**

Semantic R-Precision (SemR-p) is a novel metric designed to jointly evaluate semantic relevance and ranking quality of reference agreement in keyphrase prediction. It builds upon the R-Precision framework by incorporating embedding-based semantic similarity for non-exact matches. This work aims to provide a more holistic assessment of keyphrase quality. 
SemR-p is output automatically with the other semantic metrics when using `semantic_matching` as metric_id in the evaluation function.

The implementation of SemR-p in this fork is intended for contribution back to the main KPEval toolkit. If you use SemR-p, please cite our [paper](https://www.researchgate.net/publication/391552955_Meaning_in_Order_Order_in_Meaning_Semantic_R-precision_for_Keyphrase_Evaluation}):

```
@misc{VenturiniKinkel2024SemRp_preprint,
  author    = {Venturini, Shamira and Kinkel, Steffen},
  title     = {Meaning in Order, Order in Meaning: {S}emantic {R}-precision for Keyphrase Evaluation},
  year      = {2025},
  howpublished = {ResearchGate preprint},
  note      = {Preprint available online},
  url       = {https://www.researchgate.net/publication/391552955_Meaning_in_Order_Order_in_Meaning_Semantic_R-precision_for_Keyphrase_Evaluation},
  doi       = {10.13140/RG.2.2.31841.83044}
}
```

### Modifications in this Fork:

1.  **Added SemR-p Metric Implementation:**

SemR-p was integrated into the existing `SemanticMatchingMetric` class alongside SemP, SemR, and SemF1 (within `metrics/semantic_matching_metric.py`): This integration leverages the same core phrase-level semantic similarity calculation (using Sentence Transformer `uclanlp/keyphrase-mpnet-v1`) already implemented for SemP/R/F1. This allows users interested in phrase-level semantic reference agreement to obtain both rank-agnostic scores (SemP/R/F1) and a rank-aware score (SemR-p) simultaneously by running the `semantic_matching` metric group, without the overhead of loading the embedding model twice or running separate evaluation scripts. SemR-p complements SemP/R/F1 by incorporating crucial ranking information (via the R-Precision framework) into the phrase-level semantic evaluation, offering a more complete picture of system-level semantic performance.

2.  **Added macOS/CPU PyTorch Installation Instructions:**

- Install torch. For macOS or CPU-only users (no CUDA):

`pip install torch==1.13.1 torchvision torchaudio`

3.  **Added Data Retrieval Utility Script:**
    *   Included a new script `doc_retriever.py` that provides a function (`get_qualitative_data`) to easily load the source text, target keyphrases, and predicted keyphrases for a specific document index from the model_outputs folder.
    *   This utility facilitates qualitative analysis and inspection of specific examples.


-----------------------------

# KPEval 🛠️

KPEval is a toolkit for evaluating your keyphrase systems. 🎯 

[[Paper] (ACL 2024 Findings)](https://aclanthology.org/2024.findings-acl.117.pdf) [[Tweet]](https://x.com/DiWu0162/status/1762610119840505951)

We provide semantic-based metrics for four evaluation aspects:

- 🤝 **Reference Agreement:** evaluating the extent keyphrase predictions align with human-annotated references.
- 📚 **Faithfulness:** evaluating whether each keyphrase prediction is semantically grounded in the input.
- 🌈 **Diversity:** evaluating whether the predictions include diverse keyphrases with minimal repetitions.
- 🔍 **Utility:**  evalauting the potential of the predictions to enhance document indexing for improved information retrieval performance.

If you have any questions or suggestions, please submit an issue. Thank you!

## News 📰
- [**2024/02**] 🚀 We have released the KPEval toolkit.
- [**2023/05**] 🌟 The phrase embedding model is now available at [uclanlp/keyphrase-mpnet-v1](https://huggingface.co/uclanlp/keyphrase-mpnet-v1).

## Getting Started

We recommend setting up a conda environment:
```
conda create -n kpeval python=3.8
conda activate kpeval
```

Installing the required packages:
- Install torch. Example command if you use CUDA GPUs on linux:

      pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

- `pip install -r requirements.txt`

We provide the outputs obtained from 21 keyphrase models in [this link](https://drive.google.com/file/d/1DExgIfRlDtrunHiLUy6zci4sMkAje9lZ/view?usp=sharing). Please run `tar -xzvf kpeval_model_outputs.tar.gz` to uncompress. Please email `diwu@cs.ucla.edu` or open an issue if the link expires.

## Running Evaluation

The execution of all the evaluation aspects are integrated in the `run_evaluation.py` script. We provide a simple bash script to run the evaluation. You can simply run:

```
bash run_evaluation.sh [dataset] [model_id] [metric_id]
```

For example:

```
bash run_evaluation.sh kp20k 8_catseq semantic_matching
```

Two log files containing the evaluation results and the per-document scores will be saved to `eval_results/[dataset]/[model_id]/`. Please see below for the `metric_id` corresponding to various metrics.

## Supported Metrics 📊

The major metrics supported here are the ones introduced in [the KPEval paper](https://arxiv.org/abs/2303.15422). 

| aspect             | metric           | `metric_id`          | `result_field`                  |
|--------------------|------------------|--------------------|-------------------------------|
| reference agreement| SemF1            | semantic_matching  | semantic_f1                   |
| faithfulness       | UniEval          | unieval            | faithfulness-summ             |
| diversity          | dup_token_ratio  | diversity          | dup_token_ratio               |
| diversity          | emb_sim          | diversity          | self_embed_similarity_sbert   |
| utility            | Recall@5         | retrieval          | sparse/dense_recall_at_5      |
| utility            | RR@5             | retrieval          | sparse/dense_mrr_at_5         |

`metric_id` is the argument to provide to the evaluation script, and `result_field` is the field in the result file where the metric's results are stored.

Note: to evaluate utility, you need to prepare the training data using [DeepKPG](https://github.com/uclanlp/DeepKPG) and update the config to point to the corpus.

 In addition, we support the following metrics from various previous work:

| aspect             | metric              | metric_id             | result_field                      |
|--------------------|---------------------|-----------------------|-----------------------------------|
| reference agreement| F1@5                | exact_matching        | micro/macro_avg_f1@5              |
| reference agreement| F1@M                | exact_matching        | micro/macro_avg_f1@M              |
| reference agreement| F1@O                | exact_matching        | micro/macro_avg_f1@O              |
| reference agreement| MAP                 | exact_matching        | MAP@M                             |
| reference agreement| NDCG                | exact_matching        | avg_NDCG@M                        |
| reference agreement| alpha-NDCG          | exact_matching        | AlphaNDCG@M                       |
| reference agreement| R-Precision         | approximate_matching  | present/absent/all_r-precision    |
| reference agreement| FG                  | fg                    | fg_score                          |
| reference agreement| BertScore           | bertscore             | bert_score_[model]_all_f1         |
| reference agreement| MoverScore          | moverscore            | mover_score_all                   |
| reference agreement| ROUGE               | rouge                 | present/absent/all_rouge-l_f      |
| diversity          | Unique phrase ratio | diversity             | unique_phrase_ratio               |
| diversity          | Unique token ratio  | diversity             | unique_token_ratio                |
| diversity          | SelfBLEU            | diversity             | self_bleu                         |

## Using your own models, datasets, or metrics 🛠️
- **New dataset**: create a config file at `configs/sample_config_[dataset].gin`.
- **New model**: store your model's outputs at `model_outputs/[dataset]/[model_id]/[dataset]_hypotheses_linked.json`. The file should be in `jsonl` format containing three fields: `source`, `target`, `prediction`. If you are conducting reference-free evaluation, you may use a placeholder in the target field.
- **New metric**: just implement it in a new file in the `metrics` folder. The metric class should inherit `KeyphraseMetric`. Make sure you update `metrics/__init__.py` and `run_evaluation.py`. Also make sure you update the config file in `configs` with the parameters for your new metrics.

## Citation
If you find this toolkit useful, please consider citing the following paper.
```
@inproceedings{wu-etal-2024-kpeval,
    title = "{KPE}val: Towards Fine-Grained Semantic-Based Keyphrase Evaluation",
    author = "Wu, Di  and
      Yin, Da  and
      Chang, Kai-Wei",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.117",
    pages = "1959--1981",
}
```
