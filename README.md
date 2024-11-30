# 1. Setup and Preparation
## get Liscense
Our model is based on LLaMA 3.1 8B Instruct, for using llama 3.1 edu model you have to get liscense and access token
(`huggingface-cli login` needs access token)

https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

## Setup
```bash
git clone --recurse-submodules https://github.com/MLAI-Yonsei/bookIPs-Solvook-LLM.git
cd bookIPs-Solvook-LLM
conda create -n <name> python==3.11
conda activate <name>
pip install -e ".[torch,metrics]"
pip install --upgrade huggingface_hub
huggingface-cli login
pip install -r requirements_rag.txt
```
> [!TIP]
Use `pip install --no-deps -e .` to resolve package conflicts.

----



<br>


# 2. Fine-tuning Edu-Llama

## Train
* If you want to edit argument for training, edit file **examples/train_lora/llama3_lora_sft.yaml**

```bash
bash finetune.sh
```


> [!TIP]
> Attach `CUDA_VISIBLE_DEVICES=1,2,3` in front of code above to utilize multi GPU

> [!TIP]
> for more information to extend usage of LLaMA Factory package, visit https://github.com/hiyouga/LLaMA-Factory

----

<br>

# 3. Run RAG

## Pre-process dataset for RAG
Run `preprocess_rag.py`
```
preprocess_rag.py --data_path=<DATA_PATH>
```
* ```DATA_PATH``` &mdash; The path where 'Solvook_handout_DB_english.xlsx' exists.



## Set Vector DB
* To make and save vector DB, run `vector_db.py`:
    ```
    vector_db.py --query_path=<QUERY_PATH> --db_path=<DB_PATH> --openai_api_key=<API_KEY> \
                 --task=<TASK> --top_k=<TOP_K> --search_type=<SEARCH_TYPE>
    ```
    * ```QUERY_PATH``` &mdash; The path of query.
    * ```DB_PATH``` &mdash; The path of db.
    * ```API_KEY``` &mdash; API key for openai
    * ```TASK``` &mdash; Designate task (Options: [**1**: Paragraph matching, **2**: Relation matching, **3**: Skill matching, **4**: Method matching])
    * ```TOP_K``` &mdash; The number of retrieved contents (Default: 6)
    * ```SEARCH_TYPE``` &mdash; The methods for calculation of similarity. (Options : [**sim**: Cosine similarity, **mmr**: Maximum marginal relevance search, **bm25**: BM25, **td_idf**: TF-IDF, **sim_bm25**: Ensemble of Cosine similarity and BM25]) (Default: mmr)



## Inference
### Inference with GPT-4o
* Assure to make vector DB, first. 
* RAG-based inference with GPT-4o can simply be done with `rag_gpt4o.py`:
    ```
    rag_gpt4o.py --query_path=<QUERY_PATH> --vector_db_path=<VECTOR_DB_PATH> \
                 --openai_api_key=<API_KEY> \
                 --temperature=<TEMPERATURE> \
                 --task=<TASK> [--in_context_sample] \
                 --result_path=<RESULT_PATH> \
                 [--ignore_wandb] \
                 --wandb_project=<WANDB_PROJECT> \
                 --wandb_entity=<WANDB_ENTITY>
    ```
    * ```QUERY_PATH``` &mdash; The path of query.
    * ```VECTOR_DB_PATH``` &mdash; The path of db.
    * ```API_KEY``` &mdash; API key for openai
    * ```TEMPERATURE``` &mdash; Modulate the diversity of LLM output. Higher value allows more diverse output.
    * ```TASK``` &mdash; Designate task (Options: [**1**: Paragraph matching, **2**: Relation matching, **3**: Skill matching, **4**: Method matching])
    * ```--in_context_sample``` &mdash; Flag to put in-context sample for relation matching task.
    * ```RESULT_PATH``` &mdash; The path to save result.
    * ```--ignore_wandb``` &mdash; Flag to deactivate wandb
    * ```WANDB_PROJECT``` &mdash; Project name for wandb
    * ```WANDB_ENTITY``` &mdash; Entity name for wandb


### Inference with Edu-Llama
* For Edu-Llama, promptize is needed first with `edu_llama_promptize.py`:
    ```
    edu_llama_promptize.py --query_path=<QUERY_PATH> \
                 --vector_db_path=<VECTOR_DB_PATH> \
                 --split=<SPLIT> \
                 --task=<TASK>
    ```
    * ```QUERY_PATH``` &mdash; The path of '`solvook_handout_te.csv`' exists.
    * ```VECTOR_DB_PATH``` &mdash; The path of '`vector_db.json`' exists.
    * ```SPLIT``` &mdash; Choose which splits to load (Options: [**tr**: Training set, **val**: Validation set, **te**: Test set])
    * ```TASK``` &mdash; Designate task (Options: [**1**: Paragraph matching, **2**: Relation matching, **3**: Skill matching, **4**: Method matching])


* If you want to edit argument for training, edit file **examples/train_lora/llama3_lora_predict.yaml**

    ```bash
    bash inference.sh
    ```

---
<br>


# Acknowledgements
* We heavilty refered the code from [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory) and [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). We appreciate the authors for sharing their code.