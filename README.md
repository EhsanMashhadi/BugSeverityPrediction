# BugSeverityPrediction

### Paper
You can find the paper here: TBC

### Artifact Description
This artifact contains all data, code, and scripts required to run the paper's experiment to reproduce the results. The structure of folders and files are:
ESBS 


### Data

The `data` folder contains bugs from Defects4tJ and Bugs.jar datasets. This folder contains a preprocessing script that unify bug severity values, scale the source code metrics and create `train`, `val`, and `test` splits.

Running this script using ```bash preprocessing.sh``` command generates 6 files containing `train`, `val`, `tests` splits in `jsonl` (compatible with CodeBERT experiments) and `csv` (compatible with source code metrics experiments) formats. 

These files should be copied to the `dataset` folder that is used by the model training scripts. Files available in the `dataset` folder represent the splits used for our experiments that are provided in the paper.

### Prerequisite
1. Clone the repository
   - `git clone git@github.com:EhsanMashhadi/BugSeverityPrediction.git` 
2. Install dependencies (You may need to change the torch version for running on your GPU/CPU)
   - Install Git
   - Install SVN
   - Install [Defects4J](https://github.com/rjust/defects4j) (Follow all the steps in the provided installation guide)
   - Install [Bugs.jar](https://github.com/bugs-dot-jar/bugs-dot-jar) (You must install this in the `data_gathering` directory)
   - `pip install pandas==1.4.2`
   - `pip install jira`
   - `pip install beautifulsoup4`
   - `pip3 install lxml`
   - `pip install transformers==4.18.0`
   - `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html`
   - `pip install scikit-learn==1.1.1`
   - `pip install xgboost==1.6.1`
   - `pip install seaborn==0.11.2`
4. Adding the project root folder to the `PYTHONPATH`
   - `export PYTHONPATH=$PYTHONPATH:/rootpath/you/clone/the/project`
5. Running data preprocessing (You can skip this step and use the available files in the dataset folder to replicate paper's results)
   - `cd BugSeverityPrediction/data`
   - `bash preprocessing.sh`
   - Copy generated `jsonl` and `csv` files into the dataset folder

### Running Source Code Metrics Models Experiments (RQ1)
1. `cd BugSeverityPrediction/models/code_metrics`
2. `bash train_test.sh`
3. Results are generated in the `log` folder

### Running CodeBERT Model Experiments (RQ2)
1. `cd BugSeverityPrediction/models/code_representation/codebert`
2. Set `CodeBERT` as the `model_arch` parameter's value in `train.sh` file
3. `bash train.sh` for training the model
4. `bash inference.sh` for evaluating the model with the `test` split
5. Results are generated in the `log` folder

### Running Source Code Metrics Integration with CodeBERT Model Experiments (RQ3)

1. `cd BugSeverityPrediction/models/code_representation/codebert`
2. Set `ConcatInline` or `ConcatCLS` as the `model_arch` parameter's value in `train.sh` file
3. `bash train.sh` for training the model
4. `bash inference.sh` for evaluating the model with the `test` split
5. Results are generated in the `log` folder

### How to run with different config/hyperparameters?
   - You can change/add different hyperparameters/configs in `train.sh` and `inference.sh` files.

### Have trouble running on GPU?
1. Check the `CUDA` and `PyTorch` compatibility
2. Assign the correct values for `CUDA_VISIBLE_DEVICES`, `gpu_rank`, and `world_size` based on your GPU numbers in all scripts.
3. Run on CPU by removing the `gpu_rank`, and `world_size` options in all scripts.

### Dataset Collection
Dataset collection scripts are provided in the `ESBS` folder. This folder contains Python and Java code that fetch severity labels, extract buggy methods, and calculate source code metrics.
