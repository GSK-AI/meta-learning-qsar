# Meta-Learning Initializations for Low Resource Drug Discovery
This repo contains accompanying code for the publication ["Meta-Learning Initializations for Low Resource Drug Discovery" (Nguyen et al.)](https://arxiv.org/abs/2003.05996).

## Instructions
### Cloning and setting up your environment
```bash
git clone https://mygithub.gsk.com/gsk-tech/meta-learning-qsar.git
conda env create --name metalearning --file environment.yaml
source activate metalearning
```
### Setting PYTHONPATH
```bash
cd meta-learning-qsar
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
### Setting OE_LICENSE 
This step requires the OpenEye license file and is necessary for running src/featurize.py. Change `<path>` to the appropriate directory.
```bash
export OE_LICENSE=<path>/oe_license.txt
```
## Usage
### Reproducing experiments with ChEMBL20
Extracting and combining chunked and featurized data
```bash
python exp/preprocess.py
```
Train Baselines, MAML, FOMAML, and ANIL using the provided splits
```bash
./exp/train_and_evaluate.sh 
```
Once training is done, generate test statistics on held-out test tasks by running
```bash
./exp/test.sh 
```
### Training on custom data
#### Training
First featurize data from SMILES to graph representation.
```
python src/featurize.py \
    --data <csv file> \
    --smiles_col <name of SMILES column> \
    --output_col <name of output columns> \
    --output_path <folder to store featurized data>
```
Use `src/train_maml.py` to kick off MAML training. The two required arguments are `--save_path` and `--source`.
```
python src/train_maml.py \ 
    --save_path <directory to store checkpoint> \
    --source <directory where training and validation data is stored>
    ...
```
Use `src/validate_maml.py` to calculate validation metrics from saved checkpoints. This python script will kick off validation slurm jobs as new checkpoints are found. `--monitor_path` and `--source` should be the the same as `--save_path` and `--source` used in `src/train_maml.py`
```
python src/validate_maml.py  \
    --monitor_path <directory to store checkpoint> \
    --source <directory where training and validation data is stored> 
    ...
```
**Notes**
- Usage instructions can be found at the top of each file.
- Description of available arguments for each script can be obtained by using the `--help` flag.
- For example usage of these files, see `exp/train_and_evaluate.sh` and `exp/test.sh`.
## Contact
For questions, please feel free to reach out via email at cuong.q.nguyen@gsk.com.
