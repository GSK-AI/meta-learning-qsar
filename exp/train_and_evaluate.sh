#!/bin/bash
#### Train and evaluate Baseline, MAML, FOMAML, and ANIL with 5 different seeds

CONDA_ENV=metalearning
module load anaconda3
source activate ${CONDA_ENV}

### Directories and variables 
METRICS="average_precision_score,roc_auc_score"
SOURCE="data/chembl20"
BASE_SAVE_PATH="trained_models/chembl20"
BASE_CONFIG_PATH="configs"
NUM_DATA_VAL=96
MIN_SEED=0
MAX_SEED=4

######### Baselines ##########
CONFIG_PATH=${BASE_CONFIG_PATH}/multitask_flags.txt
SAVE_PATH=${BASE_SAVE_PATH}/multitask
# Training
sbatch  --time=5-00 \
        --mem=40G \
        --array=$MIN_SEED-$MAX_SEED \
        --cpus-per-task=4 \
        --gres=gpu:1 \
        --partition=aiml \
        --wrap "source activate ${CONDA_ENV} && python src/train_multitask.py   --flagfile ${CONFIG_PATH} \
                                                                                --source ${SOURCE} \
                                                                                --save_path ${SAVE_PATH}/\${SLURM_ARRAY_TASK_ID} \
                                                                                --seed \${SLURM_ARRAY_TASK_ID}"


############ MAML ############
# Set MAML paths
CONFIG_PATH=${BASE_CONFIG_PATH}/maml_flags.txt
SAVE_PATH=${BASE_SAVE_PATH}/maml
# Training
sbatch  --time=5-00 \
        --mem=40G \
        --array=$MIN_SEED-$MAX_SEED \
        --cpus-per-task=4 \
        --gres=gpu:1 \
        --partition=aiml \
        --wrap "source activate ${CONDA_ENV} && python src/train_maml.py    --flagfile ${CONFIG_PATH} \
                                                                            --source ${SOURCE} \
                                                                            --save_path ${SAVE_PATH}/\${SLURM_ARRAY_TASK_ID} \
                                                                            --seed \${SLURM_ARRAY_TASK_ID}"

# Validation
sbatch  --time=5-00 \
        --array=$MIN_SEED-$MAX_SEED \
        --partition=up-cpu \
        --wrap "python src/validate_maml.py  --monitor_path ${SAVE_PATH}/\${SLURM_ARRAY_TASK_ID} \
                                             --source ${SOURCE} \
                                             --num_data ${NUM_DATA_VAL} \
                                             --seed \${SLURM_ARRAY_TASK_ID} \
                                             --metrics ${METRICS}"


############ ANIL ############
# Set ANIL paths
CONFIG_PATH=${BASE_CONFIG_PATH}/anil_flags.txt
SAVE_PATH=${BASE_SAVE_PATH}/anil
## Training
sbatch  --time=5-00 \
        --mem=40G \
        --array=$MIN_SEED-$MAX_SEED \
        --cpus-per-task=4 \
        --gres=gpu:1 \
        --partition=aiml \
        --wrap "source activate ${CONDA_ENV} && python src/train_maml.py    --flagfile ${CONFIG_PATH} \
                                                                            --source ${SOURCE} \
                                                                            --save_path ${SAVE_PATH}/\${SLURM_ARRAY_TASK_ID} \
                                                                            --seed \${SLURM_ARRAY_TASK_ID}"
## Evaluation
sbatch --time=5-00 \
       --array=$MIN_SEED-$MAX_SEED \
       --partition=up-cpu \
       --wrap "python src/validate_maml.py  --monitor_path ${SAVE_PATH}/\${SLURM_ARRAY_TASK_ID} \
                                            --source ${SOURCE} \
                                            --anil \
                                            --num_data ${NUM_DATA_VAL} \
                                            --seed \${SLURM_ARRAY_TASK_ID} \
                                            --metrics ${METRICS}"


############ FOMAML ############
# Set FOMAML paths
CONFIG_PATH=${BASE_CONFIG_PATH}/fomaml_flags.txt
SAVE_PATH=${BASE_SAVE_PATH}/fomaml
## Training
sbatch  --time=5-00 \
        --mem=40G \
        --array=$MIN_SEED-$MAX_SEED \
        --cpus-per-task=4 \
        --gres=gpu:1 \
        --partition=aiml \
        --wrap "source activate ${CONDA_ENV} && python src/train_maml.py    --flagfile ${CONFIG_PATH} \
                                                                            --source ${SOURCE} \
                                                                            --save_path ${SAVE_PATH}/\${SLURM_ARRAY_TASK_ID} \
                                                                            --seed \${SLURM_ARRAY_TASK_ID}"
## Evaluation
sbatch --time=5-00 \
       --array=$MIN_SEED-$MAX_SEED \
       --partition=up-cpu \
       --wrap "python src/validate_maml.py  --monitor_path ${SAVE_PATH}/\${SLURM_ARRAY_TASK_ID} \
                                            --source ${SOURCE} \
                                            --num_data ${NUM_DATA_VAL} \
                                            --seed \${SLURM_ARRAY_TASK_ID} \
                                            --metrics ${METRICS}"

