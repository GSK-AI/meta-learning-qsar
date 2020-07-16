#!/bin/bash
#### Train and evaluate MAML, FOMAML, and ANIL with 5 different seeds

CONDA_ENV=metalearning
module load anaconda3
source activate ${CONDA_ENV}

### Directories and variables 
METRICS="average_precision_score,roc_auc_score"
SOURCE="data/chembl20"
BASE_SAVE_PATH="trained_models/chembl20"
BASE_CONFIG_PATH="configs"
MIN_SEED=0
MAX_SEED=4

######### Baselines ##########

SAVE_PATH=${BASE_SAVE_PATH}/multitask
# Testing
for f in ${SAVE_PATH}/{${MIN_SEED}..${MAX_SEED}}/best_model.pth; do
    # k-NN
    sbatch  --time=5-00 \
            --mem=40G \
            --array=$MIN_SEED-$MAX_SEED \
            --cpus-per-task=4 \
            --gres=gpu:1 \
            --partition=aiml \
            --wrap "source activate ${CONDA_ENV} && python src/evaluate_transfer_learning.py    --init_path ${CONFIG_PATH} \
                                                                                                --num_data 128 \
                                                                                                --source ${SOURCE} \
                                                                                                --save_path ${SAVE_PATH}/\${SLURM_ARRAY_TASK_ID} \
                                                                                                --split_seed \${SLURM_ARRAY_TASK_ID} \
                                                                                                --multitask \
                                                                                                --knn \
                                                                                                --metrics ${METRICS}"
    # Finetune-Top
    sbatch  --time=5-00 \
            --mem=40G \
            --array=$MIN_SEED-$MAX_SEED \
            --cpus-per-task=4 \
            --gres=gpu:1 \
            --partition=aiml \
            --wrap "source activate ${CONDA_ENV} && python src/evaluate_transfer_learning.py    --init_path ${CONFIG_PATH} \
                                                                                                --num_data 128 \
                                                                                                --source ${SOURCE} \
                                                                                                --save_path ${SAVE_PATH}/\${SLURM_ARRAY_TASK_ID} \
                                                                                                --split_seed \${SLURM_ARRAY_TASK_ID} \
                                                                                                --multitask \
                                                                                                --freeze \
                                                                                                --metrics ${METRICS}"
    # Finetune-All
    sbatch  --time=5-00 \
            --mem=40G \
            --array=$MIN_SEED-$MAX_SEED \
            --cpus-per-task=4 \
            --gres=gpu:1 \
            --partition=aiml \
            --wrap "source activate ${CONDA_ENV} && python src/evaluate_transfer_learning.py    --init_path ${CONFIG_PATH} \
                                                                                                --num_data 128 \
                                                                                                --source ${SOURCE} \
                                                                                                --save_path ${SAVE_PATH}/\${SLURM_ARRAY_TASK_ID} \
                                                                                                --split_seed \${SLURM_ARRAY_TASK_ID} \
                                                                                                --multitask \
                                                                                                --metrics ${METRICS}"

############ MAML ############
# Set MAML paths
SAVE_PATH=${BASE_SAVE_PATH}/maml
# Testing
for f in ${SAVE_PATH}/{${MIN_SEED}..${MAX_SEED}}/best_model.pth; do
    sbatch  --time=5-00 \
            --mem=40G \
            --array=$MIN_SEED-$MAX_SEED \
            --cpus-per-task=4 \
            --gres=gpu:1 \
            --partition=aiml \
            --wrap "source activate ${CONDA_ENV} && python src/evaluate_transfer_learning.py    --init_path ${f} \
                                                                                                --num_data 128 \
                                                                                                --source ${SOURCE} \
                                                                                                --split_seed \${SLURM_ARRAY_TASK_ID} \
                                                                                                --test_set \
                                                                                                --metrics ${METRICS}"
done



############ ANIL ############
# Set ANIL paths
SAVE_PATH=${BASE_SAVE_PATH}/anil
## Testing
for f in ${SAVE_PATH}/{${MIN_SEED}..${MAX_SEED}}/best_model.pth; do
    sbatch  --time=5-00 \
            --mem=40G \
            --array=$MIN_SEED-$MAX_SEED \
            --cpus-per-task=4 \
            --gres=gpu:1 \
            --partition=aiml \
            --wrap "source activate ${CONDA_ENV} && python src/evaluate_transfer_learning.py    --init_path ${f} \
                                                                                                --num_data 128 \
                                                                                                --source ${SOURCE} \
                                                                                                --split_seed \${SLURM_ARRAY_TASK_ID} \
                                                                                                --test_set \
                                                                                                --anil \
                                                                                                --metrics ${METRICS}"
done

############ FOMAML ############
# Set FOMAML paths
SAVE_PATH=${BASE_SAVE_PATH}/fomaml
## Testing
for f in ${SAVE_PATH}/{${MIN_SEED}..${MAX_SEED}}/best_model.pth; do
    sbatch  --time=5-00 \
            --mem=40G \
            --array=$MIN_SEED-$MAX_SEED \
            --cpus-per-task=4 \
            --gres=gpu:1 \
            --partition=aiml \
            --wrap "source activate ${CONDA_ENV} && python src/evaluate_transfer_learning.py    --init_path ${f} \
                                                                                                --num_data 128 \
                                                                                                --source ${SOURCE} \
                                                                                                --split_seed \${SLURM_ARRAY_TASK_ID} \
                                                                                                --test_set \
                                                                                                --metrics ${METRICS}"
done
