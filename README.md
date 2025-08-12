# Natural-Language-Processing
Project Natural Language Processing

IMDB Training dataset : https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

IMDB Validate dataset : https://www.kaggle.com/datasets/krystalliu152/imbd-movie-reviewnpl

Reference papar ElaLora method : https://arxiv.org/abs/2504.00254

Reference papar Lora method : https://arxiv.org/abs/2106.09685

#ElaLoRA :
This is the implementation of ElaLoRA: Elastic & Learnable Low-Rank Adaptation for Efficient Model Fine-Tuning.

#Repository Overview
There are several directories in this repo:

- loralib/ contains the source code of the updated package loralib, which include our implementation of ElaLoRA (loralib/elalora.py) and needs to be installed to run the examples;
  
- NLU/ contains the implementation of ElaLoRA in DeBERTaV3-base, which produces the results on the GLUE benchmark;
  
- NLU/src/transformers/trainer.py contains the trainer to update ranks for ElaLoRA algorithm;
  
- NLG_QA/ contains the implementation of ElaLoRA in BART-base, which produces the results on the XSum benchmark;
  
- IMAGE_CLASS/ contains the implementation of ElaLoRA in ViT-B/16, which produces the results on the VTAB benchmark.
  
#Setup Environment (NLU)
This repository contains the setup procedures for the NLU Task, and NLG_QA follows the same procedure. For detailed instructions regarding IMAGE_CLASS/, please refer to the README files located in IMAGE_CLASS folders.

Create and activate the conda env
conda create -n NLU python=3.7
conda activate NLU 
Install Pytorch
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
Install the pre-requisites (NLU)
Install dependencies:

cd NLU
pip install -r requirements.txt
Install transformers: (here we fork NLU examples from microsoft/LoRA and build our examples based on their transformers version, which is v4.4.2.)

pip install -e . 
Install the updated loralib:

pip install -e ../loralib/
Example Usage
Check the folder NLU for more details about reproducing the GLUE results. An example of adapting DeBERTaV3-base on RTE:

cd NLU

python -m torch.distributed.launch --master_port=8679 --nproc_per_node=1 \
examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name rte \
--apply_elalora --apply_lora \
--lora_r 10 \
--init_warmup 300 --final_warmup 500 --mask_interval 50 \
--b 4 --k 2 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 16 \
--do_train --do_eval \
--max_seq_length 256 \
--per_device_train_batch_size 64 \
--learning_rate 1.2e-3 --num_train_epochs 50 \
--warmup_steps 1000 \
--cls_dropout 0.15 --weight_decay 0 \
--evaluation_strategy steps --eval_steps 3000 \
--save_strategy steps --save_steps 30000 \
--logging_steps 500 \
--seed 6 \
--enable_scheduler True \
--root_output_dir ./ela_debertabase/glue/rte \
--overwrite_output_dir
Hyperparameter Setup
apply_lora: Apply LoRA to the target model.
apply_elalora: Further apply ElaLoRA for the model that have been modified by LoRA.
lora_module: The types of modules updated by LoRA.
lora_r: The initial rank of each incremental matrix.
b: Number of total ranks pruned/added for each round.
k: Max rank pruned/added for each matrix in each round.
init_warmup: The steps of initial warmup for budget scheduler.
final_warmup: The steps of final warmup for budget scheduler.
mask_interval: The time interval between two budget allocations.
enable_scheduler: If enabled, total number of rank change (b) will decrease with time.
beta1 and beta2: The coefficient of exponentional moving average when updating importance scores.
reg_orth_coef: The weight of orthongonal regularization.




