# reproduction

## EECBS setting
On the EECBS,
```bash
cmake -DCMAKE_BUILD_TYPE=RELEASE .
make
```

## Running Example
```bash
python main.py --wandb --gpu --eval_visualize --batch_size 8 --n_map_eval 10 --n_task_sample 50
```
batch_size : num of trajectory used in loss  
n_map_eval : num of map for eval  
n_task_sample : num of task assignment sample. 'Samples' in papaer experiments

## Environment
```bash
conda env create -f environment.yml
```

## Baseline Algorithm
LNS-EECBS is in LNS-EECBS folder