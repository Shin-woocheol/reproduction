# LNS-EECBS
high-level : LNS  
low-level : EECBS

## Running Example
```bash
python main.py --wandb --seed 7777 --max_t 10

./run_experiment.sh test_scenario_seed7777 50 25 30
scenario_fol=$1
scenario_count=$2
parallel_count=$3
max_t=$4
```

then in ./LNS_result/{scenario_fol}/summary.txt
experiment result saved.
