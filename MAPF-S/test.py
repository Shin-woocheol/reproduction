import os
import torch
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

from utils.utils import fix_seed
from nn.agent import Agent
from utils.generate_scenarios import load_scenarios
from main import run_episode

# 폴더에 저장된 모델 중 가장 마지막 itr모델 불러옴.
def find_latest_model(save_base_dir):
    model_files = [f for f in os.listdir(save_base_dir) if f.startswith("agent_epoch_") and f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError(f"No model weights found in {save_base_dir}.")
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by epoch number
    return os.path.join(save_base_dir, model_files[-1])  # Return the path to the latest model

def evaluate_model(args, exp_name):
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    agent = Agent(gpu=args.gpu).to(device)
    
    save_base_dir = os.path.join("saved_models", exp_name)
    latest_model_path = find_latest_model(save_base_dir)
    print(f"Loading model from: {latest_model_path}")
    
    agent.load_state_dict(torch.load(latest_model_path, map_location=device))
    agent.eval()

    # eval로직
    num_eval_map = [1, 3, 5, 7, 10]
    eval_result = {num: [] for num in num_eval_map}

    for i in tqdm(range(args.n_map_eval), desc="Evaluating", leave=False):
        scenario_dir = '323220_1_{}_{}/scenario_{}.pkl'.format(args.n_agent, args.n_task, i + 1)
        eval_cost, _ = run_episode(agent, args.n_agent, args.n_task, exp_name, 
                                   train=False, scenario_dir=scenario_dir, VISUALIZE=args.eval_visualize, 
                                   n_sample=args.n_task_sample)
        if eval_cost is not None:
            for num in num_eval_map:
                if i < num:
                    eval_result[num].append(eval_cost)

    subset_means = {num: (np.mean(eval_result[num]) if eval_result[num] else None) for num in num_eval_map}
    print("Evaluation Results:")
    for num, mean in subset_means.items():
        print(f"Maps: {num}, Mean Eval Cost: {mean if mean is not None else 'EMPTY_LIST'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_map_eval', type=int, default=10, help="num of maps for eval")
    parser.add_argument('--seed', type = int, default = 1)
    parser.add_argument('--n_task_sample', type=int, default=1, help="num of task assignment samples")
    parser.add_argument('--n_agent', type=int, default=10, help="num of agents")
    parser.add_argument('--n_task', type=int, default=20, help="num of tasks")
    parser.add_argument('--task_threshold', type=int, default=10, help="task rescheduling threshold")
    parser.add_argument('--eval_visualize', action='store_true', help="Enable eval visualization")
    parser.add_argument('--gpu', action='store_true', help="Enable GPU usage")
    parser.add_argument('--exp_name', type=str, required=True, help="Experiment name for saved models")
    args = parser.parse_args()

    fix_seed(args.seed)

    evaluate_model(args, args.exp_name)

#python test.py --exp_name "20241223_225702" --gpu --n_map_eval 10 --eval_visualize --n_task_sample 1