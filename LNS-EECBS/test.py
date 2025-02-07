import os
import pickle

from copy import deepcopy
from utils.generate_scenarios import load_scenarios
from utils.soc_ms import eecbs_cost

def load_tasks(exp_name, seed, max_t):
    file_path = f"saved/{exp_name}_seed{seed}_max_t{max_t}.pkl"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No saved tasks found at {file_path}")
    with open(file_path, 'rb') as f:
        tasks = pickle.load(f)
    print(f"Tasks loaded from {file_path}")
    return tasks

tasks = load_tasks(exp_name, seed, max_t)
scenario = load_scenarios(f'323220_1_{M}_{N}_eval/scenario_1.pkl')
grid, graph, agent_pos, total_tasks = scenario[0], scenario[1], scenario[2], scenario[3]
init_agent_pos = deepcopy(agent_pos)
total_tasks = [[[x, y]] for x, y in total_tasks]

eecbs_ms = eecbs_cost(agent_pos, tasks, total_tasks, exp_name, grid.shape[0])
print(f"[TEST] EECBS_MAKESPAN: {eecbs_ms}")