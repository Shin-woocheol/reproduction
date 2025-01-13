import os
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

curr_path = os.path.realpath(__file__)
fig_dir = os.path.join(Path(curr_path).parent.parent, 'fig')
gray = (0.7, 0.7, 0.7)  #light gray

def visualize(graph, tasks, trajectories, completed_tasks, exp_name, output_name="final_visualization"):
    fig_dir = os.path.join('eval_fig', exp_name)
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.get_node_attributes(graph, 'loc')  # normalized되어있던 것 말고.

    nx.draw_networkx_edges(graph, pos=pos, edge_color=gray, width=1, ax=ax)

    colors = [plt.cm.get_cmap('tab10')(i / len(trajectories)) for i in range(len(trajectories))]

    # task
    for task_pos in tasks:
        task_node = tuple(task_pos)
        if task_node in pos:
            ax.scatter(pos[task_node][0], pos[task_node][1], marker='^', s=200, color=gray, edgecolor='black', linewidth=0.5)  # Task marker with black border

    # agent, traj
    for agent_id, traj in trajectories.items():
        color = colors[agent_id % len(colors)] 

        # traj
        traj_x = [pos[tuple(p)][0] for p in traj]
        traj_y = [pos[tuple(p)][1] for p in traj]
        ax.plot(traj_x, traj_y, color=color, linestyle='-.', linewidth=2)  # Trajectory line

        # agent
        initial_pos = tuple(traj[0])
        if initial_pos in pos:
            ax.scatter(pos[initial_pos][0], pos[initial_pos][1], marker='o', s=200, color=color, edgecolor='black')  # Agent circle
            ax.text(pos[initial_pos][0], pos[initial_pos][1], str(agent_id), color='black', fontsize=10, ha='center', va='center')  # Agent ID

        # complete task 표시
        for task_id in completed_tasks.get(agent_id, []):
            task_pos = tasks[task_id]
            task_node = tuple(task_pos)
            if task_node in pos:
                ax.scatter(pos[task_node][0], pos[task_node][1], marker='^', s=200, color=color, edgecolor='black', linewidth=0.5)  # Task colored by agent

    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    try:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        file_path = os.path.join(fig_dir, f'{output_name}_1.png')
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(fig_dir, f'{output_name}_{counter}.png')
            counter += 1
        fig.tight_layout()
        fig.savefig(file_path, bbox_inches='tight')
        print(f"Saved figure as {file_path}")
    except OSError:
        print("Error: Cannot create the directory.")

    plt.clf()
    plt.close(fig)
