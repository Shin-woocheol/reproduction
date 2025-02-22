import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

curr_path = os.path.realpath(__file__)
fig_dir = os.path.join(Path(curr_path).parent.parent, 'fig')
gray = (0.5019607843137255, 0.5019607843137255, 0.5019607843137255)
black = (0.1, 0.1, 0.1)


def vis_graph(graph):
    pos = dict()
    for i in range(len(graph)):
        pos[list(graph.nodes)[i]] = graph.nodes[list(graph.nodes)[i]]['loc']
    nx.draw(graph, pos=pos, with_labels=False, node_size=50)

    try:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
    except OSError:
        print("Error: Cannot create the directory.")

    plt.savefig(fig_dir + '/graph.png')
    plt.clf()


def vis_dist(graph, agents, tasks):
    pos = dict()
    for i in range(len(graph)):
        pos[list(graph.nodes)[i]] = graph.nodes[list(graph.nodes)[i]]['loc']
    nx.draw(graph, pos=pos, nodelist=[tuple(a) for a in agents], node_color='r', node_size=100)
    for j in range(len(tasks)):
        nx.draw(graph, pos=pos, nodelist=[tuple(t) for t in tasks], node_color='b', node_size=100)

    try:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
    except OSError:
        print("Error: Cannot create the directory.")

    plt.savefig(fig_dir + '/distribution.png')
    plt.clf()


def vis_ta(graph, agents, tasks, itr, total_tasks=None, task_finished=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = dict()
    for i in range(len(graph)):
        pos[list(graph.nodes)[i]] = graph.nodes[list(graph.nodes)[i]]['loc']
    colors = [plt.cm.get_cmap('rainbow')(i / len(agents)) for i in range(len(agents))]

    labeldict = dict()
    for i, a in enumerate(agents):
        labeldict[tuple(a)] = i

    if type(tasks) == list:
        temp_tasks = dict()
        for i, t in enumerate(tasks):
            temp_tasks[i] = [{0: t}]
        tasks = temp_tasks

    task_color_dict = dict()
    if total_tasks is not None:
        for i, (t, finished) in enumerate(zip(total_tasks, task_finished)):
            labeldict[tuple(t)] = i
            if finished:
                task_color_dict[tuple(t)] = gray
            else:
                task_color_dict[tuple(t)] = gray

        # nx.draw(graph, pos=pos, nodelist=task_nodes, node_size=100, node_shape='X', node_color='gray')

    for ag_idx, task in tasks.items():
        for t in task:
            for _t in t.values():
                # labeldict[tuple(_t[0])] = ag_idx
                task_color_dict[tuple(_t[0])] = colors[ag_idx]

    # t_colors = list()
    # for c, t in enumerate(tasks.values()):
    #     t_colors += [colors[c] for _ in range(len(t))]
    #     for _t in t:
    #         task_nodes.append(tuple(list(_t.values())[0][0]))
    task_node_list = list(task_color_dict.keys())
    node_color_list = list(task_color_dict.values())

    nx.draw(graph, pos=pos, nodelist=task_node_list, node_size=100, node_shape='X', node_color=node_color_list)
    nx.draw(graph, pos=pos, nodelist=[tuple(a) for a in agents], node_size=100, node_color=colors)
    nx.draw_networkx_labels(graph, pos, labeldict)

    try:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
    except OSError:
        print("Error: Cannot create the directory.")

    ax.axis('equal')
    ax.set_title(itr)
    plt.axis('off')
    fig.tight_layout()
    fig.savefig(fig_dir + '/ta_{}.png'.format(itr), bbox_inches='tight')
    plt.clf()
    plt.close()
