from utils.astar import graph_astar


def cost(agent_pos, solution, graph):
    '''
    agent_pos : agent position list
    solution : agent별 assign된 task와 좌표
    graph : grid graph.
    '''
    #* 이 코드에서는 각 agent에 할당된 task를 순회하는데에 걸리는 step을 모두 계산함.
    #* 그래서 agent_cost_list의 sum을 해주면 모든 agent가 task 수행시 움직이는 step의 합
    #* max를 해서 넘기면 task all solve까지 걸리는 시간
    #* solution은 agent에 할당된 task와 task의 좌표.
    # h_tasks : {0: [{34: [[2, 3]]}, {1: [[1, 7]]}, {45: [[4, 25]]}],
    agent_cost_list = list()
    for i in solution:
        path = list()
        agent_cost = 0 if len(solution[i]) == 0 else graph_astar(graph, agent_pos[i],
                                                                 list(solution[i][0].values())[0][0])[1]
        for a in solution[i]:
            for b in a.values():
                path += b
        for s, g in zip(path[:-1], path[1:]):
            agent_cost += graph_astar(graph, s, g)[1]
        agent_cost_list.append(agent_cost)

    return sum(agent_cost_list), max(agent_cost_list)
