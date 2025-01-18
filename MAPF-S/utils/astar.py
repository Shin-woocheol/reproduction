import bisect
import heapq

import numpy as np


class Node:
    def __init__(self, parent=None, idx=None):
        self.parent = parent
        self.idx = idx
        self.f = 0
        self.g = 0
        self.h = 0

    def __eq__(self, other):
        return self.idx == other.idx

    def __lt__(self, other):
        return self.g + self.h < other.g + other.h

#! 나중에 map size를 늘린다면, 아래 node.h에서 normalize 풀어주는 것을 size에 맞게 바꿔줘야함.
def graph_astar(g, start, end):
    start = tuple(start)
    end = tuple(end)
    if start == end:
        return 0, 0
    path = list()
    path_cost = 0
    queue = list()
    openpath = dict()
    closepath = list()
    found = False
    node = Node(None, start)
    end_node = Node(None, end)

    openpath[start] = node
    queue.append(node)
    while openpath and not found:
        current_node = queue.pop(0)
        openpath.pop(current_node.idx)
        closepath.append(current_node)
        for new_idx in g.neighbors(current_node.idx):
            node = Node(current_node, new_idx)
            node.g = current_node.g + 1
            #! map size 바뀌면 바꿔줘야함.
            node.h = abs(np.array(g.nodes[node.idx]['loc']) - np.array(g.nodes[end_node.idx]['loc'])).sum().item() * 31 
            #* 31을 곱한 것이, map size가 32이고 feature에 position을 넣어줄 때, normalize해서 넣어줫기 때문.
            node.f = node.g + node.h

            if node in closepath:
                continue
            elif node == end_node:
                current = node
                while current is not None:
                    path.append(current.idx)
                    current = current.parent
                path = path[::-1]
                for p in range(len(path) - 1):
                    path_cost += 1 #* one step에 대한 비용을 1로 계산해주고 있음.
                # print(f"return of graph a star \n path : {path} \n path_cost : {path_cost}")
#                  path : [(28, 23), (28, 22), (28, 21), (28, 20), (28, 19)] 
#                   path_cost : 4
                return path, path_cost  # Return reversed path
            else:
                duplicated = openpath.get(node.idx)
                if not duplicated:
                    openpath[node.idx] = node
                    bisect.insort_left(queue, node)
                elif duplicated.g > node.g:
                    left = bisect.bisect_left(queue, duplicated)
                    right = bisect.bisect_right(queue, duplicated)
                    queue.pop(queue.index(duplicated, left, right))
                    openpath[node.idx] = node
                    bisect.insort_left(queue, node)


def grid_astar(grid, start, goal, w=1.0):
    """
    Four-connected Grid
    Return a path (in REVERSE order!)
    a path is a list of node ID (not x,y!)
    """
    output = list()
    (nyt, nxt) = grid.shape  # nyt = ny total, nxt = nx total
    action_set_x = [-1, 0, 1, 0]
    action_set_y = [0, -1, 0, 1]
    open_list = []
    heapq.heappush(open_list, (0, start))
    close_set = dict()
    parent_dict = dict()
    parent_dict[start] = -1
    g_dict = dict()
    g_dict[start] = 0
    gx = goal % nxt
    gy = int(np.floor(goal / nxt))
    search_success = True
    while True:
        if len(open_list) == 0:
            search_success = False
            break
        cnode = heapq.heappop(open_list)
        cid = cnode[1]
        curr_cost = g_dict[cid]
        if cid in close_set:
            continue
        close_set[cid] = 1
        if cid == goal:
            break
        cx = cid % nxt
        cy = int(np.floor(cid / nxt))
        for action_idx in range(len(action_set_x)):
            nx = cx + action_set_x[action_idx]
            ny = cy + action_set_y[action_idx]
            if ny < 0 or ny >= nyt or nx < 0 or nx >= nxt:
                continue
            if grid[ny, nx] > 0.5:
                continue
            nid = ny * nxt + nx
            heu = np.abs(gx - nx) + np.abs(gy - ny)  # manhattan heuristic
            g_new = curr_cost + 1
            if nid not in close_set:
                if nid not in g_dict:
                    heapq.heappush(open_list, (g_new + w * heu, nid))
                    g_dict[nid] = g_new
                    parent_dict[nid] = cid
                else:
                    if g_new < g_dict[nid]:
                        heapq.heappush(open_list, (g_new + w * heu, nid))
                        g_dict[nid] = g_new
                        parent_dict[nid] = cid
    # end of while

    # reconstruct path
    if search_success:
        cid = goal
        output.append(cid)
        while parent_dict[cid] != -1:
            cid = parent_dict[cid]
            output.append(cid)
    else:
        # do nothing
        print(" fail to plan !")
    return output
