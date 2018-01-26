import pandas as pd


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def viterbi(nodes, zy):
    paths = {'b': nodes[0]['b'], 's':nodes[0]['s']}  # 第一层，只有两个节点
    for layer in range(1, len(nodes)):  # 后面的每一层
        paths_ = paths.copy()  # 先保存上一层的路径
        # node_now 为本层节点， node_last 为上层节点
        paths = {}  # 清空 path
        for node_now in nodes[layer].keys():
            # 对于本层的每个节点，找出最短路径
            sub_paths = {}
            # 上一层的每个节点到本层节点的连接
            for path_last in paths_.keys():
                if path_last[-1] + node_now in zy.keys():  # 若转移概率不为 0
                    sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + zy[path_last[-1] + node_now]
            # 最短路径,即概率最大的那个
            sr_subpaths = pd.Series(sub_paths)
            sr_subpaths = sr_subpaths.sort_values()  # 升序排序
            node_subpath = sr_subpaths.index[-1]  # 最短路径
            node_value = sr_subpaths[-1]   # 最短路径对应的值
            # 把 node_now 的最短路径添加到 paths 中
            paths[node_subpath] = node_value
    # 所有层求完后，找出最后一层中各个节点的路径最短的路径
    sr_paths = pd.Series(paths)
    sr_paths = sr_paths.sort_values()  # 按照升序排序
    return sr_paths.index[-1]  # 返回最短路径（概率值最大的路径）
