import numpy as np
import copy
from .Node import Node

class TnT:
    """
        TreeInTree decision graph
        ----------
        N1 : int
            the number of merging phases that micro trees are merged into the graph
        N2 : int
             the number of rounds to grow and optimize micro trees
        graph : TnT, optional
            if None, train from scratch; else, train based on a predefined structure
    """
    def __init__(self, N1=2, N2=3, graph=None,**kwargs):
        self.N1 = N1
        self.N2 = N2
        self.kwargs_backup = kwargs
        if graph is None:
            self.graph=self._init_graph()
        else:
            self.graph = copy.deepcopy(graph)

    def _init_graph(self):
        """
            initialize graph with a leaf node
        """
        root = Node(label=0)
        return root

    def _get_children_mask(self, n, trX):  # n has to be an internal node
        """
            given a subset of train data "trX" at node "n", return sample index routed to the left/right child
            ----------
            n : Node
            trX : numpy array
        """
        if n.label is not None:
            print("access children of leaf nodes")
        if trX.shape[0] == 0:
            return np.array([]), np.array([])
        if n.substitute is None:
            pred = np.zeros(trX.shape[0])
            pred[trX[:, n.feature_index] > n.threshold] = 1
        else:
            pred = n.substitute.predict(trX)
        return (pred == 0), (pred == 1)


    def _get_mask(self, n, node_all, mask_all, X=None):
        """
            get the sample index arriving at node "n"
            ----------
            n : Node
            node_all : list of Node
                a list of parent nodes of "n"
            mask_all : list of numpy array
                a list of sample index at "node_all"
        """
        if X is None:
            X = self.trX
        if len(node_all) == 0:
            return np.array([True]*X.shape[0])
        for p in n.parents:
            if p not in node_all:
                return None
        m = np.array([False]*X.shape[0])
        for p in n.parents:
            m_temp = mask_all[node_all.index(p)]
            left_mask, right_mask = self._get_children_mask(p, X[m_temp, :])
            if n is p.left:
                m[m_temp] = m[m_temp] + left_mask
            if n is p.right:
                m[m_temp] = m[m_temp] + right_mask
        return m

    def _grow(self):
        """
            breath-first growing of TnT
        """
        nodes = [self.graph]
        node_all, mask_all = [], []
        while len(nodes) > 0:
            n = nodes.pop(0)
            m = self._get_mask(n, node_all, mask_all)

            if m is None:    # parents of n are not all visited
                continue
            trX, trY, sample_weight = self.trX[m,:], self.trY[m], self.sample_weight[m]

            if n.left is not None and n.right is not None:  # for internal nodes
                l_result = self.predict(trX, n.left)
                r_result = self.predict(trX, n.right)
                matter_index = l_result != r_result
                l_win_index = (l_result == trY) * matter_index
                r_win_index = (r_result == trY) * matter_index
                sen_trX = np.concatenate((trX[l_win_index, :], trX[r_win_index, :]), axis=0)
                sen_trY = np.concatenate((np.zeros(l_win_index.sum()), np.ones(r_win_index.sum())), axis=0)
                sen_sample_weight = np.concatenate((sample_weight[l_win_index], sample_weight[r_win_index]), axis=0)
                n.update_substitute(sen_trX, sen_trY, sen_sample_weight, **self.kwargs)
            else:  # for leaf node
                n.update_substitute(trX, trY, sample_weight, **self.kwargs)

            if n.left is not None and n.right is not None:  # internal nodes
                if n.left in nodes:
                    nodes.remove(n.left)
                nodes.append(n.left)
                if n.right in nodes:
                    nodes.remove(n.right)
                nodes.append(n.right)

            node_all.append(n)
            mask_all.append(m)

        return node_all, mask_all

    def fit(self, trX, trY, sample_weight=None):
        """
            fit on train data (trX, trY)
            ----------
            trX : numpy array
                feature array
            trY : numpy array
                ground truth
            sample_weight : numpy array (optional)
                same size as trY, sample-wise importance
        """
        self.trX, self.trY = trX, trY
        if sample_weight is None:
            self.sample_weight = np.ones_like(trY)/len(trY)
        else:
            self.sample_weight = sample_weight/np.sum(sample_weight)
        self.classes_ = np.unique(trY)
        self.kwargs = self.kwargs_backup
        if 'ccp_alpha' in self.kwargs:
            self.kwargs['ccp_alpha'] = self.kwargs['ccp_alpha']
        node_all, mask_all = [], []
        for i in range(self.N1):
            for _ in range(self.N2):
                node_all, mask_all=self._grow()
            self.prune()
            self._merge()
        self.trX, self.trY, self.sample_weight = None, None, None
        return node_all, mask_all

    def graph_traverse(self, teX=None, node=None):
        """
            breath-first traverse of a graph, return the list of nodes and a list sample index (corresponding to each node)
            ----------
            teX : numpy array
                samples at "node" for traverse
            node : Node
                starting node of graph traverse
        """
        if node is None:
            node = self.graph
        if teX is None:
            teX = self.trX
        l,_ = teX.shape
        node_all, mask_all = [node], [np.array([True]*l)]

        node_list = [node]
        while len(node_list) > 0:
            n = node_list.pop(0)
            i = node_all.index(n)
            m = mask_all[i]

            if n.label is not None:
                continue

            if n.left in node_list:
                node_list.remove(n.left)
            node_list.append(n.left)
            if n.right in node_list:
                node_list.remove(n.right)
            node_list.append(n.right)

            left_mask, right_mask = np.array([False]*teX.shape[0]), np.array([False]*teX.shape[0])
            left_mask[m], right_mask[m] = self._get_children_mask(n, teX[m, :])
            if n.left in node_all:
                temp_i = node_all.index(n.left)
                mask_all[temp_i] = mask_all[temp_i] + left_mask
            else:
                node_all.append(n.left)
                mask_all.append(left_mask)

            if n.right in node_all:
                temp_i = node_all.index(n.right)
                mask_all[temp_i] = mask_all[temp_i] + right_mask
            else:
                node_all.append(n.right)
                mask_all.append(right_mask)

        return node_all, mask_all

    def predict(self, teX, node=None):
        """
            predict on the test set
            ----------
            teX : numpy array
                test samples
            node : Node
                starting node of graph
        """
        if node is None:
            node = self.graph
        l, _ = teX.shape
        if l == 0:
            return np.array([])
        node_all, mask_all = self.graph_traverse(teX, node)
        pred=np.zeros(l)
        for i in range(len(node_all)):
            n = node_all[i]
            if (n.label is not None) and np.sum(mask_all[i])>0:
                if n.substitute is not None:
                    pred[mask_all[i]]=n.substitute.predict(teX[mask_all[i]])
                else:
                    pred[mask_all[i]]=n.label
        return pred.astype('int')

    def simple_graph_traverse(self, node=None):
        """
            similar to "graph_traverse", but with no data input
        """
        if node is None:
            node = self.graph
        node_all, node_list = [node], [node]
        while len(node_list) > 0:
            n = node_list.pop(0)
            i = node_all.index(n)
            if n.label is not None:
                continue
            if n.left in node_list:
                node_list.remove(n.left)
            node_list.append(n.left)
            if n.right in node_list:
                node_list.remove(n.right)
            node_list.append(n.right)
            if n.left in node_all:
                temp_i = node_all.index(n.left)
            else:
                node_all.append(n.left)
            if n.right in node_all:
                temp_i = node_all.index(n.right)
            else:
                node_all.append(n.right)
        return node_all

    def check_complexity(self):
        """
            return the number of internal/leaf nodes in TnT graph
        """
        node_all = self.simple_graph_traverse()
        num_internal, num_leaf = 0, 0
        for i in range(len(node_all)):
            n = node_all[i]
            if n.substitute is not None:
                print("substitute not done.")

            if n.label is None:
                num_internal += 1
            else:
                num_leaf += 1
        return num_internal, num_leaf

    def _merge(self):
        """
            merge all micro trees into TnT graph
        """
        node_all, mask_all = self.graph_traverse()
        for i in range(len(node_all)):
            n, m = node_all[i], mask_all[i]
            n.in_place_substitute()

    def prune(self):
        """
            prune TnT, remove dead nodes
        """
        deadnode, direction = [], []
        node_all, mask_all = self.graph_traverse()
        for i in range(len(node_all)):
            n, m = node_all[i], mask_all[i]

            if n.label is not None:  # leaf node
                continue

            if np.sum(m) == 0:
                n.substitute = None
                continue

            left_mask, right_mask = self._get_children_mask(n, self.trX[m, :])
            if np.sum(left_mask)==0:
                deadnode.append(n)
                direction.append(1)
            elif np.sum(right_mask)==0:
                deadnode.append(n)
                direction.append(0)

        for i in range(len(deadnode)):
            n, d = deadnode[i], direction[i]
            if len(n.parents) == 0:
                if d == 0:
                    self.graph = n.left
                    self.graph.parents = []
                    bad_node = n.right
                else:
                    self.graph = n.right
                    self.graph.parents = []
                    bad_node = n.left
                if bad_node.label is None:  # internals
                    bad_node.right.parents.remove(bad_node) if bad_node in bad_node.right.parents else bad_node.right.parents
                    bad_node.left.parents.remove(bad_node) if bad_node in bad_node.left.parents else bad_node.left.parents
                continue

            n.left.parents.remove(n) if n in n.left.parents else n.left.parents
            n.right.parents.remove(n) if n in n.right.parents else n.right.parents
            for p in n.parents:
                if p.left is n and d==0:
                    p.left = n.left
                    n.left.parents.append(p) if p not in n.left.parents else n.left.parents
                if p.right is n and d==0:
                    p.right = n.left
                    n.left.parents.append(p) if p not in n.left.parents else n.left.parents
                if p.left is n and d==1:
                    p.left = n.right
                    n.right.parents.append(p) if p not in n.right.parents else n.right.parents
                if p.right is n and d==1:
                    p.right = n.right
                    n.right.parents.append(p) if p not in n.right.parents else n.right.parents



    def get_params(self, deep=True):
        param = {'N1': self.N1, 'N2':self.N2}
        return {**param, **self.kwargs_backup}

    def set_params(self, **params):
        return self