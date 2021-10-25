import numpy as np
import copy
from sklearn.tree import DecisionTreeClassifier

class Node(object):
    def __init__(self, feature_index=None, threshold=None, label=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.label = label
        self.left = None
        self.right = None
        self.parents = []
        self.substitute = None  #sklearn tree type

    def in_place_substitute(self):
        if self.substitute is not None:
            node = self.convert_type()
            self.leaf_replace(node) # for internals only
            self.root_replace(node)

    def convert_type(self):
        tree = self.substitute.tree_
        classes = self.substitute.classes_
        node_list = []
        for i in range(tree.capacity):
            if tree.feature[i] == -2:
                node_list.append(Node(label=classes[np.argmax(tree.value[i, 0, :])]))
            else:
                node_list.append(Node(tree.feature[i], tree.threshold[i]))
        for i in range(tree.capacity):
            if tree.children_left[i] != -1:
                node_list[i].left = node_list[tree.children_left[i]]
                node_list[tree.children_left[i]].parents.append(node_list[i]) if node_list[i] not in node_list[tree.children_left[i]].parents else node_list[tree.children_left[i]].parents
            if tree.children_right[i] != -1:
                node_list[i].right = node_list[tree.children_right[i]]
                node_list[tree.children_right[i]].parents.append(node_list[i]) if node_list[i] not in node_list[tree.children_right[i]].parents else node_list[tree.children_right[i]].parents
        return node_list[0]

    def root_replace(self,node):
        self.feature_index = node.feature_index
        self.threshold = node.threshold
        self.label = node.label
        self.left = node.left
        self.right = node.right
        self.substitute = node.substitute
        if node.left is not None and node.right is not None:
            node.left.parents.remove(node) if node in node.left.parents else node.left.parents
            node.left.parents.append(self) if self not in node.left.parents else node.left.parents
            node.right.parents.remove(node) if node in node.right.parents else node.right.parents
            node.right.parents.append(self) if self not in node.right.parents else node.right.parents

    def leaf_replace(self, node):
        if self.label is not None: # leaves
            return
        left, right = self.left, self.right
        left.parents.remove(self) if self in left.parents else left.parents
        right.parents.remove(self) if self in right.parents else right.parents
        if node.label is None:
            internal = [node]
        else:
            internal = []
        while len(internal) > 0:
            l = internal.pop(0)
            if l.left.label is not None: # leaf
                if l.left.label == 0:
                    l.left = left
                    left.parents.append(l) if l not in left.parents else left.parents
                elif l.left.label == 1:
                    l.left = right
                    right.parents.append(l) if l not in right.parents else right.parents
            else:
                internal.append(l.left)

            if l.right.label is not None: # leaf
                if l.right.label == 0:
                    l.right = left
                    left.parents.append(l) if l not in left.parents else left.parents
                elif l.right.label == 1:
                    l.right = right
                    right.parents.append(l) if l not in right.parents else right.parents
            else:
                internal.append(l.right)

    def update_substitute(self, trX, trY, sample_weight, **kwargs):  #best first tree
        if len(trY)==0:
            self.substitute = None
            return
        if 'ccp_alpha' in kwargs:
            kwargs['ccp_alpha'] = kwargs['ccp_alpha'] / len(trY)
        model = DecisionTreeClassifier(**kwargs)
        model.fit(trX, trY, sample_weight)
        self.substitute = model


class TnT:
    def __init__(self, N1, N2, graph=None,**kwargs):
        self.N1 = N1
        self.N2 = N2
        self.kwargs_backup = kwargs
        if graph is None:
            self.graph=self._init_graph()
        else:
            self.graph = copy.deepcopy(graph)

    def _init_graph(self):
        root = Node(label=0)
        return root

    def _get_children_mask(self, n, trX):  # n has to be an internal node
        if n.label is not None:
            print("bug reported: access children of leaf nodes.")
        if trX.shape[0] == 0:
            return np.array([]), np.array([])
        if n.substitute is None:
            pred = np.zeros(trX.shape[0])
            pred[trX[:, n.feature_index] > n.threshold] = 1
        else:
            pred = n.substitute.predict(trX)
        return (pred == 0), (pred == 1)

    def _get_mask(self, n, node_all, mask_all, X=None):
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

    def _fit(self):
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
        self.trX = trX
        self.trY = trY
        if sample_weight is None:
            self.sample_weight = np.ones_like(trY)/len(trY)
        else:
            self.sample_weight = sample_weight
        self.classes_ = np.unique(trY)
        self.kwargs = self.kwargs_backup
        if 'ccp_alpha' in self.kwargs:
            self.kwargs['ccp_alpha'] = self.kwargs['ccp_alpha'] * len(trY)
        node_all, mask_all = [], []
        for i in range(self.N1):
            for _ in range(self.N2):
                node_all, mask_all=self._fit()
            self.prune()
            self.easy_substitute()
        return node_all, mask_all

    def graph_traverse(self, teX=None, node=None):
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

    def predict(self, teX=None, node=None):
        if node is None:
            node = self.graph
        if teX is None:
            teX = self.trX
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

    def check_complexity(self):
        node_all,  mask_all= self.graph_traverse()
        num_internal, num_leaf = 0, 0
        n_samples = 0
        for i in range(len(node_all)):
            n, m = node_all[i], mask_all[i]

            if np.sum(m) == 0:
                print("useless node detected", n, n.substitute, n.label)
            if n.substitute is not None:
                print("substitute not done.")

            if n.label is None:
                num_internal += 1
            else:
                num_leaf += 1
                n_samples += np.sum(m)
        return num_internal, num_leaf

    def easy_substitute(self):
        node_all, mask_all = self.graph_traverse()
        for i in range(len(node_all)):
            n, m = node_all[i], mask_all[i]
            n.in_place_substitute()

    def prune(self):
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