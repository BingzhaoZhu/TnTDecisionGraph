import numpy as np
from sklearn.tree import DecisionTreeClassifier

class Node(object):
    """
        A class used to represent internal/leaf node in TnT graph
        ----------
        feature_index : int
            which attribute to split on (internals)
        threshold : float
            threshold of attribute (internals)
        label : int
            class label (leaves)
    """
    def __init__(self, feature_index=None, threshold=None, label=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.label = label
        self.left = None
        self.right = None
        self.parents = []
        self.substitute = None  #sklearn tree type

    def in_place_substitute(self):
        """
            merge the micro decision tree into the graph
        """
        if self.substitute is not None:
            node = self.convert_type()
            self.leaf_replace(node) # for internals only
            self.root_replace(node)

    def convert_type(self):
        """
            convert a sklearn DecisionTreeClassifier to a tree of Node
        """
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
        """
            replace the Node with the root of micro tree
        """
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
        """
            connect micro tree to the child
        """
        if self.label is not None: # return if leaf node
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
        """
            update the micro tree
        """
        if len(trY)==0 or np.sum(sample_weight)==0:
            self.substitute = None
            return
        if 'ccp_alpha' in kwargs:
            kwargs['ccp_alpha'] = kwargs['ccp_alpha'] / np.sum(sample_weight)
        model = DecisionTreeClassifier(**kwargs)
        model.fit(trX, trY, sample_weight)
        self.substitute = model
