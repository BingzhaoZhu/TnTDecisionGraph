import numpy as np
from scipy.sparse import csr_matrix

def visTnT(model, X, y, **kwargs):
    from sknetwork.visualization import svg_digraph
    adjacency, class_label = get_adjacency(model, X, y)
    image = svg_digraph(adjacency, labels=class_label, **kwargs)
    return image
    

def get_adjacency(model, X, y):
    node_all, mask_all = model.graph_traverse(X)
    adjacency = np.zeros((len(node_all), len(node_all)))
    position = np.zeros((len(node_all), 2))
    class_label = []
    for i in range(len(node_all)):
        n, m = node_all[i], mask_all[i]
        if n.left in node_all:
            ind = node_all.index(n.left)
            adjacency[i,ind]=1
        if n.right in node_all:
            ind = node_all.index(n.right)
            adjacency[i,ind]=1
        position[i, 0] = np.random.rand()#np.sin(i/len(node_all)*2*np.pi)*10 #np.floor(np.log2(i+1))
        position[i, 1] = np.random.rand()#np.cos(i/len(node_all)*2*np.pi)*10 #(i - 3*2**(position[i, 0]-1))*100
        labels = y[m]
        values, counts = np.unique(labels, return_counts=True)
        if np.max(counts)>len(labels)/2:
            class_label.append(values[np.argmax(counts)])
        else:
            class_label.append(10)
    return csr_matrix(adjacency), class_label