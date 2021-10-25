from sklearn.tree import DecisionTreeClassifier
from TreeInTree import TnT
from sklearn.metrics import accuracy_score

# load dataset
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train=x_train.reshape(x_train.shape[0], -1)
x_test=x_test.reshape(x_test.shape[0], -1)

'''
--------------------- performance of TnT ---------------------
this code takes ~5 mins to run. To reduce runtime, use greater ccp_alpha values (1e-3/1e-2).
Console should print the following results:
    TnT accuracy (train set): 0.9591
    TnT accuracy (test set): 0.9037
    TnT model complexity:  1019  internal nodes,  635  leaf nodes
'''
tnt = TnT(N1=2, N2=5, ccp_alpha=1e-4, random_state=0)
tnt.fit(x_train, y_train)
prediction_train = tnt.predict(teX=x_train)
accuracy_train = accuracy_score(y_train, prediction_train)
print("TnT accuracy (train set):", accuracy_train)
prediction_test = tnt.predict(teX=x_test)
accuracy_test = accuracy_score(y_test, prediction_test)
print("TnT accuracy (test set):", accuracy_test)
i, l = tnt.check_complexity()
print("TnT model complexity: ", i, " internal nodes, ", l, " leaf nodes")


'''
--------------------- performance of CART ---------------------
Console should print the following results:
    CART accuracy (train set): 0.9357
    CART accuracy (test set): 0.883
    CART model complexity:  1019  internal nodes,  1020  leaf nodes
'''
cart = DecisionTreeClassifier(max_leaf_nodes=i+1, random_state=0)
cart.fit(x_train, y_train)
prediction_train = cart.predict(x_train)
accuracy_train = accuracy_score(y_train, prediction_train)
print("CART accuracy (train set):", accuracy_train)
prediction_test = cart.predict(x_test)
accuracy_test = accuracy_score(y_test, prediction_test)
print("CART accuracy (test set):", accuracy_test)
print("CART model complexity: ", i, " internal nodes, ", i+1, " leaf nodes")
