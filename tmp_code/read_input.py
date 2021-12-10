import gzip
import numpy as np

#https://archive.ics.uci.edu/ml/datasets/covertype
#http://yann.lecun.com/exdb/mnist/

image_size = 28
train_data_count = 60_000
test_data_count = 10_000

def readGzImages(path,number_of_images):
    input_X = gzip.open(path,'r')
    input_X.read(16)
    buf_X = input_X.read(image_size * image_size * number_of_images)
    X = np.frombuffer(buf_X, dtype=np.uint8).astype(np.float32)
    X = X.reshape(number_of_images ,image_size * image_size)
    return X

def readGzLabels(path, number_of_labels):
    input_y = gzip.open(path,'r')
    input_y.read(8)
    buf_y = input_y.read(1 * number_of_labels)
    y = np.frombuffer(buf_y, dtype=np.uint8).astype(np.int32)
    y = y.reshape(number_of_labels)
    return y

X = readGzImages('./MNIST_set/train-images-idx3-ubyte.gz', train_data_count)
y = readGzLabels('./MNIST_set/train-labels-idx1-ubyte.gz', train_data_count)

X_test = readGzImages('./MNIST_set/t10k-images-idx3-ubyte.gz', test_data_count)
y_test = readGzLabels('./MNIST_set/t10k-labels-idx1-ubyte.gz', test_data_count)


#prikaz primjera
# import matplotlib.pyplot as plt
# image = np.asarray(X[6]).squeeze()
# print(y[0])
# plt.imshow(image)
# plt.show()

tree_depth_size = [10,15,25]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss


# errors = []
# for i in range(10):
#     clf = RandomForestClassifier(
#         n_estimators=100, max_features="sqrt", max_depth=tree_depth_size[2],min_samples_split=5
#         )
#     clf.fit(X,y)
#     y_predict = clf.predict(X_test)
#     error_rate = zero_one_loss(y_test, y_predict)
#     errors.append(error_rate)

# print("mean: " + str(np.mean(errors)) + ", std: "+ str(np.std(errors)))


# clf = RandomForestClassifier(
#          n_estimators=100, max_features="sqrt", max_depth=tree_depth_size[2],min_samples_split=5
#          )
# clf.fit(X,y)
# a = clf.apply(X_test)
# b = clf.decision_path(X_test)
# c = clf.predict_log_proba(X_test)
# d = clf.predict_proba(X_test)




from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# clf_one = DecisionTreeClassifier(
#         max_features="sqrt", max_depth=tree_depth_size[2],min_samples_split=5
#         )

# y = [
#     [0.9,0.05,0.05],
#     [0.03,0.03,0.94],
#     [0.01,0.06,0.93]]

# pi = [[1,0,0],[0,0,1],[0,0,1]]

# y3 = np.array([y,y,y])

# pi3 = np.array([pi,pi,pi])

# e = y3 @ pi3

# i = 3



clf = RandomForestClassifier(
         n_estimators=100, max_features="sqrt", max_depth=tree_depth_size[2],min_samples_split=5
         )
def indexing(index, size):
    arr = np.zeros(size, dtype = np.int8)
    arr[index] = 1
    return arr

clf.fit(X,y)
forest = clf.estimators_
FI_X = np.array([f.predict(X) for f in forest]).T
#FI_X = np.array([[indexing(a.astype(int),10) for a in b] for b in FI_X])
FI_X1 = np.array([[indexing(a.astype(int),10) for a in b] for b in FI_X]).reshape(60000,1000)

from liblinear.liblinearutil import *
best_C, best_p, best_rate = train(y, FI_X1, '-C -s 0')



print(y)

from liblinear.liblinearutil import *
y_indexing = np.array([indexing(i,10) for i in y])
prob  = problem(y, FI_X)
param = parameter('-s 2 -c 1')
m = train(prob, param)
[W, b] = m.get_decfun()
W_R = np.array(W).reshape(10,100)
print(W)

def predict_y(W,X):
    a = np.array([f.predict(X) for f in forest]).T




