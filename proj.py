import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

X_tr = np.array(X_train)
Y_tr = np.array(y_train)
X_te = np.array(X_test)
Y_te = np.array(y_test)

X = np.concatenate((X_tr, X_te), axis=0)
Y = np.concatenate((Y_tr, Y_te), axis=0)
plt.imshow(X[425], cmap='gray', vmin=0, vmax=255)
plt.show()

Xf=np.zeros((70000,784))
Xf_tr=np.zeros((60000,784))
Xf_te=np.zeros((10000,784))

for i in range(0,70000):
    Xf[i]=X[i].flatten()
for i in range(0,60000):
    Xf_tr[i]=X_tr[i].flatten()
for i in range(0,10000):
    Xf_te[i]=X_te[i].flatten()


pca = PCA(n_components=3)
Xf3D = pca.fit_transform(Xf)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xf3D[:,0], Xf3D[:,1], Xf3D[:,2], c = Y)
plt.show()

pca = PCA(n_components=2)
Xf2D = pca.fit_transform(Xf)
plt.scatter(Xf2D[:,0], Xf2D[:,1], c = Y)
plt.show()

labels=KMeans(n_clusters=10,n_init='auto').fit_predict(Xf)
print(accuracy_score(Y,labels))

plt.scatter(Xf2D[:,0], Xf2D[:,1],c=labels)
plt.show()

clas=KNeighborsClassifier(n_neighbors=5)
clas.fit(Xf_tr,Y_tr)
print(clas.score(Xf_te,Y_te))

acc=np.zeros(10)
for i in range(0,10):
    a_tr,a_te,b_tr,b_te = train_test_split(Xf,Y,test_size=0.143)
    clas=KNeighborsClassifier(n_neighbors=5)
    clas.fit(a_tr,b_tr)
    acc[i]=clas.score(a_te,b_te)
print(acc)





