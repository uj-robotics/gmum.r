import numpy as np
import csv
import pylab as pl
import cec

#file = 'data_raw.txt'
file = 'input.txt'
reader = csv.reader(open(file, 'r'), delimiter=',')
x = list(reader)
data = np.matrix(np.array(x).astype('double'))

dataset = cec.DatasetNumpy(data)
size = dataset.size()
Z = np.zeros((size,3))

def plot(Z):
    T = Z[Z[:,2]==0]
    pl.plot(T[:,0], T[:,1], lw=0, marker='o', color='r')
    T = Z[Z[:,2]==1]
    pl.plot(T[:,0], T[:,1], lw=0, marker='o', color='g')
    T = Z[Z[:,2]==2]
    pl.plot(T[:,0], T[:,1], lw=0, marker='o', color='b')
    pl.show()

kmeanspp = cec.Kmeanspp(dataset ,3)
labels = np.zeros((size,1))
kmeanspp.init(labels)
Z[:,:-1] = data
Z[:,2:3] = labels
plot(Z)

d = {'x' : dataset, 'method.type' : 'sphere'}
c = cec.cec(d)
c.x() #check if can be accessed
labels = c.y()
Z[:,:-1] = data

for i in range(0, len(labels)):
    Z[i,-1] = labels[i]

plot(Z)
