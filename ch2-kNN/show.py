#encoding=utf-8
# 加上才能中文注释


import kNN
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat, datingLabels = kNN.file2matrix("datingTestSet2.txt")
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()