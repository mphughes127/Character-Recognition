import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.vq import kmeans 

## Adaptive Intelligence - Lab 4: Applying PCA
## Original code written in Matlab by Eleni Vasilaki,
## adapted to Python by Alvin Pastore and Alan Saul.

##Adapted by Matthew Hughes
plt.close('all')

train = np.genfromtxt ('train.csv', delimiter=",")
trainlabels = np.genfromtxt ('trainlabels.csv', delimiter=",")

[n,m]=np.shape(train)  # number of pixels and number of training data

normT = np.sqrt(np.diag(train.T.dot(train)))

train = train / np.matlib.repmat(normT.T,n,1)
data = train.T

# number of Principal Components to save
nPC = 4

PCV = np.zeros((n,nPC))

meanData = np.matlib.repmat(data.mean(axis=0),m,1)
data = data - meanData
C = np.cov(data.T)

# Solve an ordinary or generalized eigenvalue problem
# for a complex Hermitian or real symmetric matrix.
eigen_val, eigen_vec = np.linalg.eigh(C)

# sorting the eigenvalues in descending order
idx = np.argsort(eigen_val)
idx = idx[::-1]
# sorting eigenvectors according to the sorted eigenvalues
eigen_vec = eigen_vec[:,idx]
# sorting eigenvalues
eigen_val = eigen_val[idx]

#plotting eigenvalues
fig = plt.figure()
plt.plot(eigen_val)
plt.xlabel('Principal Component')
plt.ylabel('Eigen Value')
plt.axis([0,10,0,0.1])
plt.show
# save only the most significant eigen vectors
PCV[:,:nPC] = eigen_vec[:,:nPC]
#print PCV.shape
# apply transformation
FinalData = data.dot(PCV)
#print FinalData


# find indexes of data for each digit
zeroData  = (trainlabels==0).nonzero()
#oneData  = (trainlabels==1).nonzero()
twoData   = (trainlabels==2).nonzero()
fourData  = (trainlabels==4).nonzero()
sevenData = (trainlabels==7).nonzero()
eightData = (trainlabels==8).nonzero()


#Get the data for clustering
clustData=[]
clustData.extend(zeroData[0])
clustData.extend(twoData[0])
clustData.extend(fourData[0])
clustData.extend(sevenData[0])
clustData.extend(eightData[0])

newData=[]
for i in xrange(len(clustData)):
    newData.extend(FinalData[clustData[i]])   
    
#get cluster centers using kmeans clustering    
clustArray=np.array(newData).reshape(-1,4) 
clust = kmeans(clustArray,5,iter=20)
clust=clust[0]
print clust


# figure #first second and third PC
fig = plt.figure()
ax = fig.gca(projection = '3d')
# plot zeros
xcomp = FinalData[zeroData,0].flatten()
ycomp = FinalData[zeroData,1].flatten()
zcomp = FinalData[zeroData,2].flatten()
ax.plot(xcomp,ycomp,zcomp,'r.')
# plot twos
xcomp = FinalData[twoData,0].flatten()
ycomp = FinalData[twoData,1].flatten()
zcomp = FinalData[twoData,2].flatten()
ax.plot(xcomp,ycomp,zcomp,'k.')
# plot fours
xcomp = FinalData[fourData,0].flatten()
ycomp = FinalData[fourData,1].flatten()
zcomp = FinalData[fourData,2].flatten()
ax.plot(xcomp,ycomp,zcomp,'m.')
# plot sevens
xcomp = FinalData[sevenData,0].flatten()
ycomp = FinalData[sevenData,1].flatten()
zcomp = FinalData[sevenData,2].flatten()
ax.plot(xcomp,ycomp,zcomp,'y.')
# plot eights
xcomp = FinalData[eightData,0].flatten()
ycomp = FinalData[eightData,1].flatten()
zcomp = FinalData[eightData,2].flatten()
ax.plot(xcomp,ycomp,zcomp,'b.')
#plot clusters
ax.plot(clust[:,0],clust[:,1],clust[:,2], 'co ')
ax.set_title('1st, 2nd and 3rd')
ax.set_xlabel('1 PC')
ax.set_ylabel('2 PC')
ax.set_zlabel('3 PC')
plt.show()


# figure #first second and fourth PC
fig = plt.figure()
ax = fig.gca(projection = '3d')
# plot zeros
xcomp = FinalData[zeroData,0].flatten()
ycomp = FinalData[zeroData,1].flatten()
zcomp = FinalData[zeroData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'r.')
# plot twos
xcomp = FinalData[twoData,0].flatten()
ycomp = FinalData[twoData,1].flatten()
zcomp = FinalData[twoData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'k.')
# plot fours
xcomp = FinalData[fourData,0].flatten()
ycomp = FinalData[fourData,1].flatten()
zcomp = FinalData[fourData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'m.')
# plot sevens
xcomp = FinalData[sevenData,0].flatten()
ycomp = FinalData[sevenData,1].flatten()
zcomp = FinalData[sevenData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'y.')
# plot eights
xcomp = FinalData[eightData,0].flatten()
ycomp = FinalData[eightData,1].flatten()
zcomp = FinalData[eightData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'b.')
#plot clusters
ax.plot(clust[:,0],clust[:,1],clust[:,3], 'co ')
ax.set_title('1st, 2nd and 4th')
ax.set_xlabel('1 PC')
ax.set_ylabel('2 PC')
ax.set_zlabel('4 PC')
plt.show()


# figure #second third and fourth PC
fig = plt.figure()
ax = fig.gca(projection = '3d')
# plot zeros
xcomp = FinalData[zeroData,1].flatten()
ycomp = FinalData[zeroData,2].flatten()
zcomp = FinalData[zeroData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'r.')
# plot twos
xcomp = FinalData[twoData,1].flatten()
ycomp = FinalData[twoData,2].flatten()
zcomp = FinalData[twoData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'k.')
# plot fours
xcomp = FinalData[fourData,1].flatten()
ycomp = FinalData[fourData,2].flatten()
zcomp = FinalData[fourData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'m.')
# plot sevens
xcomp = FinalData[sevenData,1].flatten()
ycomp = FinalData[sevenData,2].flatten()
zcomp = FinalData[sevenData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'y.')
# plot eights
xcomp = FinalData[eightData,1].flatten()
ycomp = FinalData[eightData,2].flatten()
zcomp = FinalData[eightData,3].flatten()
ax.plot(xcomp,ycomp,zcomp,'b.')
#plot clusters
ax.plot(clust[:,1],clust[:,2],clust[:,3], 'co ')
ax.set_title('2nd, 3rd and 4th')
ax.set_xlabel('2 PC')
ax.set_ylabel('3 PC')
ax.set_zlabel('4 PC')
plt.show()

