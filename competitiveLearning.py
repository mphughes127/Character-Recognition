import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
from pylab import pcolor, show, colorbar, xticks, yticks

plt.ion()

## Adaptive Intelligence - Lab 5: Competitive Learning
## Original code written in Matlab by Eleni Vasilaki,
## adapted to Python by Alvin Pastore and Alan Saul.

plt.close('all')

#change input path here if necessary
train = np.genfromtxt ('train.csv', delimiter=",")
trainlabels = np.genfromtxt ('trainlabels.csv', delimiter=",")

[n,m]  = np.shape(train)                    # number of pixels and number of training data
eta    = 0.05                               # learning rate
etaLeak = eta/4                             # Leaky learning rate
winit  = 1                                  # parameter controlling magnitude of initial conditions
leakyThreshold = 1000                       # number of generations without activity until leaky learning activates
lowerBound=0.0375                           # parameter controlling lower bound for neuron activation

tmax   = 40000                              #time/iterations
digits = 15                                 #numer of prototypes

#W = winit * np.random.rand(digits,n)       #randomly initialise weight matrix 
initial = np.ndarray.flatten(train,'F')     #initialise weight matrix with random samples from the input
top = m-digits
ind = np.random.randint(0,top)*784
lis=initial[ind:ind+(digits*n)]
W = lis.reshape(digits,n)                   # Weight matrix (rows = output neurons, cols = input neurons)

normW = np.sqrt(np.diag(W.dot(W.T)))        
normW = normW.reshape(digits,-1)            # reshape normW into a numpy 2d array

#W = W / np.matlib.repmat(normW.T,n,1).T    # normalise using repmat
W = W / normW                               # normalise using numpy broadcasting -  http://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html

#setting up counters
counter = np.zeros((1,digits))              # counter for the winner neurons
wCount = np.ones((1,tmax+1)) * 0.25         # running avg of the weight change over time
alpha = 0.999                               
lastwon=[0]*digits                          # counter for when each neuron last won (used for leaky learning)

yl = int(round(digits/5))                   # counter for the rows of the subplot
if digits % 5 != 0:
    yl += 1

fig_neurs, axes_neurs = plt.subplots(yl,5)  # fig for the output neurons
fig_stats, axes_stats = plt.subplots(6,1)   # fig for the learning stats
fig_corr, axes_corr   = plt.subplots(1,1)   # fig for the correlation matrix


for t in range(1,tmax):

    i = math.ceil(m * np.random.rand())-1   # get a randomly generated index in the input range
    x = train[:,i]                          # pick a training instance using the random index, pick ranodm column
    
    h = W.dot(x)/digits                     # get output firing
    h = h.reshape((h.shape[0],-1))          # reshape h into a numpy 2d array

    xi = np.random.rand(digits,1) / 100     # add noise 
    output = np.max(h+xi)                   # get the max in the output firing vector + noise
    k = np.argmax(h+xi)                     # get the index of the firing neuron
    
    if output<lowerBound:                   # no weight change if below lower bound
        wCount[0,t] = wCount[0,t-1]
        lastwon = [a+1 for a in lastwon]
    else:
        counter[0,k] += 1                       # increment counter for winner neuron
    
        lastwon = [a+1 for a in lastwon]    # increment lastwon counter
        lastwon[k] = 0                      # winning unit won 0 iterations ago
        if max(lastwon)>leakyThreshold:     # if a unit has been dead for threshold generations enable leaky learning
            deadunit=np.argmax(lastwon)
            dw = etaLeak * (x.T - W[deadunit,:]) 
            W[deadunit,:] = W[deadunit,:] + dw  #update weights by reduced amount
        
        
        dw = eta * (x.T - W[k,:])               # calculate the change in weights for the k-th output neuron
                                                # get closer to the input (x - W)

        wCount[0,t] = wCount[0,t-1] * (alpha + dw.dot(dw.T)*(1-alpha)) # % weight change over time (running avg)

        W[k,:] = W[k,:] + dw                    # weights for k-th output are updated
    
    
    eta -= 0.000001                             #reduce learning rate each iteration
    
    
    # draw plots for the first timestep and then every 300 iterations and last ieration
    corr =[]
    if not t % 300 or t == 1 or t==tmax:
        for ii in range(yl):
            for jj in range(5):
                if 5*ii+jj < digits:
                    output_neuron = W[5*ii+jj,:].reshape((28,28),order = 'F')
                    corr.append(np.ndarray.flatten(output_neuron)) # add neurons to correlation matrix
                    axes_neurs[ii,jj].clear()
                    axes_neurs[ii,jj].imshow(output_neuron, interpolation='nearest')
                axes_neurs[ii,jj].get_xaxis().set_ticks([])
                axes_neurs[ii,jj].get_yaxis().set_ticks([])
        plt.draw()
        plt.pause(0.0001)
        
        # Calculate correlation matrix
        corrMatrix= np.corrcoef([corr[0],corr[1],corr[2],corr[3],corr[4],corr[5],corr[6],corr[7],corr[8],corr[9],corr[10],corr[11],corr[12],corr[13],corr[14]])
        axes_corr.clear() 
        axes_corr.set_title("Correlation Matrix")
        axes_corr.pcolormesh(corrMatrix)  # create colorplot of correlation matrix
        mesh = axes_corr.pcolormesh(corrMatrix)
        if t==1:
            fig_corr.colorbar(mesh) # create colorbar
        axes_corr.set_yticks(np.arange(0.5,15.5)) #set ticks midway through square
        axes_corr.set_xticks(np.arange(0.5,15.5))
        axes_corr.set_yticklabels(range(1,16))
        axes_corr.set_xticklabels(range(1,16))
        axes_corr.set_ylim([0,15])
        axes_corr.set_xlim([0,15])
        plt.draw()
        plt.pause(0.0001)
        
        # plot stats
        axes_stats[0].clear()
        axes_stats[0].set_title("Neuron Firing Rates")
        axes_stats[0].bar(np.arange(1,digits+1),h,align='center')
        axes_stats[0].set_xticks(np.arange(1,digits+1))
        axes_stats[0].relim()
        axes_stats[0].autoscale_view(True,True,True)

        axes_stats[1].clear()
        axes_stats[1].set_title("Input Data")
        axes_stats[1].imshow(x.reshape((28,28), order = 'F'), interpolation = 'nearest')
        axes_stats[1].get_xaxis().set_ticks([])
        axes_stats[1].get_yaxis().set_ticks([])

        axes_stats[2].clear()
        axes_stats[2].set_title("Winning Neuron")
        axes_stats[2].imshow(W[k,:].reshape((28,28), order = 'F'), interpolation = 'nearest')
        axes_stats[2].get_xaxis().set_ticks([])
        axes_stats[2].get_yaxis().set_ticks([])

        axes_stats[3].clear()
        axes_stats[3].set_title("Weight change over time (linear axis)")
        axes_stats[3].plot(wCount[0,2:t+1],'-b', linewidth=2.0)
        axes_stats[3].set_ylim([-0.001, 0.255])
        
        axes_stats[4].clear()
        axes_stats[4].set_title("Weight change over time (semilog axes)")
        axes_stats[4].semilogy(wCount[0,2:t+1],'-b', linewidth=2.0)
        
        axes_stats[5].clear()
        axes_stats[5].set_title("Number of times neuron won")
        axes_stats[5].bar(np.arange(1,digits+1),counter.T,align='center')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.0001)

# click anywhere on the stats plot to close both figures
plt.waitforbuttonpress()

