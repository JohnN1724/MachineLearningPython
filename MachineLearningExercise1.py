from numpy.compat.py3k import npy_load_module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Question 1(a)

dataframe = pd.read_csv('data_banknote_authentication.csv',header = None)
npdata = dataframe.to_numpy()
Data = npdata[:,:-1] # all columns except the last
Labels = npdata[:,-1] # last column

#array = np.split(npdata, [1,5])

Data = npdata[:,:-1] # all columns except the last
Labels = npdata[:,-1] # last column

c = np.max(Labels).astype(int) # number of classes
print('Number of classes c: ',c+1)

print('Shape of the array: ', npdata.shape)
print('This shape tells us that theres 1372 objects in the dataset (N) and 5 features(n)')

for i in range(len(Labels)):
    Labels[i] += 1

# Question 1(b)
num = 1
for x in range(4):
    for y in range(4):
        plt.subplot(4,4,num)
        num+=1
        plt.scatter(Data[:,x],Data[:,y],c= Labels, s=0.5)
        plt.axis('off')
        title = str((x+1)) + " vs " + str((y+1))
        plt.title(title)

# Question 2(a)

def nearest_mean_classifier(training_data,training_labels, \
        testing_data,testing_labels):
    # Class labels are 1, 2, ..., c
    c = np.max(training_labels).astype('int') # number of classes
    n = len(training_data[0]) # number of features
    m = np.empty((c,n)) # array to hold the means
    for i in range(c):
        m[i] = np.mean(training_data[training_labels == i+1],axis = 0)
    #print(m)
    assigned_labels = np.empty(len(testing_labels))
    for i in range(len(testing_labels)):
        x = testing_data[i] # object i from the testing data
        di = np.sum((m-x)**2,axis = 1) # distances to means
        assigned_labels[i] = np.argmin(di)+1
    e = np.mean(testing_labels != assigned_labels)
    return e, assigned_labels


# Question 2(b)

D = dataframe.to_numpy()

np.random.shuffle(D)
split = (len(D))/2
D1 = D[:int(split)] # training data
D2 = D[int(split):] # testing data
D1_Data = D1[:,:4]
D1_Labels = D1[:,4]
D2_Data = D2[:,:4]
D2_Labels = D2[:,4]

resub_error, al = nearest_mean_classifier(Data,Labels,Data,Labels)
print('Resubstitution error NMC = ',resub_error)

# Question 3

resub_error, al = nearest_mean_classifier(Data,Labels,Data,Labels)
fp = 0
tp = 0
fn = 0
tn = 0

for i in range(len(Labels)):
    if al[i] == 1:
        if Labels[i] == 1:
            tn += 1
        if Labels[i] == 2:
            fn += 1
    if al[i] == 2:
        if Labels[i] == 1:
            fp += 1
        if Labels[i] == 2:
            tp += 1

print("         Positive           Negative")
print("Positive  ",tp,"               ",fn)
print("Negative  ",fp,"               ",tn)

sens = tp/(tp+fn)
spec = tn/(tn+fp)
recall = sens
precision = tp/(tp+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
f1 = 2*((precision*recall)/(precision+recall))

print("Sensitivity ", sens)
print("Specificity ", spec)
print("Recall ", recall)
print("Precision ", precision)
print("Accuracy ", accuracy)
print("F1 Measure ", f1)
print("Operational Point ", (1-spec))


# Question 4(a)

class1 = 0
class2 = 0

zeroR = 0

for i in range(len(Labels)):
    if Labels[i] == 1:
        class1+=1
    if Labels[i] == 2:
        class2+=1
if class1 > class2:
    zeroR = class1/len(Labels)
else:
    zeroR = class2/len(Labels)
print("zeroR error rate = ",(1-zeroR))

# Question 4(b)

col1 = npdata[:,0]
col4 = npdata[:,3]
plt.figure()

plt.plot(col1[Labels == 1],col4[Labels == 1],'go', ms=3)
plt.plot(col1[Labels == 2],col4[Labels == 2],'rx', ms=6)

plt.title("Scatterplot Of Data 2")
plt.xlabel("Feature #1")
plt.ylabel("Feature #4")
plt.axis("Equal")
plt.grid(True)
plt.legend(("Genuine", "Fradulent"), loc="upper left")

# Question 4(c)

classifierLabels = np.empty(len(dataframe))

fp = 0
tp = 0
fn = 0
tn = 0

for i in range(len(dataframe)):
    if(col1[i]) > 0:
        classifierLabels[i] = 1
    if(col1[i]) < 0:
        classifierLabels[i] = 2
    if classifierLabels[i] == 1:
        if Labels[i] == 1:
            tn += 1
        if Labels[i] == 2:
            fn += 1
    if classifierLabels[i] == 2:
        if Labels[i] == 1:
            fp += 1
        if Labels[i] == 2:
            tp += 1

print("         Positive           Negative")
print("Positive  ",tp,"               ",fn)
print("Negative  ",fp,"               ",tn)
error_rate = np.mean(Labels != classifierLabels)
print("Error rate of R: ", error_rate)

# Question 5(a)

from sklearn.neighbors import KNeighborsClassifier
one_nn = KNeighborsClassifier(n_neighbors = 1)

Data2 = np.empty(shape=[0,2])

for i in range(len(Data)):
    row = np.array([col1[i], col4[i]])
    Data2 = np.append(Data2,[row],axis=0)
trd, tsd, trl, tsl = train_test_split(Data2, Labels, test_size = 0.5)
one_nn.fit(trd,trl)
plt.figure()

griddatax, griddatay = np.array(np.meshgrid(np.linspace(-7.5,7.5,200), np.linspace(-8,3,200)))

griddata = np.hstack((np.reshape(griddatax,(-1,1)), np.reshape(griddatay,(-1,1))))

grid_labels = one_nn.predict(griddata)

print('1nn error = %.4f' % error_rate)

plt.scatter(griddata[:,0],griddata[:,1], c = grid_labels)

plt.plot(griddata[grid_labels == 1,0],griddata[grid_labels == 1,1],'go', ms=3)
plt.plot(griddata[grid_labels == 2,0],griddata[grid_labels == 2,1],'rx', ms=6)

plt.title("Scatterplot Of Data 2")
plt.xlabel("Feature #1")
plt.ylabel("Feature #4")
plt.axis("Equal")
plt.grid(True)
plt.legend(("Genuine", "Fradulent"), loc="upper left")

# Question 5(b)

trd, tsd, trl, tsl = train_test_split(Data, Labels, test_size = 0.5)
one_nn.fit(trd,trl)
assigned_labels = one_nn.predict(tsd)
error_rate = np.mean(tsl != assigned_labels)

print("1nn error rate of Data from hold out protocol: ", error_rate)

trd, tsd, trl, tsl = train_test_split(Data2, Labels, test_size = 0.5)
one_nn.fit(trd,trl)
assigned_labels = one_nn.predict(tsd)
error_rate = np.mean(tsl != assigned_labels)
print("1nn Error rate of Data2 from hold out protocol: ", error_rate)

# Question 6(a)

x = x = np.linspace(-5, 5, 100)
y = ((5/4)*x)-3

plt.figure()
plt.plot(col1[Labels == 1],col4[Labels == 1],'go', ms=3)
plt.plot(col1[Labels == 2],col4[Labels == 2],'rx', ms=6)
plt.plot(x,y)
plt.title("Scatterplot of Data 2")
plt.xlabel("Feature #1")
plt.ylabel("Feature #4")
plt.axis("Equal")
plt.grid(True)
plt.legend(("Genuine", "Fradulent"), loc="upper left")

# Question 6(b)

classifierLabels = np.empty(len(dataframe))
for i in range(len(dataframe)):
    x = col1[i]
    y = col4[i]
    
    if ((10*x)-(8*y)-24) > 0:
        classifierLabels[i] = 1
    if ((10*x)-(8*y)-24) < 0:
        classifierLabels[i] = 2

error_rate = np.mean(Labels != classifierLabels)
print("Error rate of Linear Classifier: ", error_rate)

plt.show()