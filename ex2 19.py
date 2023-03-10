#TODO: make more feauters
#TODO: adujt labels
#TODO: add comments

#Bar Oren#
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn. model_selection import cross_val_predict
from sklearn import preprocessing
digits = datasets.load_digits()
indices_0_1 = np.where(np.logical_and(digits.target >=0 , digits.target <= 1))

matSumList=[]
centerMatSumList=[]
centerDifList=[]
vertSim=[]
horSim=[]
vertVar=[]
horVar=[]



def digitDataSet():
    _, axes = plt.subplots(nrows=1, ncols=(10), figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(" %i" % label)
    
    
def defualtFeature():
     # flatten the images
     n_samples = len(digits.images[indices_0_1])
     data = digits.images[indices_0_1].reshape((n_samples, -1))
     return data
 
    
def matSum(image):
    return image.sum()
    

def vert_sim(matrix):
    return np.sum(abs(matrix-np.fliplr(matrix)))
    
def sum_center_area(matrix):
    row,col = matrix.shape
    center_sum=np.sum(matrix[row//2-row//4:row//2+row//4,col//2-col//8:col//2+col//8])
    return center_sum

def hor_sim(matrix):
    return np.sum(abs(matrix-np.flipud(matrix)))


def center_Parm_dif(matrix):
     centerSum=sum_center_area(matrix)
        # Determine the dimensions of the matrix
     rows = len(matrix)
     cols = len(matrix[0])
    
     # Use list comprehension to iterate over the elements on the perimeter of the matrix
     perimeter_sum = sum([matrix[row][col] for row in range(rows) for col in range(cols) if row == 0 or row == rows - 1 or col == 0 or col == cols - 1])
    
     # Return the sum
     return abs(perimeter_sum-centerSum)
    
def column_sum_variance(matrix):
   return np.var(np.sum(matrix, axis=0))

def row_sum_variance(matrix):
    return np.var(np.sum(matrix, axis=1))



def classification():
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)
    
    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )
    
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    
    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
    #_, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 3))
    return predicted,X_test,y_test

def mis_classification(predicted,X_test,y_test):
    fig=plt.figure()
    i=1
    for (image, prediction, label) in zip(X_test, predicted, y_test):
            if prediction != label:
                ax= fig.add_subplot(3,10,i)
                ax.set_axis_off()
                image = image.reshape(8, 8)
                ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
                ax.set_title(f" {prediction} {label} ")
                i+=1
    fig.set_facecolor('xkcd:salmon')
    
    fig.suptitle('Test .mis-classification : expecte -predicted')
            
          #  print('has been classified as ', prediction, 'and should be ', lable)

def predic(predicted,X_test,y_test):
    _, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f" {prediction}")


def logisticRegression(featureA,featureB,featureC):
        # creating the X (feature) matrix
    X = np.column_stack((featureA, featureB,featureC))
    # scaling the values for better classification performance
    X_scaled = preprocessing.scale(X)
    # the predicted outputs
    Y = digits.target[indices_0_1]
    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(solver='lbfgs')
    logistic_classifier.fit(X_scaled, Y)
    # show how good is the classifier on the training data
    expected = Y
    predicted = logistic_classifier.predict(X_scaled)
    print("Logistic regression using [featureA, featureB] features:\n%s\n" % (
     metrics.classification_report(
     expected,
     predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    # estimate the generalization performance using cross validation
    predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=10)
    print("Logistic regression using [featureA, featureB] features crossvalidation:\n%s\n"% ( metrics.classification_report( expected,predicted2)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted2))


def twoDGraph(feature1,feature2,label1,label2):
    fig = plt.figure()
    fig.suptitle('feature 1: sum of matrix, for 0,and 1 ', fontsize=14)
    ax = fig.add_subplot()
    ax.scatter(feature1, feature2,c=digits.target[indices_0_1])
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    fig.show()
    
    
def threeDGraph(feature1,feature2,feature3,label1,label2,label3):
    fig = plt.figure()
    fig.suptitle('feature 1: sum of matrix, for 0,and 1 ', fontsize=14)
    ax = fig.gca(projection='3d')
    ax.scatter(feature1, feature2,feature3,c=digits.target[indices_0_1])
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_zlabel(label3)
    fig.show()
    
#q 19 : 
    ###############


predicted,X_test,y_test=classification()
mis_classification(predicted,X_test,y_test)
predic(predicted,X_test,y_test)

for image in digits.images[indices_0_1]:
    matSumList.append(matSum(image))
    vertVar.append(column_sum_variance(image))
    horVar.append(row_sum_variance(image))
    centerMatSumList.append(sum_center_area(image))
    centerDifList.append(center_Parm_dif(image))
    vertSim.append(vert_sim(image))
    horSim.append(hor_sim(image))
    
twoDGraph(matSumList,digits.target[indices_0_1],"sum_of_matrix","digit")
twoDGraph(vertVar,digits.target[indices_0_1],"sum_of_matrix","sum_of_center_matrix")
twoDGraph(horVar,digits.target[indices_0_1],"sum_of_matrix","sum_of_center_matrix")
twoDGraph(centerMatSumList,digits.target[indices_0_1],"sum_of_matrix","sum_of_center_matrix")
twoDGraph(vertSim,digits.target[indices_0_1],"sum_of_matrix","sum_of_center_matrix")
twoDGraph(horSim,digits.target[indices_0_1],"sum_of_matrix","sum_of_center_matrix")
twoDGraph(centerDifList,digits.target[indices_0_1],"sum_of_matrix","sum_of_center_matrix")



threeDGraph(centerMatSumList, vertVar, horVar,"center_sum_of_matrix","vertical var","horozontal var")
threeDGraph(centerMatSumList, vertSim, matSumList,"sum_of_matrix","center_sum_matrix","horozontal var")
threeDGraph(centerMatSumList, centerDifList, matSumList,"sum_of_matrix","center_sum_matrix","horozontal var")

#logisticRegression(matSumList, vertVar)
logisticRegression(centerMatSumList, vertSim, matSumList)


#print("mat",matSumList,"\n",vertVar,"\n",horVar,digits.target[indices_0_1])

#predicted,X_test,y_test=classification(matSumList)
#mis_classification(predicted,X_test,y_test)
#predic(predicted,X_test,y_test)
                    
print("t")

###################