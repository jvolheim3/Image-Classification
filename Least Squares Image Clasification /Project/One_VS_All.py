import numpy as np
import matplotlib.pyplot as plt

def one_Vs_allTrain(Xdic):
    '''
    Function to train data using one Vs all classifier

    :param Xdic: dictionary of images seperated by labels
    :return: Theta matrix
    '''
    # *******************************************************************************************
    #initializations
    # *******************************************************************************************
    theta = [] # return matrix
    dataLength = 60000 # length of training data
    # *******************************************************************************************
    # iterate through each image label type and run the classifier against all other labels
    # *******************************************************************************************
    for i in range(10):
        # *******************************************************************************************
        # Create a Y matrix of size (dataLength,1) with the top being all positive
        # ones and the rest being negative 1
        # *******************************************************************************************
        temp = np.append(np.ones((len(Xdic[str(i)]))), np.ones((dataLength - len(Xdic[str(i)]))) * (-1))
        yt = np.reshape(temp, (-1, 1)).transpose()
        # *******************************************************************************************
        # Create a matrix with the ith labeled data being on top and all the other data
        # bellow for example( 0 data on top followed by 1->9 ... [[L0],[L1],...,[L9]])
        # *******************************************************************************************
        x = Xdic[str(i)]
        for j in range(10):
            if i != j:
                x = np.append(x, Xdic[str(j)], axis=0)
        # *******************************************************************************************
        # Compute the theta value and add it to the list
        # *******************************************************************************************
        th = np.matmul(np.linalg.pinv(x), yt.transpose())
        theta.append(th)
    # *******************************************************************************************
    # reshaping matrix
    # *******************************************************************************************
    theta = np.asarray(theta)
    c = len(Xdic['0'][0])
    theta = theta.reshape(10, c)
    # *******************************************************************************************
    # Return theta matrix
    # *******************************************************************************************
    return theta

def one_Vs_allTest(theta, Xtest):
    '''
    Function to calculate the predictions on the test data using the trained thetas
    :param theta:
    :param Xtest:
    :return:
    '''
    # Find calculate guess based on trained Thetas
    k = np.matmul(theta, Xtest.transpose())

    # Find colum wise max value index
    prediction = np.argmax(k, axis=0)

    # reshapping prediction to proper size
    prediction = np.asarray(prediction)
    prediction = prediction.reshape(1, len(Xtest))
    return prediction

def analyse(Actual, prediction, title, Graph = True):
    '''
    Compares and analyses image labels vs predictions
    :param Actual:
    :param prediction:
    :param title:
    :return:
    '''
    # creates an array with entries = 1 when guess was wrong and zero otherwise
    new = ((Actual - prediction) != 0) * 1

    # Calculate percent error
    error = new.sum() / len(Actual[0]) * 100
    print(error)
    if Graph:
        # Construct confusion matrix
        cf = np.zeros((10, 10))
        for i in range(len(prediction[0])):
            cf[prediction[0][i]][Actual[0][i]] += 1

        # Display confusion matrix
        np.set_printoptions(suppress=True)
        print(cf)
        plt.imshow(cf, cmap=plt.cm.hot)
        plt.colorbar()
        plt.title(str(title) + ' Error % = ' + str("{:.2f}".format(error)))
        plt.xlabel('Known')
        plt.ylabel('Predicted')
        plt.show()
    return error