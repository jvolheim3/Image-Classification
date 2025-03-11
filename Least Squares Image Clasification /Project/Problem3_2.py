import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import savemat
import numpy as np
from One_VS_One import *
from One_VS_All import *
from DataProcessing import *
from TransformationFunctions import *

if __name__ == '__main__':
    '''
    # ************************************************************************************
    # Problem 3.2:
    #   - Observe change in error rate vs L
    #
    # ************************************************************************************
    # Load Raw Labeled Matlab Data Sets
    # ************************************************************************************
    mat_contents = sio.loadmat('Data/mnist.mat')
    testY = mat_contents['testY']  # 1x10000
    trainY = mat_contents['trainY']  # 1x60000
    testX = mat_contents['testX']  # 10000x784
    trainX = mat_contents['trainX']  # 60000x784
    # ************************************************************************************
    # Process Data
    # ************************************************************************************
    testX = reduceData(testX.copy())
    trainX = reduceData(trainX.copy())
    # ************************************************************************************
    # Dictionary to hold Error vs L values
    # ************************************************************************************
    dic = {'G1O': [], 'G2O': [], 'G3O': [], 'G4O': [],
           'G1A': [], 'G2A': [], 'G3A': [], 'G4A': []}
    L_list = [5,50,100,250,500,1000,1500]
    # ************************************************************************************
    # Train and Test One v One and One vs All while looping through list of L values to check
    # error rate
    # ************************************************************************************

    for L in L_list:
        print('*****'+str(L)+'*****')
        # ********************************************************************************
        # Obtain Random weights
        # ********************************************************************************
        w = np.random.normal(0, 1, (784, L))
        b = np.random.normal(0, 1, (1, L))
        # ********************************************************************************
        # Construct H(X) = [[g(x1)],[g(x2)],...,[g(xL)]] for all g(x)
        # ********************************************************************************
        Xtest = np.matmul(testX.copy(),w) + b
        Xtrain = np.matmul(trainX.copy(), w) + b
        h1 = Identity(Xtrain.copy())
        h1t = Identity(Xtest.copy())
        h2 = Sigmoid(Xtrain.copy())
        h2t = Sigmoid(Xtest.copy())
        h3 = Sinusoidal(Xtrain.copy())
        h3t = Sinusoidal(Xtest.copy())
        h4 = ReLu(Xtrain.copy())
        h4t = ReLu(Xtest.copy())
        # ********************************************************************************
        # Create X tilda = [[x],[1]]
        # ********************************************************************************
        X1t = np.append(h1t, np.ones((len(h1t), 1)), axis=1)
        X2t = np.append(h2t, np.ones((len(h2t), 1)), axis=1)
        X3t = np.append(h3t, np.ones((len(h3t), 1)), axis=1)
        X4t = np.append(h4t, np.ones((len(h4t), 1)), axis=1)
        # ********************************************************************************
        # Train One Vs One and One Vs All for each g(x)
        # ********************************************************************************
        thetaO1 = one_Vs_oneTrain(splitdata(trainY,h1.copy()))
        thetaO2 = one_Vs_oneTrain(splitdata(trainY, h2.copy()))
        thetaO3 = one_Vs_oneTrain(splitdata(trainY, h3.copy()))
        thetaO4 = one_Vs_oneTrain(splitdata(trainY, h4.copy()))

        thetaA1 = one_Vs_allTrain(splitdata(trainY, h1.copy()))
        thetaA2 = one_Vs_allTrain(splitdata(trainY, h2.copy()))
        thetaA3 = one_Vs_allTrain(splitdata(trainY, h3.copy()))
        thetaA4 = one_Vs_allTrain(splitdata(trainY, h4.copy()))
        # ********************************************************************************
        # Test and Analyse trained thetas
        # ********************************************************************************

        dic['G1O'].append(analyse(testY.copy(), one_Vs_oneTest(thetaO1, X1t), 'One Vs One\n L ='+str(L),False))
        dic['G2O'].append(analyse(testY.copy(), one_Vs_oneTest(thetaO2, X2t), 'One Vs One\n L =' + str(L),False))
        dic['G3O'].append(analyse(testY.copy(), one_Vs_oneTest(thetaO3, X3t), 'One Vs One\n L =' + str(L),False))
        dic['G4O'].append(analyse(testY.copy(), one_Vs_oneTest(thetaO4, X4t), 'One Vs One\n L =' + str(L),False))
        dic['G1A'].append(analyse(testY.copy(), one_Vs_allTest(thetaA1, X1t), 'One Vs All\n L =' + str(L),False))
        dic['G2A'].append(analyse(testY.copy(), one_Vs_allTest(thetaA2, X2t), 'One Vs All\n L =' + str(L),False))
        dic['G3A'].append(analyse(testY.copy(), one_Vs_allTest(thetaA3, X3t), 'One Vs All\n L =' + str(L),False))
        dic['G4A'].append(analyse(testY.copy(), one_Vs_allTest(thetaA4, X4t), 'One Vs All\n L =' + str(L),False))
    # ************************************************************************************
    # Store Data
    # ************************************************************************************
    savemat('Data/P3_2 Error vs L/Error_vs_L', dic)
    print(dic)
    # ************************************************************************************
    # Display Data
    # ************************************************************************************
    '''
    L_list = [5, 50, 100, 250, 500, 1000, 1500]
    # Plot One Vs One
    dic = sio.loadmat('Data/P3_2 Error vs L/Error_vs_L')
    plt.plot(L_list,dic['G1O'].reshape(7),label='Identity')
    plt.plot(L_list, dic['G2O'].reshape(7), label='Sigmoid')
    plt.plot(L_list, dic['G3O'].reshape(7), label='Sinusoidal')
    plt.plot(L_list, dic['G4O'].reshape(7), label='Relu')
    plt.ylabel('Error')
    plt.xlabel('L')
    plt.title('One vs One\n Error vs L')
    plt.legend(loc="upper right")
    plt.show()

    # Plot One Vs All
    plt.plot(L_list, dic['G1A'].reshape(7), label='Identity')
    plt.plot(L_list, dic['G2A'].reshape(7), label='Sigmoid')
    plt.plot(L_list, dic['G3A'].reshape(7), label='Sinusoidal')
    plt.plot(L_list, dic['G4A'].reshape(7), label='Relu')
    plt.ylabel('Error')
    plt.xlabel('L')
    plt.title('One Vs all\n Error vs L')
    plt.legend(loc="upper right")
    plt.show()
    # ************************************************************************************



