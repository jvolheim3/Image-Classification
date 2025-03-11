import scipy.io as sio
from scipy.io import savemat
import numpy as np
from One_VS_One import *
from One_VS_All import *
from DataProcessing import *
from TransformationFunctions import *


def problem3_1(DataPath, ProccessData=True, OneVone=True, OneVall=True, Train=True):
    '''
     Problem 3.1:
      - Randomized Feature Based Least Squares Classifiers
          - Identity Function
          - Sigmoid Function
          - Sinusoidal Function
          - Rectified Linear Unit Function
      - One Vs One Classifier
      - One Vs All Classifier

    :param DataPath:
    :param ProccessData:
    :param OneVone:
    :param OneVall:
    :param Train:
    :return:
    '''
    # ************************************************************************************

    # ************************************************************************************
    # Load Raw Labeled Matlab Data Sets
    # ************************************************************************************
    mat_contents = sio.loadmat(DataPath)
    testY = mat_contents['testY']  # 1x10000
    trainY = mat_contents['trainY']  # 1x60000
    testX = mat_contents['testX']  # 10000x784
    trainX = mat_contents['trainX']  # 60000x784


    if ProccessData:
        # ************************************************************************************
        # Process Data
        # ************************************************************************************
        testX = reduceData(testX.copy())
        trainX = reduceData(trainX.copy())
        # ************************************************************************************
        # Obtain Random weights and Store for later Use
        # ************************************************************************************
        L = 1000
        w = np.random.normal(0, 1, (784, L))
        b = np.random.normal(0, 1, (1, L))
        # ************************************************************************************
        # Construct h1,h2,h3,h4 = [[g(x1)],[g(x2)],...,[g(xL)]]
        # ************************************************************************************
        Xtest = np.matmul(testX, w) + b
        Xtrain = np.matmul(trainX, w) + b
        h1 = Identity(Xtrain)
        h2 = Sigmoid(Xtrain)
        h3 = Sinusoidal(Xtrain)
        h4 = ReLu(Xtrain)
        # ************************************************************************************
        # Storing Processed Data
        # ************************************************************************************
        savemat('Data/P3_1 gx data/train1', splitdata(trainY, h1))
        savemat('Data/P3_1 gx data/train2', splitdata(trainY, h2))
        savemat('Data/P3_1 gx data/train3', splitdata(trainY, h3))
        savemat('Data/P3_1 gx data/train4', splitdata(trainY, h4))

        htest = {'G1_Training_Data': list(np.append(h1, np.ones((len(h1), 1)), axis=1)),
                 'G1_Test_Data': list(np.append(Identity(Xtest), np.ones((len(Identity(Xtest)), 1)), axis=1)),
                 'G2_Training_Data': list(np.append(h2, np.ones((len(h2), 1)), axis=1)),
                 'G2_Test_Data': list(np.append(Sigmoid(Xtest), np.ones((len(Sigmoid(Xtest)), 1)), axis=1)),
                 'G3_Training_Data': list(np.append(h3, np.ones((len(h3), 1)), axis=1)),
                 'G3_Test_Data': list(np.append(Sinusoidal(Xtest), np.ones((len(Sinusoidal(Xtest)), 1)), axis=1)),
                 'G4_Training_Data': list(np.append(h4, np.ones((len(h4), 1)), axis=1)),
                 'G4_Test_Data': list(np.append(ReLu(Xtest), np.ones((len(ReLu(Xtest)), 1)), axis=1))
                 }
        savemat('./Data/P3_1 gx data/testdata', htest)
    # ************************************************************************************
    # Data Storage Paths
    # ************************************************************************************
    trainfiles = ['Data/P3_1 gx data/train1',
                  'Data/P3_1 gx data/train2',
                  'Data/P3_1 gx data/train3',
                  'Data/P3_1 gx data/train4']
    thetafiles = ['Data/P3_1 theta/theta1',
                  'Data/P3_1 theta/theta2',
                  'Data/P3_1 theta/theta3',
                  'Data/P3_1 theta/theta4']
    # ************************************************************************************
    # Train Data on One Vs One and One Vs All for every featured data set
    # ************************************************************************************
    f = 0
    if Train:
        for i in trainfiles:
            dic = {}
            mat = sio.loadmat(i)
            if OneVone:
                one = one_Vs_oneTrain(mat)
                dic['one'] = list(one)
            if OneVall:
                all = one_Vs_allTrain(mat)
                dic['all'] = list(all)

            # Saving thetas
            if OneVone or OneVall:
                savemat(thetafiles[f], dic)
            f += 1
    # ************************************************************************************
    # Load Test Data
    # ************************************************************************************
    htest = sio.loadmat('Data/P3_1 gx data/testdata')
    testDataKeys = list(htest.keys())
    # ************************************************************************************
    #  Test Trained Functions
    # ************************************************************************************
    f = 3
    for i in thetafiles:
        # load the theta values
        Theta = sio.loadmat(i)
        # ********************************************************************************
        # Test Classifiers on Training Data
        # ********************************************************************************
        X = np.asarray(htest[testDataKeys[f]])
        if OneVall:
            analyse(trainY, one_Vs_allTest(np.asarray(Theta['all']), X), 'One Vs all\n' + testDataKeys[f])
        if OneVone:
            analyse(trainY, one_Vs_oneTest(np.asarray(Theta['one']), X), 'One V one\n' + testDataKeys[f])
        # ********************************************************************************
        # Test Classifiers on Test Data
        # ********************************************************************************
        f += 1
        X = np.asarray(htest[testDataKeys[f]])
        if OneVall:
            analyse(testY, one_Vs_allTest(np.asarray(Theta['all']), X), 'One Vs all\n' + testDataKeys[f])
        if OneVone:
            analyse(testY, one_Vs_oneTest(np.asarray(Theta['one']), X), 'One Vs one\n' + testDataKeys[f])
        f += 1
        # ********************************************************************************


if __name__ == '__main__':
    problem3_1('./Data/mnist.mat', ProccessData=False,Train=False)
