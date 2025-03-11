import scipy.io as sio
from scipy.io import savemat
import numpy as np
from One_VS_One import *
from One_VS_All import *
from DataProcessing import *


def problem2(DataPath, OneVone=True, OneVall=True, ProccessData=True, Train=True):
    '''
    Problem 2:
      -One vs One Classifier
      -One vs All Classifier

    :param DataPath: Path to Raw data
    :param OneVone: True if run this classifier
    :param OneVAll: True if run this classifier
    :param ProccessData: if false run with pre processed data
    :param Train: if false just test data
    :return:
    '''
    # ************************************************************************************
    # Load Raw Labeled Matlab Data Sets
    # ************************************************************************************
    if ProccessData:
        mat_contents = sio.loadmat(DataPath)
        sorted(mat_contents.keys())
        testX = mat_contents['testX']  # 10000x784
        testY = mat_contents['testY']  # 1x10000
        trainY = mat_contents['trainY']  # 1x60000
        trainX = mat_contents['trainX']  # 60000x784

        # ********************************************************************************
        # Process Raw Data
        # ********************************************************************************
        mdic = splitdata(trainY, reduceData(trainX))
        mdic['trainX'] = np.append(reduceData(trainX), np.ones((len(reduceData(trainX)), 1)), axis=1)
        mdic['testX'] = np.append(reduceData(testX), np.ones((len(reduceData(testX)), 1)), axis=1)
        mdic['testY'] = testY
        mdic['trainY'] = trainY

        # Store data for later
        savemat("Data/P2 X/P2 processed", mdic)
    # ************************************************************************************
    # Reload Processed Data
    # ************************************************************************************

    mat = sio.loadmat('Data/P2 X/P2 processed')

    # ************************************************************************************
    # Train and Store Data
    # ************************************************************************************
    if OneVone and Train:
        B = one_Vs_oneTrain(mat)
        dic = {}
        dic['b'] = list(B)
        savemat('./Data/P2 theta/B_one', dic)
    if OneVall and Train:
        B = one_Vs_allTrain(mat)
        dic = {}
        dic['b'] = list(B)
        savemat('./Data/P2 theta/B_all', dic)

    # ************************************************************************************
    # Load Trained Data and Test One Vs One
    # ************************************************************************************
    if OneVone:
        B = sio.loadmat('Data/P2 theta/B_one')
        B = np.asarray(B['b'])
        analyse(mat['trainY'], one_Vs_oneTest(B, mat['trainX']), 'Training set one vs one')
        analyse(mat['testY'], one_Vs_oneTest(B, mat['testX']), 'Test set one vs one')
    # ************************************************************************************
    # Load Trained Data and Test One Vs all
    # ************************************************************************************
    if OneVall:
        B = sio.loadmat('Data/P2 theta/B_all')
        B = np.asarray(B['b'])
        analyse(mat['trainY'], one_Vs_allTest(B, mat['trainX']), 'Training set one vs all')
        analyse(mat['testY'], one_Vs_allTest(B, mat['testX']), 'Test set one vs all')
    # ************************************************************************************


if __name__ == '__main__':
    problem2('./Data/mnist.mat',ProccessData=False, Train=False)
    pass
