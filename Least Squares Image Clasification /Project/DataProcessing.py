import numpy as np

def reduceData(XdataInput, threshold=150):
    '''
    Function thresholds data
    ie all values in the np.array() <threshold will be set to zero

    :param XdataInput:
    :param threshold:
    :return:
    '''
    Xdata = XdataInput.copy()
    Xdata = np.where(Xdata > threshold, 1, 0)
    return Xdata

# split data
def splitdata(YdataInput, XdataInput):
    '''
    Sorts data based on labels into a dictionary with the keys being the labels
    :param Ydata:
    :param Xdata:
    :return:
    '''
    # removing any possible pointer issues
    Ydata = YdataInput.copy()
    Xdata = XdataInput.copy()

    # Dictionary to hold sorted images by label
    mdic = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}

    # Iterate over the length of the data and sort each image into the dictionary based
    # on its label
    for i in range(len(Ydata[0])):
        mdic[str(Ydata[0][i])] += [Xdata[i][:]]

    #  Add a row of ones to the bottom of each element in the dictionary
    # ie making them an x tilda vs just x .... 60000x784 -> 60000x785
    for i in mdic:
        mdic[i] = np.append(mdic[i], np.ones((len(mdic[i]), 1)), axis=1)
    return mdic

