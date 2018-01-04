import numpy as np
import matplotlib.pyplot as plt
import math
def expandData(data,datamax):#it expand data between 0-100
    indexlist=[0,8,27,50,64,77,77,83,83,85]
    for a in range(0,9,2):
        for i in range(indexlist[a],indexlist[a+1]):
            data[:,i]=(data[:,i]/np.max(data[:,i]))*100
        a=a+2
    return data
def clearNans(data):# for some values it can be none, it set that values to 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (math.isnan(data[i][j])):
                data[i][j] = 0
    return data
def returnSortedData(labels,data,filename,save=0):#it is helpfull to see features and effects.It sort data by labels
    datamatris=np.zeros([len(data),85])
    sortedLabels=np.sort(labels)
    i=0
    for value in sortedLabels:
        index=labels.index(value)
        datamatris[i]=data[index]
        i+=1
    if(save==1):
        np.save(filename,datamatris)
    return datamatris
def saveFigures(data,filename,show=0):#it saves figures and shows. show paramater must be 1 to show plot
    for i in range(0, 85):
        plt.plot(data[:, i])
        name = filename + str(i) + ".png"
        if (show==1):
            plt.show()
        plt.savefig(name)
        plt.close()
def convertClassification10(labels):#it returns score to class number
    for i in range(0,len(labels)):
        labels[i]=int(labels[i]/10)+1
    return labels
def convertClassification4(labels):
    for i in range(0, len(labels)):
        if (labels[i] >= 75):
            labels[i] = 1
        elif (labels[i] >= 50):
            labels[i] = 2
        elif (labels[i] >= 25):
            labels[i] = 3
        else:
            labels[i] = 4
    return labels
def convertClassification2(labels):
    for i in range(0, len(labels)):
        if (labels[i] >=50):
            labels[i] = 1
        else:
            labels[i] = 2
    return labels

#READING FILES
trainData=np.load('train31122101.npy')
valData=np.load('val31122101.npy')
testData=np.load('test.npy')
trainlabels=list(np.load('TrainScore31122101.npy'))
valLabels=list(np.load('valscore31122101.npy'))
testLabels=list(np.load('testScore.npy'))

trainData=expandData(trainData,trainData)
np.save("trainSame.npy",clearNans(trainData))
ValData=expandData(valData,trainData)
np.save('valSame.npy',clearNans(ValData))
testData=expandData(testData,trainData)
np.save('testSame.npy',clearNans(testData))

dataTrainsorted=returnSortedData(trainlabels,trainData,"trainDatasorted.npy")
dataValSorted=returnSortedData(valLabels,valData,"ValdataSorted.npy")
datatestSorted=returnSortedData(testLabels,testData,'test')

#saveFigures(dataTrainSorted,"trainsorted")
#saveFigures(dataValSorted,"valSorted")
#saveFigures(datatestSorted,"testSorted")

#Saving expanded data



#Saving classification labels
np.save("trainClass10.npy",convertClassification10(trainlabels.copy()))
np.save("ValClass10.npy",convertClassification10(valLabels.copy()))
np.save("testClass10.npy",convertClassification10(testLabels.copy()))

np.save("trainClass4.npy",convertClassification4(trainlabels.copy()))
np.save("ValClass4.npy",convertClassification4(valLabels.copy()))
np.save("testClass4.npy",convertClassification4(testLabels.copy()))

np.save("trainClass2.npy",convertClassification2(trainlabels.copy()))
np.save("ValClass2.npy",convertClassification2(valLabels.copy()))
np.save("testClass2.npy",convertClassification2(testLabels.copy()))