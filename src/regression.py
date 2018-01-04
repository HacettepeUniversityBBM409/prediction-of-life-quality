import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import math

def returnSortedData(labels,data):
    datamatris=[]
    sortedLabels=np.sort(labels)
    print(type(labels),type(data))
    print("labels:",labels)
    for value in sortedLabels:
        index=np.where(labels==value)
        print(index,labels[index],data[index])
        datamatris.append(data[index])
    return datamatris

def dtr():
    #file = open("result_dtr.txt", 'w')
    train = np.load("trainsame31122101.npy") / 100
    scores = np.load("trainscore31122101.npy")
    validation = np.load("valsame31122101.npy") / 100
    validation_scores = np.load("valscore31122101.npy")
    print("train NaNs")
    for i in range(train.shape[0]):
        for j in range(train.shape[1]):
            if (math.isnan(train[i][j])):
                print(i, j)
                train[i][j] = 0.0
    print("validation NaNs")
    for i in range(validation.shape[0]):
        for j in range(validation.shape[1]):
            if (math.isnan(validation[i][j])):
                print(i, j)
                validation[i][j] = 0.0

    x = int(validation.shape[0])
    y = int(validation[0].shape[0])

    validation = np.delete(validation, 9, 0)
    validation_scores = np.delete(validation_scores, 9, 0)

    valid = np.zeros(shape=(x - 1, y))
    valid_scores = np.zeros(shape=(x - 1))

    for i in range(x - 1):
        for j in range(y):
            valid[i][j] = validation[i][j]
        valid_scores[i] = validation_scores[i]

    print("shapes")
    print(train.shape, " ", scores.shape)
    print(valid.shape, " ", valid_scores.shape)

    file = open("dtr.txt", 'w')

    regr_1 = DecisionTreeRegressor()
    regr_1.fit(train, scores)
    result = regr_1.predict(valid)

    five = 0
    ten = 0
    fifteen = 0
    twenty = 0
    differences = np.abs(valid_scores - result)

    for diff in differences:
        if (diff <= 5):
            five += 1
        elif (diff <= 10):
            ten += 1
        elif (diff <= 20):
            twenty += 1

    five = five / valid.shape[0]
    ten = ten / valid.shape[0] + five
    fifteen = fifteen / valid.shape[0] + ten
    twenty = twenty / valid.shape[0] + fifteen
    file.write("dtr" + "_none" + " err(%5): " + str(five) + " err(%10): " + str(ten) + " err(%15): " + str(fifteen) + " err(%20): " + str(twenty))
    file.write("\n")

    plt.figure()
    sortedresult = returnSortedData(valid_scores, result)
    #sortedvalid = returnSortedData(result, valid_scores)
    plt.plot(sorted(valid_scores), 'o', label="valid")
    plt.plot(sortedresult, label="result")
    #plt.plot(sortedvalid, 'o', label="valid")
    #plt.plot(sorted(result), 'o', label="result")
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title("Decision Tree Regression")
    plt.legend()
    # plt.show()
    plt.savefig("dtr_none"+'.png')

    for i in range(1, 35, 1):
        regr_1 = DecisionTreeRegressor(max_depth=i)
        regr_1.fit(train, scores)
        result = regr_1.predict(valid)

        five = 0
        ten = 0
        fifteen = 0
        twenty = 0
        differences = np.abs(valid_scores - result)

        for diff in differences:
            if (diff <= 5):
                five += 1
            elif (diff <= 10):
                ten += 1
            elif (diff <= 20):
                twenty += 1

        five = five / valid.shape[0]
        ten = ten / valid.shape[0] + five
        fifteen = fifteen / valid.shape[0] + ten
        twenty = twenty / valid.shape[0] + fifteen
        file.write("dtr" + "_" + str(i) + " err(%5): " + str(five) + " err(%10): " + str(ten) + " err(%15): " + str(fifteen) + " err(%20): " + str(twenty))
        file.write("\n")

        plt.figure()
        sortedresult = returnSortedData(valid_scores, result)
        #sortedvalid = returnSortedData(result, valid_scores)
        plt.plot(sorted(valid_scores), 'o', label="valid")
        plt.plot(sortedresult, 'o', label="result")
        #plt.plot(sortedvalid, 'o', label="valid")
        #plt.plot(sorted(result), 'o', label="result")
        plt.xlabel('data')
        plt.ylabel('target')
        plt.title("Decision Tree Regression")
        plt.legend()
        # plt.show()
        plt.savefig("dtr_"+str(i)+'.png')
    file.close()

def svr():
    train = np.load("trainsame31122101.npy") / 100
    scores = np.load("trainscore.npy")
    validation = np.load("valsame31122101.npy") / 100
    validation_scores = np.load("valscore.npy")
    print("train NaNs")
    for i in range(train.shape[0]):
        for j in range(train.shape[1]):
            if (math.isnan(train[i][j])):
                print(i, j)
                train[i][j] = 0.0
    print("validation NaNs")
    for i in range(validation.shape[0]):
        for j in range(validation.shape[1]):
            if (math.isnan(validation[i][j])):
                print(i, j)
                validation[i][j] = 0.0

    x = int(validation.shape[0])
    y = int(validation[0].shape[0])

    validation = np.delete(validation, 9, 0)
    validation_scores = np.delete(validation_scores, 9, 0)

    valid = np.zeros(shape=(x - 1, y))
    valid_scores = np.zeros(shape=(x - 1))

    for i in range(x - 1):
        for j in range(y):
            valid[i][j] = validation[i][j]
        valid_scores[i] = validation_scores[i]

    print("shapes")
    print(train.shape, " ", scores.shape)
    print(valid.shape, " ", valid_scores.shape)

    file = open("svr.txt", 'w')

    for c in (5, 10, 25, 50, 100, 250,500):
        for funct in ('rbf', 'sigmoid', 'linear'):
            clf = SVR(C=c, gamma='auto', kernel=funct)
            model = clf.fit(train, scores)
            result = model.predict(valid)

            five = 0
            ten = 0
            fifteen = 0
            twenty = 0
            differences = np.abs(valid_scores - result)

            for diff in differences:
                if (diff <= 5):
                    five += 1
                elif (diff <= 10):
                    ten += 1
                elif (diff <= 20):
                    twenty += 1

            five = five / valid.shape[0]
            ten = ten / valid.shape[0] + five
            fifteen = fifteen / valid.shape[0] + ten
            twenty = twenty / valid.shape[0] + fifteen
            file.write( str(c) + "_" + str(funct) + " err(%5): " + str(five) + " err(%10): " + str(ten) + " err(%15): " + str(fifteen) + " err(%20): " + str(twenty))
            file.write("\n")

            plt.figure()
            sortedresult = returnSortedData(valid_scores, result)
            #sortedresult = returnSortedData(result, valid_scores)
            plt.plot(sorted(valid_scores), 'o', label="valid")
            plt.plot(sortedresult, 'o', label="result")
            plt.xlabel('data')
            plt.ylabel('target')
            plt.title("Support Vector Regression")
            plt.legend()
            # plt.show()
            plt.savefig("svr_" + funct + "_" + str(c) + '.png')
    file.close()

def rfr():
    train = np.load("trainsame31122101.npy") / 100
    scores = np.load("trainscore31122101.npy")
    validation = np.load("valsame31122101.npy") / 100
    validation_scores = np.load("valscore31122101.npy")
    print("train NaNs")
    for data in (train, validation):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (math.isnan(data[i][j])):
                    print(i, j)
                    data[i][j] = 0
    print("validation NaNs")
    for i in range(validation.shape[0]):
        for j in range(validation.shape[1]):
            if (math.isnan(validation[i][j])):
                print(i, j)
                validation[i][j] = 0.0

    x = int(validation.shape[0])
    y = int(validation[0].shape[0])
    validation = np.delete(validation, 9, 0)
    validation_scores = np.delete(validation_scores, 9, 0)

    valid = np.zeros(shape=(x - 1, y))
    valid_scores = np.zeros(shape=(x - 1))

    for i in range(x - 1):
        for j in range(y):
            valid[i][j] = validation[i][j]
        valid_scores[i] = validation_scores[i]

    print("shapes")
    print(train.shape, " ", scores.shape)
    print(valid.shape, " ", valid_scores.shape)

    file = open("rfr.txt", 'w')

    reg = RandomForestRegressor(bootstrap=True, random_state=0)
    model = reg.fit(train, scores)
    result = model.predict(valid)
    # print(clf.feature_importances_)
    five = 0
    ten = 0
    fifteen = 0
    twenty = 0
    differences = np.abs(valid_scores - result)

    for diff in differences:
        if (diff <= 5):
            five += 1
        elif (diff <= 10):
            ten += 1
        elif (diff <= 20):
            twenty += 1

    five = five / valid.shape[0]
    ten = ten / valid.shape[0] + five
    fifteen = fifteen / valid.shape[0] + ten
    twenty = twenty / valid.shape[0] + fifteen
    print("max_depth=none err(%5): ", five, " err(%10): ", ten, " err(%15): ", fifteen, " err(%20): ", twenty)
    file.write("max_depth=none err(%5): " + str(five) + " err(%10): " + str(ten) + " err(%15): " + str(fifteen) + " err(%20): "+ str(twenty))
    file.write("\n")
    sortedresult = returnSortedData(valid_scores, result)
    #sortedresult = returnSortedData(result, valid_scores)

    plt.figure()
    plt.plot(sorted(valid_scores), 'o', label="valid")
    plt.plot(sortedresult, 'o', label="result")
    plt.title("Random Forest Regression")
    plt.legend()
    # plt.show()
    plt.savefig("rfr_bs_t_none.png")

    for a in range(1, 35, 1):
        reg = RandomForestRegressor(bootstrap=True, max_depth=a, random_state=0)
        model = reg.fit(train, scores)
        result = model.predict(valid)
        five = 0
        ten = 0
        fifteen = 0
        twenty = 0
        differences = np.abs(valid_scores - result)

        for diff in differences:
            if (diff <= 5):
                five += 1
            elif (diff <= 10):
                ten += 1
            elif (diff <= 20):
                twenty += 1

        five = five / valid.shape[0]
        ten = ten / valid.shape[0] + five
        fifteen = fifteen / valid.shape[0] + ten
        twenty = twenty / valid.shape[0] + fifteen
        print("max_depth=", a, " err(%5): ", five, " err(%10): ", ten, " err(%15): ", fifteen, " err(%20): ", twenty)
        file.write("max_depth=" + str(a) + " err(%5): " + str(five) + " err(%10): " + str(ten) + " err(%15): " + str(fifteen) + " err(%20): " + str(twenty))
        file.write("\n")
        sortedresult = returnSortedData(valid_scores, result)
        #sortedresult = returnSortedData(result, valid_scores)
        plt.figure()
        plt.plot(sorted(valid_scores), 'o', label="valid")
        plt.plot(sortedresult, 'o', label="result")
        plt.title("Random Forest Regression")
        plt.legend()
        # plt.show()
        plt.savefig("rfr_bs_t_" + str(a) + ".png")
    file.close()
dtr()
svr()
rfr()