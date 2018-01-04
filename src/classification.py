from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math

def rfc():
    for classnum in (2, 4, 10):
        print(classnum)
        file = open("result_rfc_bs_" + str(classnum) + ".txt", 'w')
        train = np.load("trainsame31122101.npy") / 100
        scores = np.load("trainClass" + str(classnum) + ".npy")
        validation = np.load("valsame31122101.npy") / 100
        validation_scores = np.load("ValClass" + str(classnum) + ".npy")

        print("train NaNs")
        for data in (train, validation):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if (math.isnan(data[i][j])):
                        print(i, j)
                        data[i][j] = 0

        x = int(validation.shape[0])
        y = int(validation[0].shape[0])

        # validation = np.delete(validation,9,0)
        # validation_scores = np.delete(validation_scores,9,0)

        valid = np.zeros(shape=(x - 1, y))
        valid_scores = np.zeros(shape=(x - 1))

        for i in range(x - 1):
            for j in range(y):
                valid[i][j] = validation[i][j]
            valid_scores[i] = validation_scores[i]

        print("shapes")
        print(train.shape, " ", scores.shape)
        print(valid.shape, " ", valid_scores.shape)

        clf = RandomForestClassifier(bootstrap=True, random_state=0)
        model = clf.fit(train, scores)
        result = model.predict(valid)
        f = open(str(classnum)+"_"+"none"+".txt",'w')
        f.write(str(clf.feature_importances_))
        f.close()
        file.write("max_depth:")
        file.write("none")
        file.write("  accuracy:   ")
        file.write(str(clf.score(valid, valid_scores)))
        file.write("\n")

        for d in range(1, 35, 1):
            clf = RandomForestClassifier(bootstrap=True, max_depth=d, random_state=0)
            model = clf.fit(train, scores)
            result = model.predict(valid)
            print(d, clf.feature_importances_)
            f = open(str(classnum) + "_" + str(d) + ".txt", 'w')
            f.write(str(clf.feature_importances_))
            f.close()
            file.write("max_depth:")
            file.write(str(d))
            file.write("  accuracy:   ")
            file.write(str(clf.score(valid, valid_scores)))
            file.write("\n")
        file.close()

def dtc():
    for classnum in (2, 4, 10):
        print(classnum)
        file = open("result_dtc_" + str(classnum) + ".txt", 'w')
        train = np.load("trainsame31122101.npy") / 100
        scores = np.load("trainClass" + str(classnum) + ".npy")
        validation = np.load("valsame31122101.npy") / 100
        validation_scores = np.load("ValClass" + str(classnum) + ".npy")

        print("train NaNs")
        for data in (train, validation):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if (math.isnan(data[i][j])):
                        print(i, j)
                        data[i][j] = 0

        x = int(validation.shape[0])
        y = int(validation[0].shape[0])

        # validation = np.delete(validation,9,0)
        # validation_scores = np.delete(validation_scores,9,0)

        valid = np.zeros(shape=(x - 1, y))
        valid_scores = np.zeros(shape=(x - 1))

        for i in range(x - 1):
            for j in range(y):
                valid[i][j] = validation[i][j]
            valid_scores[i] = validation_scores[i]

        print("shapes")
        print(train.shape, " ", scores.shape)
        print(valid.shape, " ", valid_scores.shape)

        regr_1 = DecisionTreeClassifier(max_depth=i, max_features=j)
        regr_1.fit(train, scores)
        y_1 = regr_1.predict(valid)
        accuracy = regr_1.score(valid, valid_scores)
        print("depth: ", i, "accuracy: ", accuracy)
        file.write("max_depth:")
        file.write("none")
        file.write("    ")
        file.write(str(accuracy))
        file.write("\n")

        max_accuracy = 0
        depth = 0
        features = 0;
        for i in range(1, 35, 1):
            for j in range(85, 86, 1):
                regr_1 = DecisionTreeClassifier(max_depth=i, max_features=j)
                regr_1.fit(train, scores)
                y_1 = regr_1.predict(valid)
                accuracy = regr_1.score(valid, valid_scores)
                if (accuracy > max_accuracy):
                    max_accuracy = accuracy
                    depth = i
                    features = j
                print("depth: ", i, "accuracy: ", accuracy)
                file.write("max_depth:")
                file.write(str(i))
                file.write("    ")
                file.write(str(accuracy))
                file.write("\n")
        file.close()


def svc():
    for classnum in (2, 4, 10):
        print(classnum)
        file = open("result_svc_" + str(classnum) + ".txt", 'w')
        train = np.load("trainsame31122101.npy") / 100
        scores = np.load("trainClass" + str(classnum) + ".npy")
        validation = np.load("valsame31122101.npy") / 100
        validation_scores = np.load("ValClass" + str(classnum) + ".npy")

        print("train NaNs")
        for data in (train, validation):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if (math.isnan(data[i][j])):
                        print(i, j)
                        data[i][j] = 0

        x = int(validation.shape[0])
        y = int(validation[0].shape[0])

        # validation = np.delete(validation,9,0)
        # validation_scores = np.delete(validation_scores,9,0)

        valid = np.zeros(shape=(x - 1, y))
        valid_scores = np.zeros(shape=(x - 1))

        for i in range(x - 1):
            for j in range(y):
                valid[i][j] = validation[i][j]
            valid_scores[i] = validation_scores[i]

        print("shapes")
        print(train.shape, " ", scores.shape)
        print(valid.shape, " ", valid_scores.shape)

        for c in (5, 10, 25, 50, 100):
            accuracies = {'rbf': 0, 'sigmoid': 0, 'linear': 0}
            for funct in ('rbf', 'sigmoid', 'linear'):
                clf = SVC(C=c, gamma='auto', kernel=funct, decision_function_shape='ovr')
                model = clf.fit(train, scores)
                result = model.predict(valid)
                accuracies[funct] = clf.score(valid, valid_scores)

            print("C: ", c, accuracies)
            file.write("C: ")
            file.write(str(c))
            file.write("    ")
            file.write(str(accuracies['rbf']))
            file.write("    ")
            file.write(str(accuracies['sigmoid']))
            file.write("    ")
            file.write(str(accuracies['linear']))
            file.write("\n")
        file.close()

#svc()
#dtc()
rfc()