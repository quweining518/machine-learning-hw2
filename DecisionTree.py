import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def dt_train(X, y, max_nodes):
    clf = DecisionTreeClassifier(max_leaf_nodes=max_nodes)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    loss = np.sum(y_pred != y)/len(y)  #0-1 loss
    n_nodes = clf.tree_.node_count
    return clf, y_pred, loss, n_nodes

def rf_train(X, y, n_tree, max_nodes):
    rfc = RandomForestClassifier(n_estimators=n_tree, max_leaf_nodes=max_nodes)
    rfc.fit(X, y)
    y_pred = rfc.predict(X)
    loss = np.sum(y_pred != y)/len(y)  #0-1 loss
    n_nodes = rfc.estimators_[0].tree_.node_count
    return rfc, y_pred, loss, n_nodes


def test(X, y, clf):
    y_pred = clf.predict(X)
    loss = np.sum(y_pred != y) / len(y)  # 0-1 loss
    return y_pred, loss



if __name__ == '__main__':
    Ximg = np.load("../train.npy")
    Ytrain = np.load("../trainlabels.npy").astype(int)
    plt.figure(figsize=(10,10))
    labels = []
    for i in range(9):
        sample = i*1000
        image = Ximg[sample]
        labels.append(Ytrain[sample])
        plt.subplot(3,3,i+1)
        plt.imshow(image)
    plt.savefig("./Fig/Fashionmnist.png")
    plt.show()
    print(labels)

    Xtrain = Ximg.reshape((60000, -1)).astype(np.float32) / 255
    Xtest = np.load("../test.npy").reshape((10000, -1)).astype(np.float32) / 255
    Ytest = np.load("../testlabels.npy").astype(int)

    # ii Decision tree
    param_nums = [100,500,1000,3000,5000,10000,15000,20000,30000,50000]
    trainresult = []
    testresult = []
    treenodes = []
    for num in param_nums:
        clf, trainpred, trainloss, treenode = dt_train(Xtrain, Ytrain, max_nodes = num)
        trainresult.append((num,trainloss))
        treenodes.append(treenode)
        testpred, testloss = test(Xtest, Ytest, clf)
        testresult.append((num, testloss))
    fig1 = plt.figure()
    x1, y1 = zip(*trainresult)
    x2, y2 = zip(*testresult)
    plt.plot(np.log(x1), y1, label = 'Train loss')
    plt.plot(np.log(x2), y2, label = 'Test loss')
    plt.xlabel("Maximum permitted no. of leaf nodes (log scale)")
    plt.ylabel("0-1 Loss")
    plt.legend()
    plt.savefig("Fig/Decisiontree.png")
    plt.show()
    print("Decision tree done")
    print(trainresult)
    print(testresult)
    print(treenodes)



    # iii Random Forest (fix n_tree)
    n_nodes = [100,500,1000,3000,5000,10000,15000,20000,30000,50000]
    trainresult = []
    testresult = []
    treenodes = []
    n_tree = 100
    for num in n_nodes:
        rfc1, trainpred, trainloss, treenode = rf_train(Xtrain, Ytrain, n_tree=n_tree, max_nodes=num)
        trainresult.append((num*n_tree, trainloss))
        treenodes.append(treenode)
        testpred, testloss = test(Xtest, Ytest, rfc1)
        testresult.append((num*n_tree, testloss))
    fig2 = plt.figure()
    x1, y1 = zip(*trainresult)
    x2, y2 = zip(*testresult)
    plt.plot(np.log(x1), y1, label='Train loss')
    plt.plot(np.log(x2), y2, label='Test loss')
    plt.xlabel("Maximum permitted no. of leaf nodes (log scale)")
    plt.ylabel("0-1 Loss")
    plt.legend()
    plt.savefig("Fig/RFC1.png")
    plt.show()
    print("Random Forest fix trees done")
    print(trainresult)
    print(testresult)
    print(treenodes)


    # v Random Forest (fix leaves)
    n_trees = [10,50,100,200,300,500,700,800,1000]
    trainresult = []
    testresult = []
    nodes = 100
    for n in n_trees:
        rfc2, trainpred, trainloss, treenode = rf_train(Xtrain, Ytrain, n_tree=n, max_nodes=nodes)
        trainresult.append((nodes*n, trainloss))
        testpred, testloss = test(Xtest, Ytest, rfc2)
        testresult.append((nodes*n, testloss))
    fig3 = plt.figure()
    x1, y1 = zip(*trainresult)
    x2, y2 = zip(*testresult)
    plt.plot(np.log(x1), y1, label='Train loss')
    plt.plot(np.log(x2), y2, label='Test loss')
    plt.xlabel("Maximum permitted no. of leaf nodes (log scale)")
    plt.ylabel("0-1 Loss")
    plt.legend()
    plt.savefig("Fig/RF2.png")
    plt.show()
    print("Random Forest Fix tree leaves done")
    print(trainresult)
    print(testresult)



    # vi Random Forest (double descent)
    n_leaves = [100,500,1000,3000,5000,8000,10000]
    trainresult = []
    testresult = []
    n_tree_1 = 1
    for nl in n_leaves:
        rfc3, trainpred, trainloss, treenode = rf_train(Xtrain, Ytrain, n_tree=n_tree_1, max_nodes=nl)
        trainresult.append((treenode*n_tree_1, trainloss))
        testpred, testloss = test(Xtest, Ytest, rfc3)
        testresult.append((treenode*n_tree_1, testloss))
    n_trees = 1 * [2**i for i in range(1,8)]
    for nt in n_trees:
        rfc3, trainpred, trainloss, treenode = rf_train(Xtrain, Ytrain, n_tree=nt, max_nodes=nl)
        trainresult.append((treenode * nt, trainloss))
        testpred, testloss = test(Xtest, Ytest, rfc3)
        testresult.append((treenode * nt, testloss))

    fig4 = plt.figure()
    x1, y1 = zip(*trainresult)
    x2, y2 = zip(*testresult)
    plt.plot(np.log(x1), y1, label='Train loss')
    plt.plot(np.log(x2), y2, label='Test loss')
    plt.xlabel("Maximum permitted no. of leaf nodes (log scale)")
    plt.ylabel("0-1 Loss")
    plt.legend()
    plt.savefig("Fig/RF3_6.png")
    plt.show()
    print("Random forest double descent done")
    print(trainresult)
    print(testresult)











