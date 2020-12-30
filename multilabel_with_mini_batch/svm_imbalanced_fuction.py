from __future__ import print_function
import sys
from pyspark import SparkContext
import re
import numpy as np
import time


if __name__ == "__main__":
    total_time_start = time.time()
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)
        
    sc = SparkContext(appName='court classify by SVM with imbalanced measure')
    read_time_start = time.time()
    # read training and testing datasets
    training_data = sc.textFile(sys.argv[1], 1)
    test_data = sc.textFile(sys.argv[2], 1)
    read_time_end = time.time()
    read_time = read_time_end - read_time_start
    
    
    def freqArray (listOfIndices, numberofwords):
        """ function to get TF array"""
        returnVal = np.zeros (20000)
        for index in listOfIndices:
            returnVal[index] = returnVal[index] + 1
        returnVal = np.divide(returnVal, numberofwords)
        return returnVal


    def preprocess(dataset):
        """Process file to key and list of words"""
        rdd = dataset.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
        regex = re.compile('[^a-zA-Z]')
        keyAndListOfWords = rdd.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
        numberOfDocs = keyAndListOfWords.count()
        return keyAndListOfWords, numberOfDocs


    def get_tfArray(key_and_words):
        """Compute term frequency"""
        allWordsWithDocID = key_and_words.map(lambda x: (x[0], x[1], len(x[1]))).flatMap(lambda x: ((j, (x[0], x[2])) for j in x[1]))  
        # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
        allDictionaryWords = dictionary.join(allWordsWithDocID.distinct())
        justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]) )
        # Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
        allDictionaryWordsInEachDoc = justDocAndPos.groupByKey().mapValues(list)
        allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0][0], freqArray(x[1], x[0][1])))
        return allDocsAsNumpyArrays


    def true_labe(x):
        """Extract true labels by the first 2 letters of docID"""
        if x[:2] == 'AU':
            return 1.0
        else:
            return -1.0
        
        
    def loss_gradient(x):
        """The function to compute loss and gradient by row"""
        y = x[0]
        prod = y * (np.dot(para, x[1]) )
        par1 = 1 - prod
        par2 = nr * ((1 + y) / 2) * par1 + pr * ((1 - y)/ 2) * par1
        loss = max(0, par2)
        grad2 = x[1] / 2 * (pr * (1 - y) - nr * (1 + y))
        grad = 0 if prod > 1 else grad2
        return loss, grad
        
    
    keyAndListOfWords_traing, traing_count = preprocess(training_data)
    keyAndListOfWords_traing.cache()


    # compute dictionary
    allWords = keyAndListOfWords_traing.flatMap(lambda x: (x[1])).map(lambda x: (x, 1))
    allCounts = allWords.reduceByKey(lambda a,b: a + b)
    topWords = allCounts.top(20000, key= lambda x: x[1])
    topWordsK = sc.parallelize(range(20000))
    dictionary1 = topWordsK.map(lambda x : (topWords[x][0], x))
    dictionary = dictionary1
    dictionary.cache()

    # get traing_data key_dfArray pairs
    tf_training = get_tfArray(keyAndListOfWords_traing)
    
    # get label, x pairs for Logistic model
    featureArray = tf_training.map(lambda x: (true_labe(x[0]), x[1]) )
    featureArray.cache()
    
    # get pr and nr
    def pr_count(x):
        z = 1 if x[0] == 1.0 else 0
        return z
        
        
    pr = featureArray.map(lambda x: pr_count(x)).reduce(lambda a,b: a + b) / traing_count
    nr = 1 - pr
       
    # train the model
    train_time_start = time.time()
    
    """Start gradient descent"""
    # w = [para1, para2, para3, para4, para2000]
    num_iteration = 500
    para = np.zeros(20000) + 0.1
    learning_rate = 0.1
    precision = 0.1
    loss_array = []
    num_stop = []
    old_loss = 0
    

    for i in range(num_iteration):
        loss, gradient = featureArray.map(lambda x: loss_gradient(x)).reduce(lambda a,b: ([a[0] + b[0], a[1] + b[1] ]))
        para = para - learning_rate * gradient
        print(para)
        if loss < old_loss:
            learning_rate = learning_rate * 1.05
        else:
            learning_rate = learning_rate * 0.5

        loss_array.append(loss)
        num_stop.append(i)
        # Stop if the cost is not descreasing 
        if(abs(loss - old_loss) <= precision):
            print("Stoped at iteration", i)
            break

        old_loss = loss
        print('Iteration', i + 1, ', Cost =', round(loss, 4) )

    train_time_end = time.time()
    train_time = train_time_end - train_time_start
    print('Paramemters:', para)
    
    """Processing test data to get test TF array"""
    test_pre, n_test = preprocess(test_data)
    tf_test = get_tfArray(test_pre)
    tf_test.cache()
    
    # predict labels
    
    def predict(x):
        """Predict labels by trained parameters"""
        prod = np.dot(x[1], para)
        y = true_labe(x[0])
        y_predict = 1.0 if prod > 0 else -1.0
        return x[0], y, y_predict
    
    
    def confusion_matrix(x):
        """A function to get a confusion matrix"""
        z = int(x[1])
        tp = 1 if (x[1] == x[2] and z == 1) else 0
        fn = 1 if (x[1] != x[2] and z == 1) else 0
        tn = 1 if (x[1] == x[2] and z == -1) else 0
        fp = 1 if (x[1] != x[2] and z == -1) else 0
        return tp, tn, fp, fn
    
    
    # predict labels
    test_time_start = time.time()
    predictions = tf_test.map(lambda x: (predict(x)) )
    # I would be the docID and true labels with predicted labels in case of we want to know which doc was misclassified
    predictions.cache()
    
    
    # get F1 score
    matrix = predictions.map(lambda x: confusion_matrix(x)).reduce(lambda a,b: [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
    tp, tn, fp, fn = matrix
    precision2 = tp / (tp + fp)
    recall = tp/(tp + fn)
    f_score = 2 * precision2 * recall / (precision2 + recall)
    print("Task 3")
    print('The F score of this model is', round(f_score, 4))
    print()
    
    test_time_end = time.time()
    test_time = test_time_end - test_time_start

    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    
    # print time counts
    print('The total time to read the data is', round(read_time, 2), 'seconds')
    print('The total time to train the model is', round(train_time, 2), 'seconds')
    print('The total time to test the model is', round(test_time, 2), 'seconds')
    print('The total time of this program from the start to the end is', round(total_time, 2), 'seconds')
    
    sc.stop()
