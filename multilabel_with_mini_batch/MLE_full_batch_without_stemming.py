from __future__ import print_function
import sys
from pyspark import SparkContext

import re
import numpy as np
from pyspark.mllib.evaluation import MulticlassMetrics


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)
        
    sc = SparkContext(appName='MLE full batch multilabels prediction without word processing')
    regex = re.compile('[^a-zA-Z]')
    rdd1 = sc.textFile(sys.argv[1], 1)
    rdd2 = sc.textFile(sys.argv[2], 1)
        
    
    def freqArray (listOfIndices, numberofwords):
        """ function to get TF array"""
        returnVal = np.zeros(20000)
        
        for index in listOfIndices:
            returnVal[index] = returnVal[index] + 1
        returnVal = np.divide(returnVal, numberofwords)
        return returnVal


    def preprocess(dataset):
        """process data to get key and words pairs"""
        rdd = dataset.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
        regex = re.compile('[^a-zA-Z]')
        keyAndListOfWords = rdd.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
        numberOfDocs = keyAndListOfWords.count()
        return keyAndListOfWords, numberOfDocs


    def get_tfArray(key_and_words):
        """generates TF array"""
        allWordsWithDocID = key_and_words.map(lambda x: (x[0], x[1], len(x[1]))).flatMap(lambda x: ((j, (x[0], x[2])) for j in x[1]))  
        # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
        allDictionaryWords = dictionary.join(allWordsWithDocID.distinct())
        justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]) )
        # Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
        allDictionaryWordsInEachDoc = justDocAndPos.groupByKey().mapValues(list)
        allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0][0], freqArray(x[1], x[0][1])))
        return allDocsAsNumpyArrays
    
    
    def get_tf_idfArray(tf_pairs, numberOfDocs):
        """Get tf_idfArray with true labels"""
        zeroOrOne = tf_pairs.map(lambda x: (x[0], np.where(x[1] != 0, 1, 0)) )
        dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]
        idfArray = np.log(np.divide(np.full(20000, numberOfDocs), 1 + dfArray))
        allDocsAsNumpyArraysTFidf = tf_pairs.map(lambda x: (x[0], np.multiply(x[1], idfArray)))
        feature_array = allDocsAsNumpyArraysTFidf.map(lambda x: (x[0], true_labels(x[0]), x[1]) )
        return feature_array
    

    def true_labels(x):
        """Return labels, 0 denotes general docs, 1 is promotions, 2 is Australia court cases"""
        if x[:2] == 'AD':
            return 1
        
        if x[:2] == 'AU':
            return 2
        
        else:
            return 0


    def loss_and_derivative(x, coeficients, label):
        """ A single map to generate theta, loss, predicted labels and derivative 
        by rows.
        Return values: True label, prediction, loss, gradient by row    
        """
        theta = np.dot(x[2], coeficients)
        y = 1 if x[0][:2] == label else 0
        loss = -y * theta + np.log(1 + np.exp(theta))
        gradient_row = - x[2] * y + x[2] * (np.exp(theta) / (1 + np.exp(theta)))
        return loss, gradient_row


    def logistic_regression_paras(data, label):
        """logistic regression with mixmum likelihood"""
        # coef = [intercept, para1, para2, para3, para4]
        num_iteration = 300
        learning_rate = 0.01
        coef = np.zeros(20000) + 0.1
        precision = 0.01
        loss_array = []
        num_stop = []
        old_loss = 0
        regular_factor = 0.0001

        for i in range(num_iteration):
            rdd = data.map(lambda x: (loss_and_derivative(x, coef, label)))
            result = rdd.reduce(lambda a,b: [a[0] + b[0], a[1] + b[1]])
            gradient = result[1] + 2 * regular_factor * np.linalg.norm(coef)
            # I don't divide loss by the number of documents because its too small
            loss = result[0] + 2 * regular_factor * np.linalg.norm(coef)
            coef = coef - learning_rate * gradient

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

        largest_five = np.argsort(coef)[-25:]
        print('The top 25 words with largest coefficients:')
        top5_words = dictionary.filter(lambda x: x[1] in largest_five).map(lambda x: x[0]).collect()
        print(top5_words)
        return coef, loss_array
    
    
    def prediction(x, coef, lb):
        """A function to predict labels
        output is docID, label, predicted label and theta
        """
        theta = np.dot(x[2], coef)
        label = 1 if theta > 0 else 0
        y = 1 if x[1] == lb else 0
        return y, label
    
    
    def measure(rdd):
        metrics = MulticlassMetrics(rdd.map(lambda x: (float(x[1]), float(x[0])) ) )
        f1 = metrics.fMeasure(1.0, 1.0)
        print('The F score of this model is', round(f1, 4))
        print('The accuracy of this model is', round(metrics.accuracy, 2))
    
    
    promotion = rdd1.zipWithUniqueId().map(lambda x: ('AD' + str(x[1]), x[0])).map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    wiki, wiki_count = preprocess(rdd2)
    dataset = wiki.union(promotion)
    training, testing = dataset.randomSplit([0.8, 0.2], seed=13)
    print('Training data count', training.count())
    print('Testing data count', testing.count())
    
    
    training.cache()
    
    
    # make a 20000 words dictionary
    allWords = training.flatMap(lambda x: (x[1])).map(lambda x: (x, 1))
    allCounts = allWords.reduceByKey(lambda a,b: a + b)
    topWords = allCounts.top(20000, key= lambda x: x[1])
    topWordsK = sc.parallelize(range(20000))
    dictionary = topWordsK.map(lambda x : (topWords[x][0], x))
    dictionary.cache()
    
    
    tf_training = get_tfArray(training)
    training_count = training.count()
    idf_training = get_tf_idfArray(tf_training, training_count)
    idf_training.cache()
    print(idf_training.map(lambda x: x[1]).filter(lambda x: x == 0).count())
    print(idf_training.map(lambda x: x[1]).filter(lambda x: x == 1).count())
    print(idf_training.map(lambda x: x[1]).filter(lambda x: x == 2).count())
    print('Total documents:', training_count)
    
    
    # training the model hierarchically
    
    # First construct a model the classify advertisement and other docs
    print('First construct a model the classify advertisement and other docs')
    params, loss = logistic_regression_paras(idf_training, 'AD')
    
    # Second model to classify general docs and Austrilia court docs
    print('Second model to classify general docs and Austrilia court docs')
    params2, loss2 = logistic_regression_paras(idf_training, 'AU')
    
    # Third model to classify general docs and Austrilia court docs by only general file and court cases
    print('Second model to classify general docs and Austrilia court docs by only general file and court cases')
    idf_training2 = idf_training.filter(lambda x: x[1] != 1)
    idf_training2.cache()
    params3, loss3 = logistic_regression_paras(idf_training2, 'AU')
    
    
    # process the testing data
    tf_testing= get_tfArray(testing)
    testing_count = tf_testing.count()
    idf_testing = get_tf_idfArray(tf_testing, testing_count)
    idf_testing.cache()
    
    
    
    """Model evaluation"""
    pred_ad = idf_testing.map(lambda x: prediction(x, params, 1))
    pred_court = idf_testing.map(lambda x: prediction(x, params2, 2))
    pred_court2 = idf_testing.map(lambda x: prediction(x, params3, 2))
    print()
    print('Evaluation of predicting advertisement:')
    measure(pred_ad)
    print()
    print('Evaluation of predicting court cases')
    measure(pred_court)
    print('Evaluation of predicting court cases by only general file and court cases the second time')
    measure(pred_court2)
    
    sc.stop()
    