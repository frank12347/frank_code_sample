from __future__ import print_function
import sys
from pyspark import SparkContext

import re
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
import time


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)
        
    sc = SparkContext(appName='Random forest multilabels prediction for promotion, general and court cases files')
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
    
    
    promotion = rdd1.zipWithUniqueId().map(lambda x: ('AD' + str(x[1]), x[0])).map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
    wiki, wiki_count = preprocess(rdd2)
    dataset = wiki.union(promotion)
    training, testing = dataset.randomSplit([0.8, 0.2], seed=13)
    print('Training data count', training.count())
    print('Testing data count', testing.count())
    
    
    # remove stop words and do word stemming
    stemer = SnowballStemmer(language='english')
    stem_fun = np.vectorize(lambda x: stemer.stem(x)) 
    stop_words = np.array(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
       "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
       'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
       'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
       'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
       'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
       'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
       'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
       'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
       'by', 'for', 'with', 'about', 'against', 'between', 'into',
       'through', 'during', 'before', 'after', 'above', 'below', 'to',
       'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
       'again', 'further', 'then', 'once', 'here', 'there', 'when',
       'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
       'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
       'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
       'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
       'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
       "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
       'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
       'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
       'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
       'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

    def word_process(x):
        # word stemming and stop words removal
        words = np.array(x)
        words = stem_fun(words)
        words = words[np.isin(words, stop_words, invert=True)]
        return words

    
    # process training dataset
    processed_training = training.map(lambda x: (x[0], word_process(x[1])))
    processed_training.cache()
    
    
    # make a 20000 words dictionary
    allWords = processed_training.flatMap(lambda x: (x[1])).map(lambda x: (x, 1))
    allCounts = allWords.reduceByKey(lambda a,b: a + b)
    topWords = allCounts.top(20000, key= lambda x: x[1])
    topWordsK = sc.parallelize(range(20000))
    dictionary = topWordsK.map(lambda x : (topWords[x][0], x))
    dictionary.cache()
    
    
    tf_training = get_tfArray(processed_training)
    training_count = training.count()
    idf_training = get_tf_idfArray(tf_training, training_count)
    idf_training.cache()
    print(idf_training.map(lambda x: x[1]).filter(lambda x: x == 0).count())
    print(idf_training.map(lambda x: x[1]).filter(lambda x: x == 1).count())
    print(idf_training.map(lambda x: x[1]).filter(lambda x: x == 2).count())
    print('Total documents:', training_count)
    
    def parsePoint(x):
        return LabeledPoint(x[1], x[2])
    
    feature_array = idf_training.map(lambda x: parsePoint(x))
    feature_array.cache()
    
    
    def confusion_matrix(x, label):
        z = int(x[0])
        tp = 1 if (x[0] == x[1] and z == label) else 0
        fn = 1 if (x[0] != x[1] and z == label) else 0
        tn = 1 if (x[0] == x[1] and z == 0) else 0
        fp = 1 if (x[0] != x[1] and z == 0) else 0
        return tp, tn, fp, fn
    
    
    def measure(rdd, label):
        confusion = rdd.map(lambda x: confusion_matrix(x, label)).reduce(lambda a,b: [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
        tp, tn, fp, fn = confusion
        precision = tp / (tp + fp)
        recall = tp/(tp + fn)
        f_score = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (fp + fn + tp + tn)
        print("Task 3")
        print('TP', tp)
        print('TN', tn)
        print('FP', fp)
        print('FN', fn)
        print('The F score of this model is', round(f_score, 4))
        print('The Accuracy of this model is', round(accuracy, 4))
    
    
    # training the model one against others, the model has a parameter array for different lables
    
    
    print('Multilabel classification by random forest')
    train_times = time.time()
    model = RandomForest.trainClassifier(feature_array, numClasses=3, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)
    train_time_end = time.time()
    train_time = train_time_end - train_times
    
    # process the testing data
    processed_testing = testing.map(lambda x: (x[0], word_process(x[1])))
    tf_testing= get_tfArray(processed_testing)
    testing_count = tf_testing.count()
    idf_testing = get_tf_idfArray(tf_testing, testing_count)
    idf_testing.cache()
    
    
    
    

    """Model evaluation"""
    test_time_start = time.time()
    
    # get F1 score
    
    predictions = model.predict(idf_testing.map(lambda x: x[2]))
    labelsAndPredictions = idf_testing.map(lambda lp: lp[1]).zip(predictions)
    measure(labelsAndPredictions, 1)
    measure(labelsAndPredictions, 2)
    test_time_end = time.time()
    test_time = test_time_end - test_time_start
    
    print('The F1 meature of this model is', f1)
    
    # print time counts
    print('The total time to train the model is', round(train_time, 2))
    print('The total time to test the model is', round(test_time, 2))
    
    sc.stop()
    
    