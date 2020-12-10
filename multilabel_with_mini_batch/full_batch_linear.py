from __future__ import print_function

import sys
from pyspark import SparkContext
import numpy as np


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)
        
    sc = SparkContext(appName="Taxi linear regression with full batch")
    rdd = sc.textFile(sys.argv[1], 1)
    rdd = rdd.map(lambda x: x.split(','))
    
    
    def isfloat(value):
        try :
            float(value)
            return True
        except :
             return False
    
    
    def correctRows (p):
        if(len(p) == 17):
            if (isfloat(p[5]) and isfloat(p[11]) 
                and isfloat(p[4]) and isfloat(p[12]) and isfloat(p[12]) ):
                if (float(p[4]) >= 2 * 60 and float(p[4]) <= 60 * 60
                and float(p[11]) >= 3 and float(p[11]) <= 200
                and float(p[5]) >= 1 and float(p[5]) <= 50
                and float(p[15]) >= 3):
                    return p
      
    
    
    def pred(x):
        """a function to predict total amount"""
        prediction = np.dot(x[0], theta)
        loss = (x[1] - prediction) ** 2
        return np.array([x[1], prediction, loss])
    
    
    """filter and map value with dummy value 1 for the intercept theta[0]
    # I converted seconds to be minues because errors were too big 
    compared to total_amount"""
    
    rdd = rdd.filter(correctRows).map(lambda x: (1, float(x[4])/60, 
                                                 float(x[5]), float(x[11]), 
                                                 float(x[12]), float(x[16])) )
    
    # rdd.take(2)
    rdd = rdd.map(lambda x: (np.array(x[:-1]), x[-1]) )
    training, testing = rdd.randomSplit([0.9, 0.1])
    training.cache()
    testing.cache()
    n_count = training.count()
    
    
    """Start gradient descent"""
    # theta = [intercept, para1, para2, para3, para4]
    num_iteration = 300
    learning_rate = 0.0001
    theta = np.zeros(5) + 0.1
    old_loss = 0
    loss_array = []
    precision = 0.1


    def loss_grad(x, para):
        "A function to compute loss and gradient by each row"
        product = np.dot(x[0], para)
        loss = (x[1] - product) ** 2
        grad = - x[0] * (x[1] - product)
        return loss, grad


    # 300 iterations
    for i in range(num_iteration):
        """iteration to minimize squared error
        """
        rdd2 = training.map(lambda x: loss_grad(x, theta))
        loss1, grad = rdd2.reduce(lambda a,b: [a[0] + b[0], a[1] + b[1] ] )
        gradient = grad / n_count
        loss = loss1 / n_count
        # get new m and b and loss
        theta = theta - learning_rate * gradient

        if loss < old_loss:
            learning_rate = learning_rate * 1.05
        else:
            learning_rate = learning_rate * 0.5
            
        
        if(abs(loss - old_loss) <= precision):
            print("Stoped at iteration", i)
            break

        loss_array.append(loss)   
        old_loss = loss

        print('Iteration', i + 1, ', Cost =', round(loss, 4), ', Parameters =', np.round(theta, 4))
    
    
      
        
    prediction = testing.map(pred)
    prediction.cache()
    ave_error = prediction.reduce(lambda a,b: a + b )[2] / testing.count()
    print('The average MSE is', round(ave_error, 2))
    print('A few samples of predicted value and ture value are') 
    show = prediction.take(10)
    print(np.round(show, 2))
    
    
    loss_save = sc.parallelize(loss_array)
    loss_save.coalesce(1).saveAsTextFile(sys.argv[2])
    
    sc.stop()
    
  
    