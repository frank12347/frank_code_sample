
# Description

This is a multiclass classification project. The dataset contains 3 different classes: 'General wiki pages', 'Austraila court documents' and 'Adverstisements'.
I use hierarchical classfication to classify labels as a sequence of:
    Promotion versus non-promotion
    Court cases versus general documents



# Other Documents. 

Source: https://www.kaggle.com/urbanbricks/wikipedia-promotional-articles?select=promotional.csv
Wiki pages:CS777 course from professor Kia Teymourian  


# How to run  

Run the tasks by submitting the task to spark-submit. 
Run the python files on Google Cloud.
Run the python files on Amazon AWS.


```python

spark-submit full_batch.py 

argument1: gs://cs777_project/promotion.txt
argument2: gs://cs777_project/wiki_page.txt

```



```python

spark-submit MLE_full_batch_without_stemming.py 
argument1: gs://cs777_project/promotion.txt
argument2: gs://cs777_project/wiki_page.txt

```



```python

spark-submit MLE_mini_batch.py 
argument1: gs://cs777_project/promotion.txt
argument2: gs://cs777_project/wiki_page.txt

```



```python

spark-submit svm_mini_batch.py 
argument1: gs://cs777_project/promotion.txt
argument2: gs://cs777_project/wiki_page.txt

```



```python

spark-submit logistic_regression.py 
argument1: gs://cs777_project/promotion.txt
argument2: gs://cs777_project/wiki_page.txt

```



```python

spark-submit random_forest.py 
argument1: gs://cs777_project/promotion.txt
argument2: gs://cs777_project/wiki_page.txt
        
```



```python

spark-submit mini_batch_linear.py 
argument1: gs://metcs777/taxi-data-sorted-large.csv.bz2
argument2: your path to save loss
        
```




```python

spark-submit full_batch_linear.py 
argument: gs://metcs777/taxi-data-sorted-large.csv.bz2
argument2: your path to save loss




