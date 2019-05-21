## Building and deploying a spam detection model
Spam is an unsolicited email message, instant message, or text message – usually sent to the recipient for commercial purposes. In other words, the recipient never explicitly asked for it, yet the get them.

We’re going to learn how to build a spam detection model in python to automatically classify a message as either spam or ham(legitimate messages). When we’re done building our model, we will host it on Heroku using Flask, a lightweight web application framework so that it becomes available for anyone to use.


## Building the Spam Message Classifier

* Loading the Dataset: The dataset we will use is a collection of SMS messages tagged as spam or ham and can be found [here](https://www.kaggle.com/uciml/sms-spam-collection-dataset/downloads/sms-spam-collection-dataset.zip/1#spam.csv) – go ahead and download it. Once you have the dataset, open your jupyter notebook and let’s get to work. Our first step is to load in the data using pandas read_csv function.

```python
import pandas as pd
df = pd.read_csv('spam.csv', encoding="latin-1")
```

Note that we specified encoding=”latin-1″ while reading the csv. This is because the csv file is utf-8 encoded. Failure to add that encoding, you’ll get an error message. Let’s go on

![Jupyter](images/Capture.PNG)

* Cleaning the Dataset: Observing the dataframe, we will notice how we have three columns Unnamed: 2, Unnamed: 3, Unnamed: 4 whose rows are NaN values. We will drop them because they’re not useful for our classification. We will also rename the v1 and v2 columns and give them appropriate titles:

```python
#Drop the columns not needed
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
#Create a new column label which has the same values as v1 then set the ham and spam values to 0 and 1 which is the standard format for our prediction.
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
#Create a new column having the same values as v2 column
df['message'] = df['v2']
#Now drop the v1 and v2
df.drop(['v1', 'v2'], axis=1, inplace=True)
df.head(10)
```
![Jupyter](images/Capture1.PNG)
