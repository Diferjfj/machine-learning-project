import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import logisticregression
if __name__ == "__main__":
    path = './'
    filename = 'SMSSpamCollection.txt'
    df = pd.read_csv(path + filename, delimiter='\t', header=None)
    y,X_train = df[0],df[1]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_train)
    y = y.map({'spam': 1, 'ham': 0}).values
    print("Loaded dataset:", df.shape)
    print(df.head())

    lr = LogisticRegression(batch_size=200)
    lr.fit(X, y)

    testX = vectorizer.transform(['Urgent: hello, Your mobile was awarded a Prize!',
                             'hello,how are you'])
    predictions = lr.predict(testX)[0]
    if(predictions==1):
        predictions='spam'
    else:
        predictions='ham'
    print(predictions)

