# Spam Classification using Naive Bayes

## Introduction

This project aims to classify spam and non-spam (ham) messages using Naive Bayes classification. The dataset used for this project is from the UCI Machine Learning Repository, containing SMS messages labeled as spam or ham. 

## Dependencies

This project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk

## Usage

To use this project, follow these steps:

1. Import the necessary libraries:

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import nltk
    from nltk.corpus import stopwords
    import string
    from nltk.stem.porter import PorterStemmer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
    from sklearn.metrics import accuracy_score, precision_score
    ```

2. Load the dataset:

    ```python
    df = pd.read_csv("spam.csv")
    ```

3. Perform data cleaning:

    ```python
    # Dropping unnecessary columns
    df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
    
    # Changing column names
    df.rename(columns={'v1':'Target', 'v2':'Text'}, inplace=True)
    
    # Drop duplicate values
    df.drop_duplicates(inplace=True)
    
    # Converting Target column into numerical 0 and 1
    encoder = LabelEncoder()
    df['Target'] = encoder.fit_transform(df['Target'])
    ```

4. Explore the data:

    ```python
    # Ham and Spam count
    df['Target'].value_counts()
    
    # Check the Target value using pie chart
    plt.pie(df['Target'].value_counts(), labels=['Ham','Spam'], autopct='%.2f')
    plt.show()
    ```

5. Perform data preprocessing:

    ```python
    def process_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        x = []
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                x.append(i)
        text = x[:]
        x.clear()
        ps = PorterStemmer()
        for i in text:
            x.append(ps.stem(i))
        return " ".join(x)

    df['Transformed_text'] = df['Text'].apply(process_text)
    ```

6. Build the model:

    ```python
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['Transformed_text'])
    y = df['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=2)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred1 = gnb.predict(X_test)

    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    y_pred2 = bnb.predict(X_test)

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred3 = mnb.predict(X_test)
    ```

7. Evaluate the model:

    ```python
    print("GaussianNB Accuracy:", accuracy_score(y_test, y_pred1))
    print("GaussianNB Precision:", precision_score(y_test, y_pred1))

    print("BernoulliNB Accuracy:", accuracy_score(y_test, y_pred2))
    print("BernoulliNB Precision:", precision_score(y_test, y_pred2))

    print("MultinomialNB Accuracy:", accuracy_score(y_test, y_pred3))
    print("MultinomialNB Precision:", precision_score(y_test, y_pred3))
    ```

## Contributors

- Zeinab Atef Abdallah
