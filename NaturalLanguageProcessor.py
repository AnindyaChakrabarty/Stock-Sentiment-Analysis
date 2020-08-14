
from Input import Input ,MongoDB, Report
from Classifier import Classifier, Utility

class NaturalLanguageProcessor:
    def __init__(self,input):
        import pandas as pd
        import os
        import numpy as np
        import logging
        import warnings
        warnings.filterwarnings('ignore')
        self.input_=input
        self.util_=Utility()
        self.logger_=self.util_.SetLogger()
        self.dataset_=input.readMongoData()
        self.Header_=list(self.dataset_.columns)
        self.Header_.remove('Date')
        self.dataset_=self.dataset_[self.Header_]
        print(self.dataset_.head())
        self.dependentVariableName_=input.dependentVariableName_
        self.dataFileName_=input.collectionName_
        self.parentDirectory_ = os.path.dirname(os.getcwd())
        self.exploratoryDataAnalysisDir_= self.util_.makeDir(self.parentDirectory_,"Natural Language Processing")
    def joinNews(self):
        import pandas as pd
        self.dataset_["Combined_News"]=0
        print("**************Collating Headlines *********************")
        for row in range(len(self.dataset_.index)):
            self.dataset_["Combined_News"][row]=  " ".join(str(x) for x in self.dataset_.iloc[row,2:len(self.dataset_.columns)])
        self.data_=self.dataset_[[self.dependentVariableName_,"Combined_News"]]
    def cleanNews(self):
        import re
        import nltk
        from nltk.corpus import stopwords
        self.data_["Combined_News"] = self.data_["Combined_News"].map(lambda x : ' '.join(re.sub("[^a-zA-Z]"," ",x).split()))
        self.data_["Combined_News"] = self.data_["Combined_News"].map(lambda x: x.lower())
        self.data_["Combined_News"] = self.data_["Combined_News"].map(lambda x : ' '.join([w for w in x.split() if w not in stopwords.words('english')]))
    def lemmatizeData(self):
        import nltk
        from nltk.stem import WordNetLemmatizer
        lemmer = WordNetLemmatizer()
        self.data_["Combined_News"] = self.data_["Combined_News"].map(lambda x : ' '.join([lemmer.lemmatize(w) for w in x.split()]))
    def bagOfWord(self,max_features=1000):
        print("**************Performing Bag of Words *********************")
        from sklearn.feature_extraction.text import CountVectorizer
        import pandas as pd
        cv_vectorizer = CountVectorizer(min_df=.015, max_df=.8, max_features=max_features, ngram_range=[1, 3])
        cv = cv_vectorizer.fit_transform(data["Combined_News"])
        print("Bow-CV :", cv.shape)
        dataBow= pd.DataFrame(cv.toarray(), columns=cv_vectorizer.get_feature_names())
        dataBow = pd.concat([self.data_, dataBow], axis = 1)
        dataBow.drop("Combined_News", axis = 1,inplace = True)
        self.data_=dataBow
        print(self.data_.head())
        print(self.data_.tail())
    def TFIDF(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        import pandas as pd
        print("**************Performing TFIDF *********************")
        tfidf_vectorizer = TfidfVectorizer(min_df=.02, max_df=.7, ngram_range=[1,3])
        tfidf = tfidf_vectorizer.fit_transform(self.data_["Combined_News"])
        print("TF:IDF :", tfidf.shape)
        dataTfidf = pd.DataFrame(tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names(), index=self.data_.index)
        dataTfidf = pd.concat([self.data_, dataTfidf], axis = 1)
        dataTfidf.drop("Combined_News", axis = 1,inplace = True)
        self.data_=dataTfidf
        print(self.data_.head())
        print(self.data_.tail())
    def run(self):
        self.joinNews()
        self.cleanNews()
        self.lemmatizeData()
        self.TFIDF()
        return self.data_

     


















def NLP():
    print("**************Welcome to NLP *********************")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas_profiling
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')

    import re
    from bs4 import BeautifulSoup
    
    import nltk
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    print("**************Reading Data *********************")
    df= pd.read_csv("Data.csv",encoding="ISO-8859-1")
    df["Combined_News"]=0
    print("**************Collating Headlines *********************")
    for row in range(len(df.index)):
      df["Combined_News"][row]=  " ".join(str(x) for x in df.iloc[row,2:len(df.columns)])
    
    data=df[["Label","Combined_News"]]
    data["Combined_News"] = data["Combined_News"].map(lambda x : ' '.join(re.sub("[^a-zA-Z]"," ",x).split()))
    data["Combined_News"] = data["Combined_News"].map(lambda x: x.lower())
    data["Combined_News"] = data["Combined_News"].map(lambda x : ' '.join([w for w in x.split() if w not in stopwords.words('english')]))
    print("**************Lemmatizing Data *********************")
    lemmer = WordNetLemmatizer()
    data["Combined_News"] = data["Combined_News"].map(lambda x : ' '.join([lemmer.lemmatize(w) for w in x.split()]))
           
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer

    common_words = get_n_words(data["Combined_News"], "top", 15)
    rare_words = get_n_words(data["Combined_News"], "bottom", 15)

    common_words = dict(common_words)
    names = list(common_words.keys())
    values = list(common_words.values())
    plt.subplots(figsize = (15,10))
    bars = plt.bar(range(len(common_words)),values,tick_label=names)
    plt.title('15 most common words:')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .01, yval)
    plt.show()

    rare_words = dict(rare_words)
    names = list(rare_words.keys())
    values = list(rare_words.values())
    plt.subplots(figsize = (15,10))
    bars = plt.bar(range(len(rare_words)),values,tick_label=names)
    plt.title('15 most rare words:')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .001, yval)
    plt.show()
    print("**************Performing Bag of Words *********************")
    no_features = 1000
    cv_vectorizer = CountVectorizer(min_df=.015, max_df=.8, max_features=no_features, ngram_range=[1, 3])
    cv = cv_vectorizer.fit_transform(data["Combined_News"])
    print("Bow-CV :", cv.shape)
    dataBow= pd.DataFrame(cv.toarray(), columns=cv_vectorizer.get_feature_names())
    
    dataBow = pd.concat([data, dataBow], axis = 1)
    dataBow.drop("Combined_News", axis = 1,inplace = True)
    print(dataBow.head())
    print(dataBow.tail())
    print("**************Performing TFIDF *********************")
    tfidf_vectorizer = TfidfVectorizer(min_df=.02, max_df=.7, ngram_range=[1,3])
    tfidf = tfidf_vectorizer.fit_transform(data["Combined_News"])
    print("TF:IDF :", tfidf.shape)
    dataTfidf = pd.DataFrame(tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names(), index=data.index)
    dataTfidf = pd.concat([data, dataTfidf], axis = 1)
    dataTfidf.drop("Combined_News", axis = 1,inplace = True)
    print(dataTfidf.head())
    print(dataTfidf.tail())
    
    #runModel(dataBow)
    #runModel(dataTfidf)
    tuneModel(dataBow)


def get_n_words(corpus, direction, n):
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    if direction == "top":
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    else:
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=False)
    return words_freq[:n]
