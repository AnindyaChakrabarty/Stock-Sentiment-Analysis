
# MACHINE LEARNING PROJECT 
# Author : Anindya Chakrabarty 

from Input import Input ,MongoDB, Report
class Utility:
    
    
    def makeDir(self,parentDirectory, dirName):
        import os
        new_dir = os.path.join(parentDirectory, dirName+'\\')
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)
        return new_dir
    def SetLogger(self):
        import logging
        logger = logging.getLogger("Classifier")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        file_handler = logging.FileHandler('LogFile.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger
    def stopwatchStart(self):
        import time
        self.start_=time.perf_counter()
    def stopwatchStop(self):
        import time
        self.finish_=time.perf_counter()
    def showTime(self):
        import time
        import numpy as np
        print(f'This operation has finished in {np.round(self.finish_-self.start_,2)}  second(s)')

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
    def splitData(self):
        from sklearn.model_selection import  train_test_split
        from collections import Counter
        Y = self.data_[self.input_.dependentVariableName_]
        X=self.data_.drop(self.input_.dependentVariableName_,axis=1)
        self.X_train_, self.X_test_, self.Y_train_, self.Y_test_ = train_test_split(X, Y, train_size = 0.85, random_state = 21)
        print('Original  Training Dataset Shape {}'.format(Counter(self.Y_train_)))
        print('Original  Testing Dataset Shape {}'.format(Counter(self.Y_test_)))

    def isImbalence(self,threshold):
        imbl=self.data_[self.input_.dependentVariableName_].value_counts()
        if (imbl[1]/imbl[0]<threshold or imbl[0]/imbl[1]<threshold):
            print(f'We have imbalence dataset with count of 1 in Total Data : {imbl[1]} and count of 0 in Total Data : {imbl[0]}')
        else:
            print(f'We do not have imbalence dataset with count of 1 in Total Data : {imbl[1]} and count of 0 in Total Data : {imbl[0]}')
        return (imbl[1]/imbl[0]<threshold or imbl[0]/imbl[1]<threshold)

    def overSampling(self,ratio):
        from imblearn.over_sampling import RandomOverSampler
        from collections import Counter
        os=RandomOverSampler(ratio)
        self.X_train_, self.Y_train_ =os.fit_resample(self.X_train_, self.Y_train_)
        print('Over Sampled  Training Dataset Shape {}'.format(Counter(self.Y_train_)))
    def SMOTE(self,k):
        from imblearn.over_sampling import SMOTE
        from collections import Counter
        smote=SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=100)
        self.X_train_, self.Y_train_ =smote.fit_resample(self.X_train_, self.Y_train_)
        print('SMOTE  Training Dataset Shape {}'.format(Counter(self.Y_train_)))

    def handlingImbalanceData(self):
        if (self.isImbalence(0.5)):
            #self.overSampling(1)
            self.SMOTE(1)
        else:
            print("Data set is balanced and hence no changes made")
    def run(self):
        self.joinNews()
        self.cleanNews()
        self.lemmatizeData()
        self.TFIDF()
        self.splitData()
        self.handlingImbalanceData()

   
          
class Classifier:

    def __init__(self,input):
        import pandas as pd
        import os
        import numpy as np
        pd.set_option('display.max_columns', None)
        self.input_=input
        self.bestModels_={}
        self.NLP_=NaturalLanguageProcessor(self.input_)
        self.NLP_.run()
        self.NLP_.logger_.debug("Ending Natural Language Processing")
        self.util_=Utility()
        self.parentDirectory_ = os.path.dirname(os.getcwd())
        self.Model_Dir_= self.util_.makeDir(self.parentDirectory_,"Machine Learning Models")
    
    def getHyperParameters(self):
         
         self.grid_params_NaiveBayesClassifier_ = {'alpha' : [1,2,3]}
         self.grid_params_RandomForestClassifier_ = {'n_estimators' : [100,200,300,400,500],'max_depth' : [10, 7, 5, 3],'criterion' : ['entropy', 'gini']}
         self.grid_params_XGBClassifier_={'n_estimators' : [100,200,300],'learning_rate' : [1.0, 0.1, 0.05],'max_depth':[2,3,6],'min_child_weight':[1,2]}
         self.grid_params_AdaBoostClassifier_={'n_estimators' : [100,200,300],'learning_rate' : [1.0, 0.1, 0.05]}
         self.grid_params_GradientBoostingClassifier_={'n_estimators' : [100,200,300],'learning_rate' : [1.0, 0.1, 0.05],'max_depth':[2,3,6]}
         self.grid_params_KernelSupportVectorMachine_=[{'kernel': ['rbf','sigmoid','linear'], 'gamma': [1e-2]}]
         self.grid_params_LogisticRegression_= {'C' : [0.0001, 0.01, 0.05, 0.2, 1],'penalty' : ['l1', 'l2']} 
         self.grid_params_ExtraTreesClassifier_={'n_estimators' : [100,200,300,400,500],'max_depth' : [10, 7, 5, 3],'criterion' : ['entropy', 'gini']}
            
    def tuneNaiveBayesClassifier(self):
        
        import numpy as np
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        print("**************Tuning Naive Bayes Classifier*********************")
        self.classifier_ = MultinomialNB()
        grid_object = GridSearchCV(estimator =self.classifier_, param_grid = self.grid_params_NaiveBayesClassifier_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.NLP_.X_train_,self.NLP_.Y_train_)
        
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Naive Bayes Classifier':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.NLP_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.NLP_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.NLP_.X_test_, self.NLP_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Naive Bayes Classifier')
   
    def tuneRandomForestClassifier(self):
        
        import numpy as np
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        print("**************Tuning Random Forest Classifier*********************")
        
        self.classifier_ = RandomForestClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_RandomForestClassifier_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.NLP_.X_train_, self.NLP_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Random Forest Classifier':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.NLP_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.NLP_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.NLP_.X_test_, self.NLP_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Random Forest Classifier')
        
    def tuneXGBClassifier(self):
        
        import numpy as np
        from xgboost import XGBClassifier
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        print("**************Tuning XG Boost Classifier*********************")
        
        self.classifier_=XGBClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_XGBClassifier_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.NLP_.X_train_, self.NLP_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'XG Boost Classifier':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.NLP_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.NLP_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.NLP_.X_test_, self.NLP_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned XG Boost Classifier')
    
    def tuneAdaBoostClassifier(self):
        
        import numpy as np
        from sklearn.ensemble import  AdaBoostClassifier
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        print("**************Tuning Ada Boost Classifier*********************")
       
        self.classifier_ = AdaBoostClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_AdaBoostClassifier_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.NLP_.X_train_, self.NLP_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Ada Boost Classifier':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.NLP_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.NLP_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.NLP_.X_test_, self.NLP_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
       
        return self.getResult('Tuned AdaBoost Classifier')
   
    def tuneGradientBoostingClassifier(self):
       
        import numpy as np
        from sklearn.ensemble import  GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        print("**************Tuning Grdient Boosting Classifier*********************")
       
        self.classifier_ = GradientBoostingClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_GradientBoostingClassifier_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.NLP_.X_train_, self.NLP_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Grdient Boosting Classifier':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.NLP_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.NLP_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.NLP_.X_test_, self.NLP_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Gradient Boosting Classifier')
        
    def tuneKernelSupportVectorMachine(self):
        
        import numpy as np
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        
        print("**************Tuning Kernel Support Vector Machine*********************")
           
        self.classifier_=SVC(probability=True)
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_KernelSupportVectorMachine_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.NLP_.X_train_, self.NLP_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Kernel Support Vector Machine':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.NLP_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.NLP_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.NLP_.X_test_, self.NLP_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Support Vector Machine')
    
    def tuneLogisticRegression(self):
        
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        
        print("**************Tuning Logistic Regression*********************")
        
        self.classifier_=LogisticRegression()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_LogisticRegression_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.NLP_.X_train_, self.NLP_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Logistic Regression':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.NLP_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.NLP_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.NLP_.X_test_, self.NLP_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Logistic Regression')

    def tuneExtraTreesClassifier(self):
        
        import numpy as np
        from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,cross_val_score
        from sklearn.ensemble import ExtraTreesClassifier
        print("**************Tuning Extra Trees Classifier*********************")
        
        self.classifier_ = ExtraTreesClassifier()
        grid_object = GridSearchCV(estimator = self.classifier_, param_grid = self.grid_params_ExtraTreesClassifier_, scoring = 'accuracy', cv = 10, n_jobs = -1)
        grid_object.fit(self.NLP_.X_train_, self.NLP_.Y_train_)
        print("Best Parameters : ", grid_object.best_params_)
        print("Best_ROC-AUC : ", round(grid_object.best_score_ * 100, 2))
        print("Best model : ", grid_object.best_estimator_)
        self.bestModels_.update({'Extra Trees Classifier':grid_object.best_estimator_})
        self.Y_pred_ = grid_object.best_estimator_.predict(self.NLP_.X_test_)
        self.probs_ = grid_object.best_estimator_.predict_proba(self.NLP_.X_test_)
        kfold = KFold(n_splits=10, random_state=25, shuffle=True)
        results = cross_val_score(grid_object.best_estimator_, self.NLP_.X_test_, self.NLP_.Y_test_, cv=kfold)
        results = results * 100
        results = np.round(results,2)
        print("Cross Validation Accuracy : ", round(results.mean(), 2))
        print("Cross Validation Accuracy in every fold : ", results)
        
        return self.getResult('Tuned Extra Trees Classifier')      
        
    def compareModel(self):
        self.report_=Report()
        self.getHyperParameters()
        
        self.NLP_.logger_.debug("Tuning Logistic Regression ")
        
        self.NLP_.util_.stopwatchStart()
        lst=self.tuneLogisticRegression()
        self.report_.insertResult(lst)
        self.NLP_.util_.stopwatchStop()
        self.NLP_.util_.showTime()
        self.NLP_.logger_.debug("Tuning  Extra Trees Classifier ")
        self.NLP_.util_.stopwatchStart()
        lst=self.tuneExtraTreesClassifier()
        self.report_.insertResult(lst)
        self.NLP_.util_.stopwatchStop()
        self.NLP_.util_.showTime()
        self.NLP_.logger_.debug("Tuning Naive Bayes Classifier ")
        self.NLP_.util_.stopwatchStart()
        lst=self.tuneNaiveBayesClassifier()
        self.report_.insertResult(lst)
        self.NLP_.util_.stopwatchStop()
        self.NLP_.util_.showTime()
        self.NLP_.logger_.debug("Tuning Random Forest Classifier ")
        self.NLP_.util_.stopwatchStart()
        lst=self.tuneRandomForestClassifier()
        self.report_.insertResult(lst)
        self.NLP_.util_.stopwatchStop()
        self.NLP_.util_.showTime()
        self.NLP_.logger_.debug("Tuning AdaBoost Classifier ")
        self.NLP_.util_.stopwatchStart()
        lst=self.tuneAdaBoostClassifier()
        self.report_.insertResult(lst)
        self.NLP_.util_.stopwatchStop()
        self.NLP_.util_.showTime()
        self.NLP_.logger_.debug("Tuning Gradient Boosting Classifier ")
        self.NLP_.util_.stopwatchStart()
        lst=self.tuneGradientBoostingClassifier()
        self.report_.insertResult(lst)
        self.NLP_.util_.stopwatchStop()
        self.NLP_.util_.showTime()
        self.NLP_.logger_.debug("Tuning XGBClassifier ")
        self.NLP_.util_.stopwatchStart()
        lst=self.tuneXGBClassifier()
        self.report_.insertResult(lst)
        self.NLP_.util_.stopwatchStop()
        self.NLP_.util_.showTime()
        self.NLP_.logger_.debug("Tuning  Support Vector Machine ")
        self.NLP_.util_.stopwatchStart()
        lst=self.tuneKernelSupportVectorMachine()
        self.report_.insertResult(lst)
        self.NLP_.util_.stopwatchStop()
        self.NLP_.util_.showTime()
        self.report_.report_= self.report_.report_.sort_values(["Accuracy"], ascending =False)
        print(self.report_.report_)
        self.input_.writeMongoData(self.report_.report_,"TunedModelComparisonReport")
        self.NLP_.logger_.debug("Ending Model Calibration ")

    def compareModel1(self):
        self.getHyperParameters()
        self.algoCall_={"Tuning Naive Bayes Classifier ":self.tuneNaiveBayesClassifier(),
                        "Tuning Random Forest Classifier":self.tuneRandomForestClassifier(),
                        "Tuning AdaBoost Classifier":self.tuneAdaBoostClassifier(),
                        "Tuning Gradient Boosting Classifier":self.tuneGradientBoostingClassifier(),
                        "Tuning XGBClassifier":self.tuneXGBClassifier(),
                        "Tuning Support Vector Machine":self.tuneKernelSupportVectorMachine()}
        self.report_=Report()               
        for key in self.algoCall_:
            self.report_.insertResult(self.algoCall_[key])    
        self.report_.report_= self.report_.report_.sort_values(["Accuracy"], ascending =False)
        self.NLP_.logger_.debug("Ending Program. Thanks for your visit ")
    
    def getResult(self,algoName):
        from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, precision_score, recall_score,  accuracy_score, precision_recall_curve
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        report=[algoName]
        print("\n", "Confusion Matrix")
        cm = confusion_matrix(self.NLP_.Y_test_, self.Y_pred_)
        print("\n", cm, "\n")
        #sns.heatmap(cm, square=True, annot=True, cbar=False, fmt = 'g', cmap='RdBu',
                #xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
        #plt.xlabel('true label')
        #plt.ylabel('predicted label')
        #plt.show()
        print("\n", "Classification Report", "\n")
        print(classification_report(self.NLP_.Y_test_, self.Y_pred_))
        print("Overall Accuracy : ", round(accuracy_score(self.NLP_.Y_test_, self.Y_pred_) * 100, 2))
        print("Precision Score : ", round(precision_score(self.NLP_.Y_test_, self.Y_pred_, average='binary') * 100, 2))
        print("Recall Score : ", round(recall_score(self.NLP_.Y_test_, self.Y_pred_, average='binary') * 100, 2))
        preds = self.probs_[:,1] # this is the probability for 1, column 0 has probability for 0. Prob(0) + Prob(1) = 1
        fpr, tpr, threshold = roc_curve(self.NLP_.Y_test_, preds)
        roc_auc = auc(fpr, tpr)
        print("AUC : ", round(roc_auc * 100, 2), "\n")
        report.append(round(accuracy_score(self.NLP_.Y_test_, self.Y_pred_) * 100, 2))
        report.append(round(precision_score(self.NLP_.Y_test_, self.Y_pred_, average='binary') * 100, 2))
        report.append(round(recall_score(self.NLP_.Y_test_, self.Y_pred_, average='binary') * 100, 2))
        report.append(round(roc_auc * 100, 2))

        plt.figure()
        plt.plot(fpr, tpr, label='Best Model on Test Data (area = %0.2f)' % roc_auc)
        plt.plot([0.0, 1.0], [0, 1],'r--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('RoC-AUC on Test Data')
        plt.legend(loc="lower right")
        #plt.savefig('Log_ROC')
        #plt.show()
        return report

    def predict(self,newData):
        self.NLP_.logger_.debug("Starting Model prediction ")
        import pandas as pd
        import numpy as np
        self.predictionReport_=Report()
        self.newInput_=Input(self.input_.databaseName_,newData,self.input_.dependentVariableName_)
        self.newDataSet_=self.newInput_.readMongoData()
        self.NLP_.Header_.remove(self.input_.dependentVariableName_)     
        self.newData_=self.newDataSet_[self.NLP_.Header_].copy()
        self.newData_ = pd.get_dummies(self.newData_,drop_first=False)
        self.newData_=self.newData_.reindex(columns=list(self.NLP_.X_train_.columns),fill_value=0)
        
        for key in self.bestModels_:
            self.predictionReport_.insertPredictionResults([key,int(self.bestModels_[key].predict(self.newData_)),int(np.round(self.bestModels_[key].predict_proba(self.newData_)[0][0],2)*100),int(np.round(self.bestModels_[key].predict_proba(self.newData_)[0][1],2)*100)])              
        print(self.predictionReport_.predictionReport_)
        self.NLP_.logger_.debug("Ending Model prediction. Good Bye")
        
   
        

    
  












 





        

        
    




       




