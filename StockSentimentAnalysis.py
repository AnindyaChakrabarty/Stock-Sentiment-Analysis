
from Input import Input ,MongoDB, Report
from Classifier import Classifier, Utility,NaturalLanguageProcessor


inp=Input("StockSentimentAnalysis","RawData","Label")
clf=Classifier(inp)
clf.compareModel()

