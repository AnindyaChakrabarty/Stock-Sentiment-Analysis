
from Input import Input ,MongoDB, Report
from Classifier import Classifier, Utility
from NaturalLanguageProcessor import NaturalLanguageProcessor

inp=Input("StockSentimentAnalysis","RawData","Label")
nlp=NaturalLanguageProcessor(inp)
nlp.run()