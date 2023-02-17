import pandas as pd
import re
import os
from nltk.util import bigrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm import MLE
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Laplace
import time
from nltk import MLEProbDist
import csv
from joblib import dump, load

class Normalize():
    _data_frame = None
    _comments = None
    PUNCTUATIONS = None
    _corpus = dict()
    

    def __init__(self, file_path) -> None:
        #checks the file on which the data is present upon
        try:
            self._data_frame = pd.read_csv(file_path)

            """
            We only need the following columns,
            1) comment_text <str>: comment made by an user.
            2) toxic <0,1>: 0 if the comment made by a user is not toxic, 1 otherwise. 
            """
            self._data_frame = self._data_frame.drop(columns = ["id", "obscene", "threat", "severe_toxic", "insult", "identity_hate"])

            #setting the class variables here:
            self.PUNCTUATIONS = "[\#\$\%\&\'\(\)\*\+\,\ \-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~\"]"
        except Exception as e:
            print("[ERR] The following error occured while trying to read data from file: " + str(e))

    def create_list(self) -> None:
        """
        Coverts the data frame into a two dimensional List, [ [comment_text <str>, toxic <0,1>] ]
        """
        try:
            self._comments =  self._data_frame.values.tolist()
            print("[INFO] Created corpus for the training file provided.")
        except Exception as e:
            print("[ERR] The following error occured while trying to create a corpus for entire data: " + str(e))
       
    def lower(self) -> None:
        """
        Normailize.lower(), converts every tuple[0] i.e. comment_text into lower case for consistency.
        """
        try:
            for row in self._comments:
                row[0] = row[0].lower()
        except Exception as e:
            print("The following error occured while, coverting the comments into lower case : " + str(e))

    def stripper(self, comment) -> str():
        """
        Normalize.remove_punctuations requires an additional method for iteration and discarding special characters,
        this method supports this functionality.
        """
        try:
            comment = re.sub("[\']", "", comment)
            comment = re.findall("[a-zA-Z]+",comment)
            comment = " ".join(comment)
            return comment
        except Exception as e:
            pass

    def remove_punctuations(self) -> None:
        """
        Normalize.remove_punctuations(), aims to remove all the special characters that are used within comment_text columns of the Data_frame.
        """
        try:
            for row in self._comments:
                row[0] = self.stripper(row[0])
        except Exception as e:
            print("[ERR] The following error occured while, discarding Punctuations: " + str(e))

    def build_corpus(self) -> None:
        """
        LISTS ARE SLOW Af! -> The goal is to have a key value pairs like -> < key : comment_text <str>, toxic <int> >
        """
        try: 
            for comment, toxic_check in self._comments:
                if self._corpus.get(comment, None) is None:
                    self._corpus[comment] = toxic_check
                else:
                    continue
        except Exception as e:
            print("[ERR] The following error occured while trying to create Corpus: " + str(e))

    def get_data(self) -> list():
        self.create_list()
        self.lower()
        self.remove_punctuations()
        self.build_corpus()
        return self._corpus

class LanguageModels():
    """
    This class aims to implement the two methods, that are required for the task.
    1) train_LM(path_to_train_file: <str>): that trains a Language Model using a Bi gram, for specific file.
    2) test_LM(path_to_test_file: <str>, LM_model: <object>): that tests the previously trained model. 
    """
    _corpus = None
    _train_Data = None
    _toxic_corpus = None
    _non_toxic_corpus = None
    _toxic_bigrams = None
    _non_toxic_bigrams = None
    LM_full = None
    LM_not = None
    LM_toxic = None

    def setter(self, path_to_train_file) -> None:
        """
        setter creates the corpus that is needed to fit these three language models.
        """
        self._corpus = Normalize(path_to_train_file).get_data()
        self._toxic_corpus = list()
        self._non_toxic_corpus = list()
        self._train_Data = list()
        self._bigrams = list()
        self._toxic_bigrams = list()
        self._non_toxic_bigrams = list()

        #we tokenize the sentences here:
        for sentence, toxicity in self._corpus.items():

            #lets add padding:
            sentence = list(pad_both_ends(sentence.split(), n=2))
            self._train_Data.append(sentence)

            #if the comment is toxic, then it has to be added to the toxic corpus, or non_toxic corpus otherwise.
            if toxicity == 1 :
                self._toxic_corpus.append(sentence)
            else:
                self._non_toxic_corpus.append(sentence)

        
            

    def save_model(self, model, model_name) -> None:
        """
        This method, saves a fitted model as it is very computationally expensive to fit the model with large
        training data, every iteration/run.
        """
        try:
            dump(model, (model_name+".joblib"))
            print("[INFO] Saved the model for model persistence.")
        except Exception as e:
            print("[ERR] The following error occured while trying to save the fitted model: " + str(e))
    
    def load_model(self, model_name):
        """
        This method loads a model, if it does not exist returns a false boolean value.
        """
        try:
            print("[PROCESS] Loading the '%s' model, as it already has been trained, and saved previously."%(model_name))
            if model_name  == "Full_LM":
                self.LM_full = load(model_name+".joblib")
            elif model_name == "LM_not":
                self.LM_not = load(model_name+".joblib")
            elif model_name == "LM_toxic":
                self.LM_toxic = load(model_name+".joblib")
            
            return True
        except:
            return False

    def get_bigrams(self, list_of_sentences) -> list():
        try:
            bigramms = list()
            for sentence in list_of_sentences:
                bigramms.extend(list(bigrams(sentence)))
            return bigramms
        except Exception as e:
            print("[ERR] The following error occured while trying to create Bigrams!: "+ str(e))

    def train_LM(self, path_to_train_file) -> Laplace(2):
        """
        train_lm method, takes a string argument which is filepath to the training data,
        if the model(s) that are supposed to be trained, already exist as a result of Model persistence, 
        we load them.
        or create and fit new models and also save them otherwise.
        """
        try:
            self.setter(path_to_train_file) #opening the training data csv file.
            """self._bigrams = self.get_bigrams(self._train_Data) #creating bigrams for all comments
            print("[INFO] Created bigrams for entire corpus!")
            self._toxic_bigrams = self.get_bigrams(self._toxic_corpus) #creating bigrams for only comments with toxic label as 1.
            print("[INFO] Created toxic comment based bigrams")
            self._non_toxic_bigrams = self.get_bigrams(self._non_toxic_corpus) #creating bigrams for comments with toxic label as 0.
            print("[INFO] Created non toxic comment based bigrams")"""
            
            
            if self.load_model("Full_LM") is False:
                print("[UPDATE] There is no pre-trained model for complete corpus, creating one right now.")
                train, vocab = padded_everygram_pipeline(2, self._train_Data)
                self.LM_full = Laplace(2)

                #let's apply laplace smoothing for -inf scores when we use ln to determine score
                # and 0.000 underflow when we use regular score method.
                print("[PROCESS] Fitting a Language Model on entire data of Comment_text, please wait this might take some time.")
                self.LM_full.fit( train, vocab)
                self.save_model(self.LM_full, "Full_LM")
                

            if self.load_model("LM_not") is False:
                #the Language model does not exist, hence we are to create one.
                print("[UPDATE] There is no pre-trained model for Non_toxic_corpus, creating one right now.")
                train, vocab = padded_everygram_pipeline(2, self._non_toxic_corpus)
                self.LM_not = Laplace(2) #create an instance of the model.
                print("[PROCESS] Fitting a Language Model on non_toxic_comments corpus, please wait this might take some time.")
                self.LM_not.fit(train, vocab) #train the model.
                self.save_model(self.LM_not, "LM_not") #Model persistence.
            
            if self.load_model("LM_toxic") is False:
                #the Language model does not exist, hence we are to create one.
                print("[UPDATE] There is no pre-trained model for toxic_corpus, creating one right now.")
                train, vocab = padded_everygram_pipeline(2, self._toxic_corpus)
                self.LM_toxic = Laplace(2) #create an instance of the model.
                print("[PROCESS] Fitting a Language Model on toxic_comments corpus, please wait this might take some time.")
                self.LM_toxic.fit(train, vocab) #train the model.
                self.save_model(self.LM_toxic, "LM_toxic") #Model persistence.

            #lets present vocab of each model.                   
            print("[INFO] LM_full stats: "+ str(self.LM_full.vocab))
            print("[INFO] LM_not stats: "+ str(self.LM_not.vocab))
            print("[INFO] LM_toxic stats: "+ str(self.LM_toxic.vocab))
            return [self.LM_not, self.LM_not, self.LM_toxic]
        except Exception as e:
            print("[ERR] The following error occured while trying to Train the Language model: " + str(e))
    
    def test(self) -> None:
        """
        this is a default test method to test the newly created Language model on the Bigrams.
        """
        print("The model is trained on " + str(len(self._train_Data)) + " sentences")
        print("We have a total of " + str(len(self._bigrams)) + "Bi grams")
        print(self.LM_full.counts)
        print("Starting Language model testing now! this might take a long while!")
        counter = 0
        summation = 0
        for bigramm in self._bigrams:
            value = self.LM_full.score(bigramm[0], [bigramm[1]])
            print("Score for '%s' being followed by '%s' is %f"%(bigramm[0], bigramm[1], value))
            time.sleep(0.05)
            os.system("cls")
            counter  = counter + 1
            summation = summation + value

        counter = 0
        summation = 0
        for toxic_bigram in self._toxic_bigrams:
            value = self.LM_full.score(toxic_bigram[0], [toxic_bigram[1]])
            print("Score for '%s' being followed by '%s' is %f"%(toxic_bigram[0], toxic_bigram[1], toxic_bigram))
            os.system("cls")
            counter  = counter + 1
            summation = summation + value
        print("A total of %d scores have been determined and the average score is %f "%(counter, (summation/counter)))
           
    def create_output_file(self) -> None:
        """
        Creates a output file, which in this case is a CSV file
        """
        try:
            self._file_handler = open("output.csv", 'w', encoding="UTF-8")
            self._type_writer = csv.writer(self._file_handler)
        except Exception as e:
            print("The following error occured while trying to create an output file: " + str(e))


    def writeResult(self, comment_text, score_value):
        """
        Writes an individual tuple/new row, to the newly created csv file.
        """
        try:
            self._type_writer.writerow([comment_text, score_value])
        except Exception as e:
            print("The following error occured while trying to write a tuple to the csv file: " + str(e))

    def test_LM(self, path_to_test_file, LM_model) -> str():
        pass


obj = LanguageModels()
LMs=obj.train_LM("trainingData/train.csv")
big = ("you sir are my hero".split())
print(big[0], big[1])
print("Log Score of comment on LM_full: %f, and normal score on LM_full: %f"%(LMs[0].logscore(big),LMs[0].score(big)))
print("Log Score of comment on LM_not: %f, and normal score on LM_not: %f"%(LMs[1].logscore(big[0],big[1]),LMs[1].score(big[0],big[1])))
print("Log Score of comment on LM_toxic: %f, and normal score on LM_toxic: %f"%(LMs[2].logscore(big[0],big[1]),LMs[2].score(big[0],big[1])))
