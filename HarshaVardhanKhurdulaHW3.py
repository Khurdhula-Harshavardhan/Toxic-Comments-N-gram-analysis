import pandas as pd
import re

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
            print("The following error occured while trying to read data from file: " + str(e))

    def create_list(self) -> None:
        """
        Coverts the data frame into a two dimensional List, [ [comment_text <str>, toxic <0,1>] ]
        """
        self._comments =  self._data_frame.values.tolist()
        print("created list")
       
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
            print("The following error occured while, discarding Punctuations: " + str(e))

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
            print("The following error occured while trying to create Corpus: " + str(e))

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

    def train_LM(self, path_to_train_file) -> object():
        print("Going to fetch data")
        values = Normalize(path_to_train_file).get_data()
        print(values)


    def test_LM(self, path_to_test_file, LM_model) -> str():
        pass


LanguageModels().train_LM("trainingData/train.csv")