# Import libraries
import pandas as pd
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin


# Define utility functions
def tokenize(text):
    """ Custom text tokenizer
    
        Processes text in three steps:
        - Converts text to lower case & splits string into tokens
        - Lemmatizes tokens
        - Removes stopwords
        
        Args: 
            text (str): Input text
        Returns:
            str: Tokens
    """
    # Remove all non-alpha-numeric characters and tokenize text
    tokens = word_tokenize(re.sub('[^a-z0-9]', ' ', text.lower().strip()))
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]
    
    # Remove stopwords
    clean_tokens = [tok for tok in clean_tokens if tok not in stopwords.words("english")]
    
    return clean_tokens


# Define custom transformer classes which can be used in feature unions
# as part of a sklearn machine learning pipeline
class CharacterCount(BaseEstimator, TransformerMixin):
    """ Custom sklearn transformer class to count number of characters
    """
    def character_count(self, text):
        """ Counts the number of characters in string
        
            Args: 
                text (str): Input text
            Returns:
                int: Number of characters
        """
        return len(text)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_new = pd.Series(X).apply(self.character_count)
        return pd.DataFrame(X_new.astype(str).astype(int)).to_sparse()
    
    
class WordCount(BaseEstimator, TransformerMixin):
    """ Custom sklearn transformer class to count number of words
    """
    def word_count(self, text):
        """ Counts the number of stopwords in string
        
            Args: 
                text (str): Input text
            Returns:
                int: Number of words
        """
        tokens = nltk.word_tokenize(re.sub('[^a-z]', ' ', text.lower()))
        return len(tokens)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_new = pd.Series(X).apply(self.word_count)
        return pd.DataFrame(X_new.astype(str).astype(int)).to_sparse()
    
    
class StopwordCount(BaseEstimator, TransformerMixin):
    """ Custom sklearn transformer class to count number of stopwords
    """
    def stopword_count(self, text):
        """ Counts the number of stopwords in string
        
            Args: 
                text (str): Input text
            Returns:
                int: Number of stopwords
        """
        tokens = nltk.word_tokenize(re.sub('[^a-z]', ' ', text.lower()))
        stopword_tokens = [tok for tok in tokens if tok in stopwords.words("english")]
        return len(stopword_tokens)

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_new = pd.Series(X).apply(self.stopword_count)
        return pd.DataFrame(X_new.astype(str).astype(int)).to_sparse()
    
    
class StartingVerb(BaseEstimator, TransformerMixin):
    """ Custom sklearn transformer class to check if first word is a verb
    """
    def starting_verb(self, text):
        """ Checks if first word in text is a verb
        
            Args: 
                text (str): Input text
            Returns:
                bool: True if first word is verb else False
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) >= 1:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged).to_sparse()