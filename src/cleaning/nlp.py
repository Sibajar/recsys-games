import pandas as pd
from pandas import DataFrame
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm.auto import tqdm

def apply_preprocess_text(df: pd.DataFrame, column: str) -> DataFrame:
    '''
    Applies a function in pandas that preprocesses the text of a column.

    Args:
        df (pd.DataFrame): The pandas DataFrame to apply the function to.
        column (str): The name of the column to apply the function to.

    Returns:
        pd.DataFrame: Dataframe where the provided column has had the text preprocessed
    '''
    # Ensure column is string type
    df[column] = df[column].astype(str)

    # Use tqdm to display a progress bar during the apply operation
    tqdm.pandas()

    # Apply the function to the specified column in the DataFrame
    df[column] = df[column].progress_apply(preprocess_text)

    return df

# Define the NLP preprocessing functions
def preprocess_text(text: str) -> str:
    '''
    Applies tokenization, stop word removal, and stemming to a text string.

    Args:
        text (str): The input text string to preprocess.

    Returns:
        str: The preprocessed text string with stop words removed and words stemmed.
    '''
    # Tokenize the text into words
    words = word_tokenize(text.lower())

    # Remove stop words from the text
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stem the words in the text
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Join the words back into a single string
    text = ' '.join(words)

    return text

if __name__ == "__main__":
    df = pd.read_csv("data/all_games.csv")
    df = apply_preprocess_text(df, "summary")
    
   