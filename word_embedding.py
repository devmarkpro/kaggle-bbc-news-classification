import pandas as pd
import numpy as np
from gensim.models import KeyedVectors


class BBCNewsWordEmbedding:
    def __init__(self, df: pd.DataFrame, text_column: str, id_column: str):
        """
        Initialize the word embedding class with a DataFrame and the text column name.
        This class is designed to handle word embeddings using GloVe vectors.
        Args:
            df (pd.DataFrame): The DataFrame containing the text data.
            text_column (str): The name of the column containing the text data.
        """
        self.df = df
        self.text_column = text_column
        self.id_column = id_column
        self.word_vectors = KeyedVectors.load("glove-wiki-gigaword-100")

    def _load_glove_embeddings(self, glove_file_path: str) -> dict:
        """
        Load GloVe embeddings from a file.

        Args:
            glove_file_path (str): The path to the GloVe file.

        Returns:
            dict: A dictionary containing the GloVe embeddings.
        """
        embedding_dict = {}
        with open(glove_file_path, 'r', encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embedding_dict[word] = vector
        return embedding_dict

