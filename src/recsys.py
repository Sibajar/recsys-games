from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from pandas import DataFrame
import numpy as np
import scipy
from typing import List, Tuple
from . import clean, nlp
from tqdm import tqdm

class GameRecommender:
    """
    Recommends similar items based on a content-based filtering approach.

    Parameters
    ----------
    df : pandas DataFrame
        The dataset containing the item descriptions and features.
    dtm : numpy array
        The document-term matrix of the item descriptions.
    metric : str, optional
        The similarity metric to use, by default "cosine". Possible values are 'cosine', 'euclidean', and 'manhattan'.
    score_type : str, optional
        The score type to use, by default "meta_score". Possible values are 'meta_score', 'user_review', and None.
    """

    def __init__(self, df: pd.DataFrame, num_recs: int = 10, metric: str = "cosine", score_type: str = "meta_score") -> None:
        """
        Initializes the GameRecommender class.

        Parameters
        ----------
        df : pandas DataFrame
            The dataset containing the item descriptions and features.
        num_recs : int, optional
            The number of recommendations to generate, by default 10.
        metric : str, optional
            The similarity metric to use, by default "cosine". Possible values are 'cosine', 'euclidean', and 'manhattan'.
        score_type : str, optional
            The type of rating score to use, by default "meta_score". Possible values are 'meta_score', 'user_review', and None.
        """
        # Assign the input parameters as attributes to the instance
        self.df = df
        self.num_recs = num_recs
        self.metric = metric
        self.score_type = score_type
        self._similarities = None

    def fit(self) -> None:
        """
        Fits the similarity matrix using the specified metric.
        """
        # Create similarity matrix
        dtm = self.get_dtm()
        self._similarities = self.compute_similarity(dtm)

    def predict(self, target_index: int, score_type: str = "meta_score") -> List[int]:
        """
        Recommends items based on similarity to the target item and previously recommended items.

        Parameters
        ----------
        target_index : int
            The index of the target item for which recommendations are being made.
        score_type : str, optional
            The type of rating score to use for recommendations. Default is "meta_score".

        Returns
        -------
        recommendations : List[int]
            A list of recommended item indices.
        """

        # Initialize a list to store the recommended item indices
        recommendations = [target_index]

        # Get the similarity of each item to the target item
        similarities = self._similarities[target_index].copy()

        # If using rating scores then normalize these for use
        if score_type is not None:
            scores = self.get_normalized_scores(score_type)
        else:
            pass

        # Calculate the quality scores for each item and select the top item
        for i in range(self.num_recs):

            # Set the similarities of previously recommended items to 0
            similarities[recommendations] = 0

            # Calculate the quality scores for each item
            # Use ratings if specified, else only consider similarity
            if score_type is not None:
                quality_scores = self.compute_quality_score(similarities, recommendations, scores)
            
            else:
                quality_scores = self.compute_quality_score(similarities, recommendations)

            # Get the index of the item with the highest quality score
            top_item_index = quality_scores.argsort()[-1]

            # Add the top item to the recommendations list and the set of previously recommended items
            recommendations.append(top_item_index)

        # Don't return first recommendation--target index
        return recommendations[1:]
    
    def get_normalized_scores(self, score_type: str = "meta_score") -> np.ndarray:
        """
        Normalize the scores in either the meta_score or user_review column.

        Parameters
        ----------
        score_type : str, optional
            The type of score to normalize. Either "meta_score" or "user_review".
            Default is "meta_score".

        Returns
        -------
        scores : numpy array
            The normalized scores as a numpy array.
        """
        # Use either the meta_score or user_review column for scores
        if score_type == "meta_score":
            scores = self.df["meta_score"].values.reshape(-1, 1)
        
        elif score_type == "user_review": 
            scores = self.df["user_review"].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        scores = scaler.fit_transform(scores)
        
        return scores[:,0]
    
    def compute_dissimilarities(self, recommendations: List[int]) -> np.array:
        """
        Computes the dissimilarities of the candidate items with respect to the previously recommended items.

        Parameters
        ----------
        recommendations : List[int]
            A list of the curernt recommended item indices.

        Returns
        -------
        dissimilarities : numpy array
            The average dissimilarities of the candidate items with respect to the previously recommended items.
        """
        # Calculate the dissimilarities of the candidate items with respect to the previously recommended items
        dissimilarities = 1 - self._similarities[list(recommendations)].copy()
        
        # Floating point error can make numbers go below zero, fix like this--
        dissimilarities = np.clip(dissimilarities, 0, None)

        # Return the average dissimilarities
        return dissimilarities.mean(axis=0)
    
    def compute_quality_score(self, sim_matrix: np.array, recommendations: List[int], scores = None) -> np.array:
        """
        Compute the quality scores for each item in the similarity matrix.

        Parameters
        ----------
        sim_matrix : numpy array
            The similarity matrix as a numpy array.
        recommendations : List[int]
            A list of the current recommended item indices.
        scores : numpy array, optional
            An array of scores to be used in the quality score calculation. If not provided, only similarity is considered.

        Returns
        -------
        quality_scores : numpy array
            The quality scores for each item in the similarity matrix as a numpy array.
        """
        # Get dissimilarities
        sim_matrix = sim_matrix.copy()
        ds_matrix = self.compute_dissimilarities(recommendations)

        # Compute quality score
        if scores is None:
            quality_scores = sim_matrix * ds_matrix
        else:
            quality_scores = sim_matrix * ds_matrix * scores
            
        return quality_scores


    def compute_similarity(self, dtm) -> np.ndarray:
        """
        Compute the similarity matrix using a specified similarity measure.

        Parameters
        ----------
        dtm : sp.spmatrix
            The document-term matrix.

        Returns
        -------
        np.ndarray
            The similarity matrix as a numpy array.

        Raises
        ------
        ValueError
            If an invalid similarity metric is specified.

        Notes
        -----
        The similarity measure is determined by the `metric` attribute of the class.

        Possible values for `metric` are:

        - 'cosine'
        - 'euclidean'
        - 'manhattan'
        """
        # Compute the similarity matrix using the specified metric
        if self.metric == 'cosine':
            sim_matrix = cosine_similarity(dtm)
        elif self.metric == 'euclidean':
            sim_matrix = 1 / (1 + euclidean_distances(dtm))
        elif self.metric == 'manhattan':
            sim_matrix = 1 / (1 + manhattan_distances(dtm))
        else:
            raise ValueError(f"Invalid metric: {self.metric}. Possible values are 'cosine', 'euclidean', and 'manhattan'.")

        return sim_matrix
    
    def get_dtm(self):
        """
        Creates a document-term matrix from the "summary" column of the DataFrame using CountVectorizer.

        Returns
        -------
        dtm : scipy.sparse.csr_matrix
            A document-term matrix in sparse CSR format, where each row represents a document and each column represents
            a term.
        """
        vectorizer = CountVectorizer()
        dtm = vectorizer.fit_transform(self.df["summary"])
        
        return dtm

