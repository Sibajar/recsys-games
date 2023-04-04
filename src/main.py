
import pandas as pd
from . import clean, nlp
from .recsys import GameRecommender
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    df = pd.read_csv('data/all_games.csv')
    df = clean.clean_data(df)
    df = nlp.apply_preprocess_text(df, "summary")


    recsys = GameRecommender(df, metric = 'cosine')
    recsys.fit()
    
    ##### Vars to Use #####
    # Use index 3
    target = 0
    target_name = df.iloc[target]["name"]

    # Different types of weighting using scores or no scores
    different_types = [None, "meta_score", "user_review"]
    cols_to_show = ["name", "meta_score", "user_review", "platform"]
    
    print(f"Recommendations for: {target_name}")
    line_break = lambda: print("----------------------\n")
    line_break()

    ##### Demonstrate different models #####
    for model_type in different_types:
        
        # Print statements
        print(f"Using {model_type} as param for 'score_type'")
        line_break()

        # Get recs
        recs = recsys.predict(target, model_type)
        recs = df.iloc[recs][cols_to_show]
        print(recs)

