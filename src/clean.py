import pandas as pd
from pandas import DataFrame

# Lists containing different console types
CONSOLE_DICT = {
    "Xbox": ["Xbox", "Xbox 360", "Xbox One", "Xbox Series X"],
    "Nintendo": ["3DS", "DS", "Game Boy Advance",  "GameCube", "Nintendo 64","Switch", "Wii", "Wii U"],
    "Sega": ["Dreamcast"],
    "PlayStation": ["PlayStation", "PlayStation 2", "PlayStation 3","PlayStation 4", "PlayStation 5", "PlayStation Vita", "PSP"],
    "PC": ["PC", "Stadia"]
    }

def clean_data(df: DataFrame) -> DataFrame:
    '''
    Runs entire cleaning routine on metacritic games dataset. It will:
        - Extrapolate year and decade into separate columns
        - Clean/trim whitespace in the platform column
        - Cleans 'tbd' from scores to 0
        - Generalize platform col into platform_type

    Args:
        df (pandas.DataFrame): The dataframe containing the dataset
    Returns
        pandas.DataFrame: Cleaned dataframe
    '''

    df = extract_year_and_decade(df, "release_date")
    df = clean_leading_whitespace(df, "platform")
    df = clean_scores(df)
    df = map_console_type(df, CONSOLE_DICT)

    return df

def extract_year_and_decade(df: DataFrame, date_col_name: str) -> DataFrame:
    '''
    Extracts the year and decade from a date column in a DataFrame.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the date column.
        date_col_name (str): The name of the date column.
        
    Returns:
        pandas.DataFrame: The input DataFrame with new "year" and "decade" columns added.
    '''
    # Convert the date column to datetime format
    df[date_col_name] = pd.to_datetime(df[date_col_name])
    
    # Extract the year and decade from the date column
    df['year'] = df[date_col_name].dt.year
    df['decade'] = (df['year'] // 10) * 10
    
    # Return the updated DataFrame
    return df

def clean_leading_whitespace(df: DataFrame, col_name: str) -> DataFrame:
    '''
    Cleans the leading whitespace characters in a column of a DataFrame.
    
    Args:
        df (pandas.DataFrame): The DataFrame to clean.
        col_name (str): The name of the column to clean.
        
    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    '''
    # Clean the leading whitespace characters in the column
    df[col_name] = df[col_name].str.strip()
    
    # Return the cleaned DataFrame
    return df

def clean_scores(df: DataFrame) -> DataFrame:
    """
    Replace 'tbd' values in 'meta_score' and 'user_review' columns with 0.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame containing the scores to clean.

    Returns
    -------
    pandas DataFrame
        The cleaned DataFrame.

    """
    # Replace 'tbd' values with 0 in 'meta_score' and 'user_review' columns
    df[['meta_score', 'user_review']] = df[['meta_score', 'user_review']].replace('tbd', 0)

    return df

def map_console_type(df: DataFrame, console_dict: str) -> DataFrame:
    '''
    Maps the console variations in a DataFrame to their console types.
    
    Args:
        df (pandas.DataFrame): The DataFrame to map.
        console_dict (dict): A dictionary of console types and variations.
        
    Returns:
        pandas.DataFrame: The input DataFrame with a new "platform_type" column.
    ''' 
    # Create a dictionary that maps each console variation to its type
    console_map = {variation: console_type for console_type, variations in console_dict.items() for variation in variations}
    
    # Use the dictionary to map the "Platform" column to console types
    df['platform_type'] = df['platform'].apply(lambda x: next((console_map[variation] for variation in console_map.keys() if variation in x), None))
    
    # Raise an assertion error if any platform types are missing
    assert df['platform_type'].notnull().all(), 'Some platform types are missing from the console dictionary.'
    
    # Return the updated DataFrame
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/all_games.csv")
    df_clean = clean_data(df)
    print(df_clean.head())