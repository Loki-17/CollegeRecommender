import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_college_data(csv_path):
    """
    Load college data from CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: DataFrame with college data
    """
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error loading data from {csv_path}: {str(e)}")
        raise

def get_college_details(college_name, df):
    """
    Get details of a specific college.
    
    Args:
        college_name (str): Name of the college
        df (pd.DataFrame): DataFrame with college data
        
    Returns:
        dict: College details
    """
    try:
        college_data = df[df['college_name'] == college_name]
        if college_data.empty:
            return None
        
        # Calculate average rating
        avg_rating = college_data['user_rating'].mean()
        
        # Get reviews
        reviews = college_data['review'].tolist()
        
        # Count reviews
        review_count = len(reviews)
        
        # Get abbreviation
        abbreviation = college_data['college_abbreviation'].iloc[0]
        
        return {
            'name': college_name,
            'abbreviation': abbreviation,
            'avg_rating': round(avg_rating, 1),
            'reviews': reviews,
            'review_count': review_count
        }
    except Exception as e:
        logger.error(f"Error getting details for college {college_name}: {str(e)}")
        return None

def normalize_scores(scores):
    """
    Normalize scores to be between 0 and 1.
    
    Args:
        scores (np.array): Array of scores
        
    Returns:
        np.array: Normalized scores
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.ones_like(scores)
    
    return (scores - min_score) / (max_score - min_score)
