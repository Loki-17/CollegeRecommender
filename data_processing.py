import pandas as pd
import numpy as np
import re
import nltk
import logging
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure logging
logger = logging.getLogger(__name__)

# Set NLTK data path to a writeable directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.insert(0, nltk_data_dir)

# Download necessary NLTK data
try:
    logger.info(f"Downloading NLTK data to {nltk_data_dir}")
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)
    logger.info("NLTK data download complete")
except Exception as e:
    logger.error(f"Failed to download NLTK data: {str(e)}")

def combine_datasets(df1, df2):
    """
    Combine two college review datasets.
    
    Args:
        df1 (pd.DataFrame): First dataset
        df2 (pd.DataFrame): Second dataset
        
    Returns:
        pd.DataFrame: Combined dataset
    """
    # Ensure columns match
    assert df1.columns.tolist() == df2.columns.tolist(), "Datasets have different columns"
    
    # Combine datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Drop duplicates if any (based on all columns)
    combined_df = combined_df.drop_duplicates()
    
    logger.info(f"Combined dataset shape: {combined_df.shape}")
    return combined_df

def clean_text(text):
    """
    Clean text data by removing special characters, numbers, etc.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    """
    Preprocess text by tokenizing, removing stopwords, and lemmatizing.
    
    Args:
        text (str): Text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    try:
        # Simple tokenization using split (no dependency on punkt_tab)
        tokens = text.split()
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except:
            # Fallback for stopwords if NLTK resource is not available
            common_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
            stop_words = common_stopwords
            
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize if available
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        except:
            # Skip lemmatization if not available
            logger.warning("Lemmatization skipped due to missing resource")
            pass
        
        # Join tokens back to string
        processed_text = ' '.join(tokens)
        
        return processed_text
    except Exception as e:
        logger.error(f"Error in text preprocessing: {str(e)}")
        # Return original text if processing fails
        return text

def preprocess_data(df):
    """
    Preprocess college review data.
    
    Args:
        df (pd.DataFrame): DataFrame with college review data
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Clean and preprocess reviews
    logger.info("Cleaning and preprocessing reviews...")
    processed_df['cleaned_review'] = processed_df['review'].apply(clean_text)
    processed_df['processed_review'] = processed_df['cleaned_review'].apply(preprocess_text)
    
    # Handle missing values
    processed_df = processed_df.dropna(subset=['college_name', 'review', 'user_rating'])
    
    # Remove entries with empty processed reviews
    processed_df = processed_df[processed_df['processed_review'].str.strip() != '']
    
    # Calculate average rating per college
    college_ratings = processed_df.groupby('college_name')['user_rating'].mean().reset_index()
    college_ratings.rename(columns={'user_rating': 'avg_rating'}, inplace=True)
    
    # Merge back to the processed dataframe
    processed_df = pd.merge(processed_df, college_ratings, on='college_name', how='left')
    
    # Count reviews per college
    review_counts = processed_df.groupby('college_name').size().reset_index(name='review_count')
    processed_df = pd.merge(processed_df, review_counts, on='college_name', how='left')
    
    # Drop duplicates based on college_name
    processed_df = processed_df.drop_duplicates(subset=['college_name'])
    
    logger.info(f"Preprocessed data shape: {processed_df.shape}")
    return processed_df
