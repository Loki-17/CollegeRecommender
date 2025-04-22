import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from utils import normalize_scores
import logging
import re

# Configure logging
logger = logging.getLogger(__name__)

# Define reputation rankings for top Indian colleges
# Based on various Indian education ranking systems like NIRF with additional adjustments
# for diversity in recommendations
reputation_rankings = {
    # IITs (Indian Institutes of Technology) - Premier engineering institutes
    # Slightly reduced scores to allow more diversity
    "IIT Madras": 98,
    "IIT Delhi": 97,
    "IIT Bombay": 96,
    "IIT Kanpur": 95,
    "IIT Kharagpur": 94,
    "IIT Roorkee": 93,
    "IIT Guwahati": 92,
    "IIT Hyderabad": 91,
    "IIT Jodhpur": 86,
    
    # NITs (National Institutes of Technology) - Boosted some to increase their representation
    "NIT Tiruchirappalli": 91,
    "NIT Surathkal": 90,
    "NIT Rourkela": 88,
    "NIT Warangal": 87,
    "NIT Calicut": 86,
    "NIT Durgapur": 82,
    "NIT Kurukshetra": 81,
    
    # IIITs (Indian Institutes of Information Technology) - Boosted to increase representation
    "IIIT Hyderabad": 89,
    "IIIT Delhi": 88,
    "IIIT Bangalore": 87,
    
    # Medical colleges
    "AIIMS Delhi": 98,
    "AIIMS Jodhpur": 90,
    "Christian Medical College Vellore": 95,
    
    # Other prestigious institutions - Boosted to ensure representation
    "BITS Pilani": 93,
    "Jadavpur University": 88,
    "Anna University": 85,
    "Delhi University": 86,
    "Jamia Millia Islamia": 80,
    "BMS College of Engineering": 84,  # Boosted
    
    # Medium-ranked institutions - Some boosted to ensure they appear in recommendations
    "Chandigarh University": 75,  # Boosted
    "Amity University": 72,  # Boosted
    "Sharda University": 68,  # Boosted
    "VIT Vellore": 83,  # Added a popular college
    "KIIT University": 80,  # Added
    "SRM University": 79,  # Added
    "Manipal Institute of Technology": 85,  # Added
    
    # Lower-middle ranked institutions
    "Galgotias University": 65,  # Boosted to give it a fair chance
    "Lovely Professional University": 63,  # Boosted
    "Presidency University": 67,
    "PES University": 76,
    "Christ University": 72,
}

def get_college_reputation_score(college_name):
    """
    Get the reputation score for a college based on its name.
    Uses partial matching to handle slight variations in college names.
    
    Args:
        college_name (str): Name of the college
        
    Returns:
        float: Reputation score (0-100)
    """
    # Exact match
    if college_name in reputation_rankings:
        return reputation_rankings[college_name]
    
    # Partial match
    for known_college, score in reputation_rankings.items():
        # Check if the known college name is a subset of the given college name
        # or if the given college name is a subset of the known college name
        if known_college in college_name or college_name in known_college:
            return score
        
        # Check for abbreviations like IIT, NIT, etc.
        if any(abbr in college_name for abbr in ["IIT", "NIT", "AIIMS", "BITS", "IIIT"]) and \
           any(abbr in known_college for abbr in ["IIT", "NIT", "AIIMS", "BITS", "IIIT"]):
            # Extract the abbreviation and location
            college_abbr_match = re.search(r'(IIT|NIT|AIIMS|BITS|IIIT)\s+(\w+)', college_name)
            known_abbr_match = re.search(r'(IIT|NIT|AIIMS|BITS|IIIT)\s+(\w+)', known_college)
            
            if college_abbr_match and known_abbr_match:
                if (college_abbr_match.group(1) == known_abbr_match.group(1) and 
                    college_abbr_match.group(2) == known_abbr_match.group(2)):
                    return score
    
    # Default score for unknown colleges
    return 40  # Middle-range default

def train_model(df):
    """
    Train a linear regression model on college review data.
    
    Args:
        df (pd.DataFrame): DataFrame with preprocessed college review data
        
    Returns:
        tuple: (model, vectorizer, feature_names)
    """
    # Extract features from processed reviews
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['processed_review'])
    feature_names = vectorizer.get_feature_names_out()
    
    # Target variable is the average rating
    y = df['avg_rating'].values
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    logger.info(f"Model trained with {len(feature_names)} features")
    return model, vectorizer, feature_names

def get_recommendations(user_input, model, vectorizer, df, top_n=5):
    """
    Generate college recommendations based on user input.
    
    Args:
        user_input (str): User's preferences/query
        model (LinearRegression): Trained model
        vectorizer (TfidfVectorizer): Fitted vectorizer
        df (pd.DataFrame): DataFrame with college data
        top_n (int): Number of recommendations to return
        
    Returns:
        tuple: (list of college names, list of scores)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        from data_processing import clean_text, preprocess_text
        
        # Preprocess user input
        logger.info(f"Processing user input: {user_input[:50]}...")
        clean_input = clean_text(user_input)
        processed_input = preprocess_text(clean_input)
        logger.info(f"Processed input: {processed_input[:50]}...")
    except Exception as e:
        logger.error(f"Error preprocessing user input: {str(e)}")
        raise
    
    # Convert to feature vector
    input_vector = vectorizer.transform([processed_input])
    
    # Calculate similarity scores for all colleges
    X = vectorizer.transform(df['processed_review'])
    
    # Predict ratings
    predicted_ratings = model.predict(X)
    
    # Get reputation scores for all colleges
    reputation_scores = np.array([get_college_reputation_score(college) for college in df['college_name']])
    
    # Normalize reputation scores to 0-5 scale to match rating scale
    reputation_scores = (reputation_scores / 100) * 5
    
    # Combine with weights:
    # - 50% reputation score (real-world reputation is very important)
    # - 30% actual rating from reviews
    # - 20% predicted rating from model
    combined_scores = (
        0.5 * reputation_scores + 
        0.3 * df['avg_rating'].values + 
        0.2 * predicted_ratings
    )
    
    # Check for engineering or tech keywords in the input
    is_engineering_focused = any(word in user_input.lower() for word in 
                                ['engineering', 'computer', 'tech', 'cse', 'it', 'software'])
    
    # Check for medical keywords in the input
    is_medical_focused = any(word in user_input.lower() for word in 
                            ['medical', 'doctor', 'mbbs', 'medicine', 'hospital', 'healthcare'])
    
    # Boost scores for domain-specific colleges
    if is_engineering_focused:
        for i, college in enumerate(df['college_name']):
            if any(tech_college in college for tech_college in ['NIT', 'IIIT', 'Tech', 'Engineering']):
                combined_scores[i] *= 1.2  # 20% boost
            elif 'IIT' in college:
                combined_scores[i] *= 1.1  # Only 10% boost for IITs for more diversity
    
    if is_medical_focused:
        for i, college in enumerate(df['college_name']):
            if any(med_college in college for med_college in ['Medical', 'Medicine', 'Health']):
                combined_scores[i] *= 1.2  # 20% boost
            elif 'AIIMS' in college:
                combined_scores[i] *= 1.1  # Only 10% boost for AIIMS for more diversity
    
    # Get indices sorted by score
    sorted_indices = np.argsort(combined_scores)[::-1]
    
    # Get diversified recommendations
    selected_indices = []
    iit_count = 0
    nit_count = 0
    
    # Limit number of similar institutions to ensure diversity
    for idx in sorted_indices:
        college_name = df.iloc[idx]['college_name']
        
        # Limit IITs to max 1 in top 5
        if 'IIT' in college_name:
            if iit_count < 1:
                selected_indices.append(idx)
                iit_count += 1
        # Limit NITs to max 2 in top 5
        elif 'NIT' in college_name:
            if nit_count < 2:
                selected_indices.append(idx)
                nit_count += 1
        # Add other colleges
        else:
            selected_indices.append(idx)
        
        # Break when we have enough recommendations
        if len(selected_indices) >= top_n:
            break
    
    # If we don't have enough, add more from the sorted list
    if len(selected_indices) < top_n:
        for idx in sorted_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
                if len(selected_indices) >= top_n:
                    break
    
    # Get college names and scores
    top_colleges = [df.iloc[idx]['college_name'] for idx in selected_indices]
    top_scores = normalize_scores(combined_scores[selected_indices])
    
    logger.info(f"Generated {len(top_colleges)} recommendations")
    logger.info(f"Top recommendations: {', '.join(top_colleges)}")
    return top_colleges, top_scores
