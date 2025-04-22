import os
import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pandas as pd
import numpy as np
from data_processing import preprocess_data, combine_datasets
from recommendation_model import train_model, get_recommendations
from utils import load_college_data

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Load and preprocess data
try:
    logger.info("Loading and preprocessing data...")
    df1 = pd.read_csv('attached_assets/indian_colleges_reviews.csv')
    df2 = pd.read_csv('attached_assets/indian_colleges_reviews_v2.csv')
    
    # Combine datasets
    combined_df = combine_datasets(df1, df2)
    
    # Preprocess data
    processed_df = preprocess_data(combined_df)
    
    # Train model
    model, vectorizer, feature_names = train_model(processed_df)
    
    logger.info("Data loaded and model trained successfully")
except Exception as e:
    logger.error(f"Error during data loading or model training: {str(e)}")
    model, vectorizer, feature_names, processed_df = None, None, None, None

# Routes
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Process user input and generate recommendations."""
    if request.method == 'POST':
        user_input = request.form.get('user_input', '')
        
        if not user_input:
            flash('Please enter your preferences to get recommendations.', 'warning')
            return redirect(url_for('index'))
        
        try:
            recommendations, scores = get_recommendations(
                user_input, 
                model, 
                vectorizer, 
                processed_df, 
                top_n=5
            )
            
            # Import reputation score function
            from recommendation_model import get_college_reputation_score
            
            # Gather complete college information
            colleges_info = []
            for i, college in enumerate(recommendations):
                # Find the college in the processed dataframe
                college_data = processed_df[processed_df['college_name'] == college].iloc[0].to_dict()
                
                # Add match score
                college_data['score'] = round(scores[i] * 100, 2)  # Convert to percentage
                
                # Add reputation score
                reputation_score = get_college_reputation_score(college)
                college_data['reputation_score'] = reputation_score
                
                colleges_info.append(college_data)
            
            # Log the recommendations
            logger.info(f"Recommendations for input: '{user_input[:100]}...'")
            for college in colleges_info:
                logger.info(f"College: {college['college_name']}, Reputation: {college.get('reputation_score', 'N/A')}, Match: {college['score']}%")
            
            return render_template('recommendations.html', 
                                  colleges=colleges_info, 
                                  user_input=user_input)
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            flash('An error occurred while generating recommendations. Please try again.', 'danger')
            return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/api/colleges', methods=['GET'])
def get_colleges():
    """API endpoint to get list of colleges for autocomplete."""
    try:
        colleges = processed_df['college_name'].unique().tolist()
        return jsonify({'colleges': colleges})
    except Exception as e:
        logger.error(f"Error fetching colleges: {str(e)}")
        return jsonify({'colleges': []})

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('index.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
