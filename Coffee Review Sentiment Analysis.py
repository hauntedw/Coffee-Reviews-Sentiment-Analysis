import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from collections import Counter
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def analyze_coffee_reviews(csv_file):
    """
    Analyze sentiment and extract flavor descriptors from coffee reviews
    
    Args:
        csv_file: Path to CSV file with coffee review data
        
    Returns:
        DataFrame with sentiment analysis and flavor descriptors
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Combine description and full review for analysis
    df['text_to_analyze'] = df['full_review']

    # Filter out "LATEST REVIEWS" entries
    df = df[df['title'] != 'LATEST REVIEWS']
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Common coffee flavor descriptors 
    flavor_descriptors = [
        'fruity', 'chocolatey', 'nutty', 'citrus', 'floral', 'berry', 'caramel',
        'sweet', 'bitter', 'acidic', 'smooth', 'balanced', 'bright', 'bold',
        'earthy', 'spicy', 'woody', 'herbal', 'rich', 'full-bodied', 'light',
        'medium', 'dark', 'intense', 'complex', 'clean', 'crisp', 'mellow',
        'robust', 'aromatic', 'smoky', 'winey', 'honey', 'vanilla', 'tobacco',
        'leather', 'cherry', 'blueberry', 'apple', 'cocoa', 'chocolate', 'nut',
        'hazelnut', 'almond', 'cinnamon', 'clove', 'molasses', 'toffee'
    ]
    
    # Analysis results
    results = []
    
    for idx, row in df.iterrows():
        text = row['text_to_analyze']
        
        # Skip empty reviews
        if pd.isna(text) or text.strip() == '':
            continue
            
        # VADER sentiment analysis
        sentiment = sia.polarity_scores(text)
        
        # TextBlob for additional sentiment and subjectivity
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment
        
        # Extract score as numeric value
        score = None
        if not pd.isna(row['score']):
            score_match = re.search(r'(\d+)', str(row['score']))
            if score_match:
                score = int(score_match.group(1))
        
        # Extract flavor descriptors
        found_descriptors = []
        doc = nlp(text.lower())
        
        # Count occurrences of flavor descriptors
        descriptor_counts = Counter()
        for descriptor in flavor_descriptors:
            count = len(re.findall(r'\b' + descriptor + r'\b', text.lower()))
            if count > 0:
                descriptor_counts[descriptor] = count
                found_descriptors.append(descriptor)
        
        # Extract top descriptors
        top_descriptors = [desc for desc, count in descriptor_counts.most_common(5)]
        
        # Extract adjectives (potential flavor descriptors not in our list)
        adjectives = [token.text for token in doc if token.pos_ == 'ADJ']
        top_adjectives = [adj for adj, count in Counter(adjectives).most_common(5)]
        
        result = {
            'title': row['title'],
            'score': score,
            'compound_sentiment': sentiment['compound'],
            'positive_sentiment': sentiment['pos'],
            'negative_sentiment': sentiment['neg'],
            'neutral_sentiment': sentiment['neu'],
            'textblob_polarity': textblob_sentiment.polarity,
            'textblob_subjectivity': textblob_sentiment.subjectivity,
            'top_descriptors': ', '.join(top_descriptors),
            'top_adjectives': ', '.join(top_adjectives),
            'descriptor_count': len(found_descriptors),
            'all_descriptors': ', '.join(found_descriptors)
        }
        
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Merge with original data - FIXED: added 'origin' to the columns list
    final_df = pd.merge(
        df[['title', 'price', 'url', 'origin']], 
        results_df, 
        on='title', 
        how='right'
    )
    
    return final_df

def visualize_sentiment_trends(analysis_df):
    """
    Create visualizations for coffee review sentiment analysis
    
    Args:
        analysis_df: DataFrame with sentiment analysis results
    """
    # Set up the plotting style
    sns.set(style="whitegrid")
    
    # Plot 1: Sentiment vs. Score correlation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='score', y='compound_sentiment', data=analysis_df)
    plt.title('Coffee Review Score vs. Sentiment')
    plt.xlabel('Coffee Score')
    plt.ylabel('Sentiment Score')
    plt.savefig('score_vs_sentiment.png')
    
    # Plot 2: Top 10 flavor descriptors
    plt.figure(figsize=(12, 8))
    all_descriptors = []
    for descriptors in analysis_df['all_descriptors'].dropna():
        if descriptors:
            all_descriptors.extend([d.strip() for d in descriptors.split(',')])
    
    descriptor_counts = Counter(all_descriptors)
    top_descriptors = descriptor_counts.most_common(15)
    
    descriptors, counts = zip(*top_descriptors)
    sns.barplot(x=list(counts), y=list(descriptors))
    plt.title('Top 15 Coffee Flavor Descriptors')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig('top_descriptors.png')
    
    # Plot 3: Sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(analysis_df['compound_sentiment'], kde=True)
    plt.title('Distribution of Coffee Review Sentiment')
    plt.xlabel('Sentiment Score')
    plt.savefig('sentiment_distribution.png')
    
    # Plot 4: Average sentiment by descriptor
    plt.figure(figsize=(14, 10))
    descriptor_sentiment = {}
    
    for idx, row in analysis_df.iterrows():
        if pd.isna(row['all_descriptors']) or row['all_descriptors'] == '':
            continue
            
        descriptors = [d.strip() for d in row['all_descriptors'].split(',')]
        for descriptor in descriptors:
            if descriptor not in descriptor_sentiment:
                descriptor_sentiment[descriptor] = []
            descriptor_sentiment[descriptor].append(row['compound_sentiment'])
    
    # Calculate average sentiment for each descriptor
    avg_sentiment = {
        desc: sum(sentiments)/len(sentiments) 
        for desc, sentiments in descriptor_sentiment.items() 
        if len(sentiments) >= 5  # Only include descriptors with enough data
    }
    
    # Sort and plot
    sorted_descriptors = sorted(avg_sentiment.items(), key=lambda x: x[1], reverse=True)
    if sorted_descriptors:
        top_n = 20
        descriptors, sentiments = zip(*sorted_descriptors[:top_n])
        
        sns.barplot(x=list(sentiments), y=list(descriptors))
        plt.title(f'Average Sentiment by Coffee Descriptor (Top {top_n})')
        plt.xlabel('Average Sentiment Score')
        plt.tight_layout()
        plt.savefig('descriptor_sentiment.png')
    
    # Create radar chart of top descriptors
    plt.figure(figsize=(10, 10))
    top_descriptors_count = dict(descriptor_counts.most_common(8))
    
    # Convert to percentages for better radar chart
    total = sum(top_descriptors_count.values())
    categories = list(top_descriptors_count.keys())
    values = [count/total*100 for count in top_descriptors_count.values()]
    
    # Create the radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # Close the polygon
    angles += angles[:1]  # Close the polygon
    categories += categories[:1]  # Close the polygon
    
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
    ax.set_title("Coffee Flavor Profile Distribution")
    plt.tight_layout()
    plt.savefig('flavor_radar.png')
    
    print("Visualizations saved as PNG files")

def export_data_for_dashboard(analysis_df):
    """Export analysis results in the format needed for the React dashboard with relative sentiment scoring"""
    import numpy as np
    
    # First, let's examine the dataframe columns
    print("\n=== DATAFRAME COLUMN CHECK ===")
    print(f"Columns in dataframe: {list(analysis_df.columns)}")
    print(f"Total rows in dataframe: {len(analysis_df)}")
    
    # Check if 'origin' is in the dataframe
    has_origin_column = 'origin' in analysis_df.columns
    print(f"Has 'origin' column: {has_origin_column}")
    
    # Check if compound_sentiment exists
    has_sentiment = 'compound_sentiment' in analysis_df.columns
    print(f"\nHas 'compound_sentiment' column: {has_sentiment}")
    
    # Get top descriptors
    all_descriptors = []
    for descriptors in analysis_df['all_descriptors'].dropna():
        if descriptors:
            all_descriptors.extend([d.strip() for d in descriptors.split(',')])
    
    from collections import Counter
    descriptor_counts = Counter(all_descriptors)
    top_descriptors = descriptor_counts.most_common(8)
    
    # Format as needed by React component
    dashboard_data = {
        "topDescriptors": [
            {"name": desc, "value": count} for desc, count in top_descriptors
        ],
        
        # Initialize empty for now
        "sentimentByOrigin": [],
        
        # Get sentiment distribution
        "sentimentDistribution": [],
        
        # Radar data (same as top descriptors but with different format)
        "radarData": [
            {"descriptor": desc, "frequency": count} for desc, count in top_descriptors
        ]
    }
    
    # Calculate sentiment by origin with a relative approach
    print("\n=== SENTIMENT BY ORIGIN CALCULATION ===")
    if has_origin_column and has_sentiment:
        print("Both 'origin' and 'compound_sentiment' columns found, proceeding...")
        
        # Process origins and gather sentiments
        sentiment_by_origin = {}
        processed_rows = 0
        skipped_rows = 0
        all_sentiments = []  # Collect all sentiment values for percentile calculation
        
        for idx, row in analysis_df.iterrows():
            origin_value = row['origin']
            compound_sentiment = row.get('compound_sentiment')
            
            # Skip invalid data
            if pd.isna(origin_value) or not isinstance(origin_value, str) or origin_value == 'N/A':
                skipped_rows += 1
                continue
            
            if pd.isna(compound_sentiment):
                skipped_rows += 1
                continue
            
            all_sentiments.append(compound_sentiment)
            
            # Split by semicolon if present (for multi-origin coffees)
            if ';' in origin_value:
                origins = [org.strip() for org in origin_value.split(';')]
            else:
                origins = [origin_value.strip()]
            
            # Skip empty origins
            origins = [org for org in origins if org and org != 'N/A']
            if not origins:
                skipped_rows += 1
                continue
            
            processed_rows += 1
            
            # Add to each origin's sentiment list
            for origin in origins:
                if origin not in sentiment_by_origin:
                    sentiment_by_origin[origin] = []
                sentiment_by_origin[origin].append(compound_sentiment)
        
        print(f"\nProcessed {processed_rows} rows with valid origins and sentiments")
        print(f"Skipped {skipped_rows} rows due to missing/invalid data")
        
        # Calculate sentiment quartiles for the entire dataset
        if all_sentiments:
            q1 = np.percentile(all_sentiments, 25)
            q2 = np.percentile(all_sentiments, 50)  # median
            q3 = np.percentile(all_sentiments, 75)
            
            print(f"\nSentiment quartiles for all reviews:")
            print(f"  Q1 (25th percentile): {q1:.3f}")
            print(f"  Q2 (median): {q2:.3f}")
            print(f"  Q3 (75th percentile): {q3:.3f}")
            
            # Instead of using absolute thresholds, we'll use relative positioning
            # Based on the dataset's own distribution
            
            origin_sentiment_data = []
            min_reviews = 3  # Only include origins with enough data
            qualifying_origins = 0
            
            for origin, sentiments in sentiment_by_origin.items():
                if len(sentiments) >= min_reviews:
                    qualifying_origins += 1
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    
                    # Calculate percentage of reviews in each relative category
                    below_avg = sum(1 for s in sentiments if s < q2) / len(sentiments)
                    above_avg = 1 - below_avg
                    
                    # For more detail, split into quartiles
                    bottom_25 = sum(1 for s in sentiments if s < q1) / len(sentiments)
                    mid_lower = sum(1 for s in sentiments if q1 <= s < q2) / len(sentiments)
                    mid_upper = sum(1 for s in sentiments if q2 <= s < q3) / len(sentiments)
                    top_25 = sum(1 for s in sentiments if s >= q3) / len(sentiments)
                    
                    origin_sentiment_data.append({
                        "name": origin,
                        "reviewCount": len(sentiments),
                        "averageSentiment": round(avg_sentiment, 2),
                        "belowMedian": round(below_avg, 2),
                        "aboveMedian": round(above_avg, 2),
                        "bottom25": round(bottom_25, 2),
                        "midLower25": round(mid_lower, 2),
                        "midUpper25": round(mid_upper, 2),
                        "top25": round(top_25, 2)
                    })
            
            print(f"\n{qualifying_origins} origins have at least {min_reviews} reviews")
            
            # Sort by number of reviews and take top 5
            if origin_sentiment_data:
                origin_sentiment_data.sort(key=lambda x: x["reviewCount"], reverse=True)
                dashboard_data["sentimentByOrigin"] = origin_sentiment_data[:5]
                
                print("\nTop 5 origins by review count (included in JSON):")
                for entry in dashboard_data["sentimentByOrigin"]:
                    print(f"  - {entry['name']}: {entry['reviewCount']} reviews, "
                          f"avg sentiment: {entry['averageSentiment']}, "
                          f"{entry['aboveMedian']*100:.1f}% above median, "
                          f"{entry['belowMedian']*100:.1f}% below median")
            else:
                print("\nNo origins qualified for inclusion (need at least 3 reviews)")
        else:
            print("No valid sentiment values found")
    else:
        if not has_origin_column:
            print("ERROR: 'origin' column is missing from the dataframe")
        if not has_sentiment:
            print("ERROR: 'compound_sentiment' column is missing from the dataframe")
    
    # Calculate sentiment distribution bins
    if has_sentiment:
        # Create more detailed bins for better visualization
        bins = [-1.0, -0.5, 0, 0.5, 0.75, 0.85, 0.95, 1.0]  # Adjusted for coffee reviews which skew positive
        bin_labels = ['Very Negative (-1.0 to -0.5)', 
                      'Negative (-0.5 to 0)', 
                      'Neutral (0 to 0.5)', 
                      'Positive (0.5 to 0.75)', 
                      'Very Positive (0.75 to 0.85)', 
                      'Excellent (0.85 to 0.95)', 
                      'Outstanding (0.95 to 1.0)']
        
        sentiment_values = analysis_df['compound_sentiment']
        hist, _ = np.histogram(sentiment_values, bins=bins)
        
        dashboard_data["sentimentDistribution"] = [
            {"name": label, "count": int(count)} 
            for label, count in zip(bin_labels, hist)
        ]
    
    # Save as JSON
    import json
    with open('dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print("\nDashboard data exported to dashboard_data.json")
    return dashboard_data

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Analyze the coffee reviews
    analysis_results = analyze_coffee_reviews('coffee_reviews_cleaned.csv')
    
    # Save the analysis results
    analysis_results.to_csv('coffee_sentiment_analysis.csv', index=False)
    print(f"Saved analysis results for {len(analysis_results)} reviews")
    
    # Create visualizations
    visualize_sentiment_trends(analysis_results)

    # Export data for dashboard
    export_data_for_dashboard(analysis_results)