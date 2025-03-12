import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

def scrape_coffee_reviews(num_pages=10):
    """
    Scrape coffee reviews from CoffeeReview.com
    
    Args:
        num_pages: Number of pages to scrape
        
    Returns:
        DataFrame with coffee reviews
    """
    base_url = "https://www.coffeereview.com/reviews/page/{}/"
    
    all_reviews = []
    
    for page in range(1, num_pages + 1):
        try:
            url = base_url.format(page)
            print(f"Scraping page: {url}")
            
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
            
            # Check if the request was successful
            if response.status_code != 200:
                print(f"Failed to load page {page}: Status code {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug - print page title to verify content
            print(f"Page {page} title: {soup.title.text if soup.title else 'No title found'}")
            
            # Find all review links on the page
            review_links = []
            
            # Look for all anchor tags with href containing "/review/"
            all_links = soup.find_all('a', href=True)
            for link in all_links:
                if '/review/' in link['href'] and link['href'] not in review_links:
                    review_links.append(link['href'])
            
            # Remove duplicates and ensure all URLs are absolute
            review_links = list(set(review_links))
            review_links = [link if link.startswith('http') else f"https://www.coffeereview.com{link}" for link in review_links]
            
            print(f"Found {len(review_links)} review links on page {page}")
            if review_links:
                print("Sample links:")
                for link in review_links[:3]:
                    print(f" - {link}")
            
            # Process each review
            for review_url in review_links:
                try:
                    # Add randomized delay before requesting review page
                    time.sleep(random.uniform(1, 3))
                    
                    print(f"Scraping review: {review_url}")
                    review_response = requests.get(review_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
                    
                    if review_response.status_code != 200:
                        print(f"Failed to load review {review_url}: Status code {review_response.status_code}")
                        continue
                        
                    review_soup = BeautifulSoup(review_response.content, 'html.parser')
                    
                    # Initialize all review data fields
                    title = "N/A"
                    roaster = "N/A"
                    origin = "N/A"
                    price = "N/A"
                    score = "N/A"
                    aroma = "N/A"
                    acidity = "N/A"
                    body = "N/A"
                    flavor = "N/A"
                    aftertaste = "N/A"
                    review_date = "N/A"
                    full_review = ""
                    
                    # 1. Extract title using review-title class
                    title_element = review_soup.find('h1', class_='review-title')
                    if title_element and title_element.text.strip():
                        title = title_element.text.strip()
                        print(f"Found title: {title}")
                    
                    # 2. Extract roaster using review-roaster class 
                    roaster_element = review_soup.find('p', class_='review-roaster')
                    if roaster_element and roaster_element.text.strip():
                        roaster = roaster_element.text.strip()
                        print(f"Found roaster: {roaster}")
                    
                    # 3. Extract score using review-template-rating class
                    score_element = review_soup.find('span', class_='review-template-rating')
                    if score_element and score_element.text.strip():
                        score = score_element.text.strip()
                        print(f"Found score: {score}")
                    
                    # 4. Extract review attributes from the proper table structure
                    # First find the row-2 div
                    row2 = review_soup.find('div', class_='row row-2')
                    if row2:
                        # Extract metadata from column 1
                        col1 = row2.find('div', class_='column col-1')
                        if col1:
                            meta_table = col1.find('table', class_='review-template-table')
                            if meta_table:
                                rows = meta_table.find_all('tr')
                                for row in rows:
                                    cells = row.find_all('td')
                                    if len(cells) >= 2:
                                        label = cells[0].text.strip().lower()
                                        value = cells[1].text.strip()
                                        
                                        if 'coffee origin' in label or 'origin' in label:
                                            origin = value
                                            print(f"Found origin: {origin}")
                                        elif 'price' in label or 'est. price' in label:
                                            price = value
                                            print(f"Found price: {price}")
                        
                        # Extract ratings from column 2
                        col2 = row2.find('div', class_='column col-2')
                        if col2:
                            rating_table = col2.find('table', class_='review-template-table')
                            if rating_table:
                                rows = rating_table.find_all('tr')
                                for row in rows:
                                    cells = row.find_all('td')
                                    if len(cells) >= 2:
                                        category = cells[0].text.strip().lower()
                                        value = cells[1].text.strip()
                                        
                                        if 'review date' in category:
                                            review_date = value
                                            print(f"Found review date: {review_date}")
                                        elif 'aroma' in category:
                                            aroma = value
                                            print(f"Found aroma: {aroma}")
                                        elif 'acidity' in category or 'structure' in category:
                                            acidity = value
                                            print(f"Found acidity: {acidity}")
                                        elif 'body' in category:
                                            body = value
                                            print(f"Found body: {body}")
                                        elif 'flavor' in category:
                                            flavor = value
                                            print(f"Found flavor: {flavor}")
                                        elif 'aftertaste' in category:
                                            aftertaste = value
                                            print(f"Found aftertaste: {aftertaste}")
                    
                    # 5. Extract the full review text - target specific sections
                    review_sections = []
                    
                    # Find all h2 headers in the review template
                    entry_content = review_soup.find('div', class_='entry-content')
                    if entry_content:
                        review_template = entry_content.find('div', class_='review-template')
                        if review_template:
                            h2_tags = review_template.find_all('h2')
                            for h2 in h2_tags:
                                section_title = h2.text.strip()
                                section_content = ""
                                
                                # Find the next paragraph after the h2
                                next_p = h2.find_next('p')
                                if next_p:
                                    section_content = next_p.text.strip()
                                    review_sections.append(f"{section_title}: {section_content}")
                    
                    # Combine all sections into the full review
                    if review_sections:
                        full_review = " ".join(review_sections)
                        print(f"Found full review text ({len(full_review)} chars)")
                    
                    # Fallback methods if primary methods fail
                    
                    # Fallback for title
                    if title == "N/A":
                        # Try any h1 elements
                        for h1 in review_soup.find_all('h1'):
                            if h1.text.strip() and h1.get('class') != 'review-title':
                                title = h1.text.strip()
                                print(f"Found title (fallback): {title}")
                                break
                        
                        # If still no title, extract from URL
                        if title == "N/A":
                            url_match = re.search(r'/review/([^/]+)/?$', review_url)
                            if url_match:
                                title_slug = url_match.group(1)
                                title = " ".join(word.capitalize() for word in title_slug.split('-'))
                                print(f"Generated title from URL: {title}")
                    
                    # Fallback for score
                    if score == "N/A":
                        score_patterns = [r'(\d{2,3})\s*points', r'score:\s*(\d{2,3})', r'rated\s*(\d{2,3})']
                        
                        for elem in review_soup.find_all(['div', 'span', 'p', 'h2', 'h3']):
                            for pattern in score_patterns:
                                score_match = re.search(pattern, elem.text.lower())
                                if score_match:
                                    score = score_match.group(1)
                                    print(f"Found score (fallback): {score}")
                                    break
                            if score != "N/A":
                                break
                    
                    # Fallback for review text if no sections were found
                    if not full_review:
                        paragraphs = []
                        if entry_content:
                            paragraphs = entry_content.find_all('p')
                        elif review_soup.find('article'):
                            paragraphs = review_soup.find('article').find_all('p')
                            
                        if paragraphs:
                            full_review = ' '.join([p.text.strip() for p in paragraphs])
                            print(f"Found review text using fallback method ({len(full_review)} chars)")
                    
                    # Add review to dataset
                    review_data = {
                        'title': title,
                        'roaster': roaster,
                        'origin': origin,
                        'score': score,
                        'price': price,
                        'review_date': review_date,
                        'aroma': aroma,
                        'acidity': acidity,
                        'body': body,
                        'flavor': flavor,
                        'aftertaste': aftertaste,
                        'full_review': full_review,
                        'url': review_url
                    }
                    
                    all_reviews.append(review_data)
                    print(f"Added review: {title} - Score: {score}")
                    print(f"Details - Aroma: {aroma}, Acidity: {acidity}, Body: {body}, Flavor: {flavor}, Aftertaste: {aftertaste}")
                        
                except Exception as e:
                    print(f"Error processing review {review_url}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Be respectful with rate limiting
            time.sleep(random.uniform(3, 6))
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
    
    # Convert to DataFrame
    reviews_df = pd.DataFrame(all_reviews)
    return reviews_df

# Function to clean numeric data in the DataFrame
def clean_review_data(df):
    """
    Clean the review data:
    - Convert score and rating attributes to numeric
    - Extract numeric values from price
    - Format dates consistently
    
    Args:
        df: DataFrame with raw coffee review data
        
    Returns:
        DataFrame with cleaned data
    """
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Clean score column - extract just the number
    cleaned_df['score_numeric'] = cleaned_df['score'].apply(
        lambda x: int(re.search(r'(\d+)', str(x)).group(1)) if re.search(r'(\d+)', str(x)) else None
    )
    
    # Clean rating attributes - convert to numeric
    for col in ['aroma', 'acidity', 'body', 'flavor', 'aftertaste']:
        cleaned_df[f'{col}_numeric'] = cleaned_df[col].apply(
            lambda x: float(re.search(r'(\d+(\.\d+)?)', str(x)).group(1)) if re.search(r'(\d+(\.\d+)?)', str(x)) else None
        )
    
    # Extract price as a numeric value
    cleaned_df['price_numeric'] = cleaned_df['price'].apply(
        lambda x: float(re.search(r'\$(\d+(\.\d+)?)', str(x)).group(1)) if re.search(r'\$(\d+(\.\d+)?)', str(x)) else None
    )
    
    # Standardize review date format
    # This assumes dates are in a somewhat consistent format
    def standardize_date(date_str):
        if pd.isna(date_str) or date_str == 'N/A':
            return None
        
        # Handle common date formats
        date_patterns = [
            (r'(\w+)\s+(\d{4})', '%B %Y'),  # February 2025
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', '%m/%d/%Y'),  # 2/15/2025
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d')   # 2025-02-15
        ]
        
        for pattern, format_str in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    return pd.to_datetime(date_str, format=format_str)
                except:
                    pass
        
        return None
    
    cleaned_df['review_date_clean'] = cleaned_df['review_date'].apply(standardize_date)
    
    return cleaned_df

# Example usage
if __name__ == "__main__":
    print("Starting coffee review scraper...")
    reviews = scrape_coffee_reviews(num_pages=10)  # Just trying page 1 for testing
    
    if len(reviews) > 0:
        # Clean the data
        cleaned_reviews = clean_review_data(reviews)
        
        # Save both raw and cleaned data
        reviews.to_csv('coffee_reviews_raw.csv', index=False)
        cleaned_reviews.to_csv('coffee_reviews_cleaned.csv', index=False)
        
        print(f"Saved {len(reviews)} reviews to CSV files")
        print(f"Preview of the cleaned data:")
        print(cleaned_reviews[['title', 'roaster', 'score_numeric', 'origin', 'review_date_clean', 
                              'aroma_numeric', 'acidity_numeric', 'body_numeric', 'flavor_numeric', 'aftertaste_numeric']].head())
    else:
        print("No reviews were found. Please check the script and website structure.")