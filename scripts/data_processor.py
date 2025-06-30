import pandas as pd
import numpy as np
import pyodbc
import json
import re
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from dotenv import load_dotenv

load_dotenv()

class DataProcessor:
    def __init__(self):
        self.connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={os.getenv('MSSQL_SERVER', 'localhost')};"
            f"DATABASE={os.getenv('MSSQL_DATABASE', 'GadgetRecommendations')};"
            f"UID={os.getenv('MSSQL_USERNAME', 'sa')};"
            f"PWD={os.getenv('MSSQL_PASSWORD', 'your_password')}"
        )
        
    def connect_to_database(self):
        """Establish database connection"""
        return pyodbc.connect(self.connection_string)
    
    def clean_product_data(self, products_data):
        """Clean and standardize product data"""
        df = pd.DataFrame(products_data)
        
        # Clean product names
        df['name'] = df['name'].str.strip()
        df['name'] = df['name'].str.replace(r'\s+', ' ', regex=True)
        
        # Extract brand from product name
        df['brand'] = df['name'].apply(self.extract_brand)
        
        # Clean price data
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Clean rating data
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating'] = df['rating'].clip(0, 5)  # Ensure ratings are between 0-5
        
        # Standardize categories
        df['category'] = df['category'].str.title()
        df['subcategory'] = df.get('subcategory', '').str.title()
        
        # Remove duplicates based on name and source
        df = df.drop_duplicates(subset=['name', 'source'], keep='first')
        
        # Fill missing values
        df['rating'] = df['rating'].fillna(0)
        df['price'] = df['price'].fillna(0)
        df['brand'] = df['brand'].fillna('Unknown')
        
        return df
    
    def extract_brand(self, product_name):
        """Extract brand from product name"""
        common_brands = [
            'Apple', 'Samsung', 'OnePlus', 'Xiaomi', 'Realme', 'Oppo', 'Vivo',
            'Google', 'Huawei', 'Honor', 'Nokia', 'Motorola', 'Sony', 'LG',
            'Asus', 'Lenovo', 'HP', 'Dell', 'Acer', 'MSI', 'Canon', 'Nikon'
        ]
        
        product_name_lower = product_name.lower()
        for brand in common_brands:
            if brand.lower() in product_name_lower:
                return brand
        
        # Try to extract first word as brand
        first_word = product_name.split()[0]
        return first_word if len(first_word) > 2 else 'Unknown'
    
    def extract_features_from_specifications(self, specs_json):
        """Extract numerical features from specifications"""
        features = {}
        
        if not specs_json or specs_json == 'null':
            return features
            
        try:
            specs = json.loads(specs_json) if isinstance(specs_json, str) else specs_json
            
            # Extract RAM
            for key, value in specs.items():
                if 'ram' in key.lower() or 'memory' in key.lower():
                    ram_match = re.search(r'(\d+)\s*gb', value.lower())
                    if ram_match:
                        features['ram_gb'] = int(ram_match.group(1))
                        break
            
            # Extract Storage
            for key, value in specs.items():
                if 'storage' in key.lower() or 'internal' in key.lower():
                    storage_match = re.search(r'(\d+)\s*gb', value.lower())
                    if storage_match:
                        features['storage_gb'] = int(storage_match.group(1))
                        break
            
            # Extract Screen Size
            for key, value in specs.items():
                if 'display' in key.lower() or 'screen' in key.lower():
                    screen_match = re.search(r'(\d+\.?\d*)\s*inch', value.lower())
                    if screen_match:
                        features['screen_inches'] = float(screen_match.group(1))
                        break
            
            # Extract Battery
            for key, value in specs.items():
                if 'battery' in key.lower():
                    battery_match = re.search(r'(\d+)\s*mah', value.lower())
                    if battery_match:
                        features['battery_mah'] = int(battery_match.group(1))
                        break
            
            # Extract Camera
            for key, value in specs.items():
                if 'camera' in key.lower() and 'rear' in key.lower():
                    camera_match = re.search(r'(\d+)\s*mp', value.lower())
                    if camera_match:
                        features['camera_mp'] = int(camera_match.group(1))
                        break
                        
        except Exception as e:
            print(f"Error extracting features: {e}")
            
        return features
    
    def save_products_to_database(self, df):
        """Save cleaned products to database"""
        conn = self.connect_to_database()
        cursor = conn.cursor()
        
        try:
            for _, row in df.iterrows():
                # Check if product already exists
                cursor.execute("""
                    SELECT ProductID FROM Products 
                    WHERE Name = ? AND Source = ?
                """, row['name'], row['source'])
                
                existing = cursor.fetchone()
                
                if not existing:
                    # Insert new product
                    cursor.execute("""
                        INSERT INTO Products (Name, Brand, Category, SubCategory, Price, Rating, 
                                            ReviewCount, Specifications, Description, ImageURL, 
                                            ProductURL, Source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, 
                    row['name'], 
                    row.get('brand', ''),
                    row.get('category', ''),
                    row.get('subcategory', ''),
                    row.get('price', 0),
                    row.get('rating', 0),
                    row.get('review_count', 0),
                    row.get('specifications', ''),
                    row.get('description', ''),
                    row.get('image_url', ''),
                    row.get('url', ''),
                    row['source']
                    )
                    
                    # Get the inserted product ID
                    cursor.execute("SELECT @@IDENTITY")
                    product_id = cursor.fetchone()[0]
                    
                    # Extract and save features
                    features = self.extract_features_from_specifications(row.get('specifications', ''))
                    
                    for feature_name, feature_value in features.items():
                        cursor.execute("""
                            INSERT INTO ProductFeatures (ProductID, FeatureName, FeatureValue, NumericValue)
                            VALUES (?, ?, ?, ?)
                        """, product_id, feature_name, str(feature_value), float(feature_value))
            
            conn.commit()
            print(f"Successfully saved {len(df)} products to database")
            
        except Exception as e:
            print(f"Error saving to database: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def create_feature_vectors(self):
        """Create feature vectors for ML model"""
        conn = self.connect_to_database()
        
        # Load products with features
        query = """
        SELECT p.ProductID, p.Name, p.Brand, p.Category, p.Price, p.Rating,
               pf.FeatureName, pf.NumericValue
        FROM Products p
        LEFT JOIN ProductFeatures pf ON p.ProductID = pf.ProductID
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Pivot features to create feature matrix
        feature_df = df.pivot_table(
            index=['ProductID', 'Name', 'Brand', 'Category', 'Price', 'Rating'],
            columns='FeatureName',
            values='NumericValue',
            fill_value=0
        ).reset_index()
        
        # Encode categorical variables
        le_brand = LabelEncoder()
        le_category = LabelEncoder()
        
        feature_df['brand_encoded'] = le_brand.fit_transform(feature_df['Brand'].fillna('Unknown'))
        feature_df['category_encoded'] = le_category.fit_transform(feature_df['Category'].fillna('Unknown'))
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ['Price', 'Rating'] + [col for col in feature_df.columns if col.endswith('_gb') or col.endswith('_mah') or col.endswith('_mp') or col.endswith('_inches')]
        
        if numerical_cols:
            feature_df[numerical_cols] = scaler.fit_transform(feature_df[numerical_cols].fillna(0))
        
        return feature_df, le_brand, le_category, scaler
    
    def process_text_features(self):
        """Process text features using TF-IDF"""
        conn = self.connect_to_database()
        
        query = """
        SELECT ProductID, Name, Description, Specifications
        FROM Products
        WHERE Description IS NOT NULL OR Specifications IS NOT NULL
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Combine text fields
        df['combined_text'] = (
            df['Name'].fillna('') + ' ' + 
            df['Description'].fillna('') + ' ' + 
            df['Specifications'].fillna('')
        )
        
        # Create TF-IDF vectors
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = tfidf.fit_transform(df['combined_text'])
        
        return df, tfidf_matrix, tfidf

def main():
    """Main processing function"""
    processor = DataProcessor()
    
    # Load scraped data (assuming it exists)
    try:
        df = pd.read_csv('scraped_products.csv')
        print(f"Loaded {len(df)} products from CSV")
        
        # Clean the data
        cleaned_df = processor.clean_product_data(df.to_dict('records'))
        print(f"Cleaned data: {len(cleaned_df)} products")
        
        # Save to database
        processor.save_products_to_database(cleaned_df)
        
        # Create feature vectors
        feature_df, le_brand, le_category, scaler = processor.create_feature_vectors()
        print(f"Created feature vectors for {len(feature_df)} products")
        
        # Process text features
        text_df, tfidf_matrix, tfidf = processor.process_text_features()
        print(f"Processed text features for {len(text_df)} products")
        
        print("Data processing completed successfully!")
        
    except FileNotFoundError:
        print("scraped_products.csv not found. Please run the web scraper first.")
    except Exception as e:
        print(f"Error in data processing: {e}")

if __name__ == "__main__":
    main()
