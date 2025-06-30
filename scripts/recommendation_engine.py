import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import pyodbc
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class RecommendationEngine:
    def __init__(self):
        self.connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={os.getenv('MSSQL_SERVER', 'localhost')};"
            f"DATABASE={os.getenv('MSSQL_DATABASE', 'GadgetRecommendations')};"
            f"UID={os.getenv('MSSQL_USERNAME', 'sa')};"
            f"PWD={os.getenv('MSSQL_PASSWORD', 'your_password')}"
        )
        self.models = {}
        
    def connect_to_database(self):
        """Establish database connection"""
        return pyodbc.connect(self.connection_string)
    
    def load_product_data(self):
        """Load product data with features"""
        conn = self.connect_to_database()
        
        query = """
        SELECT p.ProductID, p.Name, p.Brand, p.Category, p.SubCategory,
               p.Price, p.Rating, p.ReviewCount, p.Source,
               pf.FeatureName, pf.NumericValue
        FROM Products p
        LEFT JOIN ProductFeatures pf ON p.ProductID = pf.ProductID
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        return df
    
    def create_content_based_model(self):
        """Create content-based recommendation model"""
        df = self.load_product_data()
        
        # Create feature matrix
        feature_matrix = df.pivot_table(
            index=['ProductID', 'Name', 'Brand', 'Category', 'Price', 'Rating'],
            columns='FeatureName',
            values='NumericValue',
            fill_value=0
        ).reset_index()
        
        # Normalize features
        feature_cols = [col for col in feature_matrix.columns if col not in ['ProductID', 'Name', 'Brand', 'Category']]
        
        if feature_cols:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_matrix[feature_cols] = scaler.fit_transform(feature_matrix[feature_cols])
        
        # Create similarity matrix
        similarity_matrix = cosine_similarity(feature_matrix[feature_cols])
        
        self.models['content_based'] = {
            'feature_matrix': feature_matrix,
            'similarity_matrix': similarity_matrix,
            'feature_cols': feature_cols
        }
        
        return feature_matrix, similarity_matrix
    
    def create_collaborative_filtering_model(self):
        """Create collaborative filtering model using user interactions"""
        conn = self.connect_to_database()
        
        query = """
        SELECT UserID, ProductID, InteractionValue
        FROM UserInteractions
        WHERE InteractionValue IS NOT NULL
        """
        
        interactions_df = pd.read_sql(query, conn)
        conn.close()
        
        if len(interactions_df) == 0:
            print("No user interactions found. Generating sample data...")
            interactions_df = self.generate_sample_interactions()
        
        # Create user-item matrix
        user_item_matrix = interactions_df.pivot_table(
            index='UserID',
            columns='ProductID',
            values='InteractionValue',
            fill_value=0
        )
        
        # Apply SVD for dimensionality reduction
        svd = TruncatedSVD(n_components=50, random_state=42)
        user_factors = svd.fit_transform(user_item_matrix)
        item_factors = svd.components_.T
        
        self.models['collaborative'] = {
            'user_item_matrix': user_item_matrix,
            'svd': svd,
            'user_factors': user_factors,
            'item_factors': item_factors
        }
        
        return user_item_matrix, svd
    
    def generate_sample_interactions(self):
        """Generate sample user interactions for testing"""
        conn = self.connect_to_database()
        
        # Get product IDs
        products_df = pd.read_sql("SELECT ProductID FROM Products", conn)
        product_ids = products_df['ProductID'].tolist()
        
        # Generate sample interactions
        interactions = []
        for user_id in range(1, 101):  # 100 sample users
            # Each user interacts with 5-15 random products
            num_interactions = np.random.randint(5, 16)
            user_products = np.random.choice(product_ids, num_interactions, replace=False)
            
            for product_id in user_products:
                interaction = {
                    'UserID': f'user_{user_id}',
                    'ProductID': int(product_id),
                    'InteractionValue': np.random.uniform(1, 5),  # Rating between 1-5
                    'InteractionType': np.random.choice(['view', 'like', 'purchase'], p=[0.6, 0.3, 0.1])
                }
                interactions.append(interaction)
        
        # Save to database
        cursor = conn.cursor()
        for interaction in interactions:
            cursor.execute("""
                INSERT INTO UserInteractions (UserID, ProductID, InteractionType, InteractionValue)
                VALUES (?, ?, ?, ?)
            """, interaction['UserID'], interaction['ProductID'], 
                interaction['InteractionType'], interaction['InteractionValue'])
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return pd.DataFrame(interactions)
    
    def get_content_based_recommendations(self, product_id, num_recommendations=10):
        """Get content-based recommendations for a product"""
        if 'content_based' not in self.models:
            self.create_content_based_model()
        
        model = self.models['content_based']
        feature_matrix = model['feature_matrix']
        similarity_matrix = model['similarity_matrix']
        
        # Find product index
        try:
            product_idx = feature_matrix[feature_matrix['ProductID'] == product_id].index[0]
        except IndexError:
            return []
        
        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[product_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top recommendations (excluding the product itself)
        recommendations = []
        for i, score in sim_scores[1:num_recommendations+1]:
            rec_product = feature_matrix.iloc[i]
            recommendations.append({
                'product_id': int(rec_product['ProductID']),
                'name': rec_product['Name'],
                'brand': rec_product['Brand'],
                'category': rec_product['Category'],
                'price': rec_product['Price'],
                'rating': rec_product['Rating'],
                'similarity_score': float(score),
                'recommendation_type': 'content_based'
            })
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, num_recommendations=10):
        """Get collaborative filtering recommendations for a user"""
        if 'collaborative' not in self.models:
            self.create_collaborative_filtering_model()
        
        model = self.models['collaborative']
        user_item_matrix = model['user_item_matrix']
        svd = model['svd']
        
        if user_id not in user_item_matrix.index:
            return self.get_popular_recommendations(num_recommendations)
        
        # Get user vector
        user_idx = user_item_matrix.index.get_loc(user_id)
        user_vector = model['user_factors'][user_idx]
        
        # Calculate scores for all items
        scores = np.dot(user_vector, model['item_factors'].T)
        
        # Get products user hasn't interacted with
        user_interactions = user_item_matrix.loc[user_id]
        unrated_items = user_interactions[user_interactions == 0].index
        
        # Get recommendations
        recommendations = []
        item_scores = [(item_id, scores[i]) for i, item_id in enumerate(user_item_matrix.columns) if item_id in unrated_items]
        item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)
        
        # Get product details
        conn = self.connect_to_database()
        for product_id, score in item_scores[:num_recommendations]:
            query = """
            SELECT ProductID, Name, Brand, Category, Price, Rating
            FROM Products
            WHERE ProductID = ?
            """
            product_df = pd.read_sql(query, conn, params=[product_id])
            
            if not product_df.empty:
                product = product_df.iloc[0]
                recommendations.append({
                    'product_id': int(product['ProductID']),
                    'name': product['Name'],
                    'brand': product['Brand'],
                    'category': product['Category'],
                    'price': product['Price'],
                    'rating': product['Rating'],
                    'predicted_score': float(score),
                    'recommendation_type': 'collaborative'
                })
        
        conn.close()
        return recommendations
    
    def get_popular_recommendations(self, num_recommendations=10):
        """Get popular products as fallback recommendations"""
        conn = self.connect_to_database()
        
        query = """
        SELECT TOP (?) ProductID, Name, Brand, Category, Price, Rating, ReviewCount
        FROM Products
        WHERE Rating > 0
        ORDER BY (Rating * LOG(ReviewCount + 1)) DESC
        """
        
        df = pd.read_sql(query, conn, params=[num_recommendations])
        conn.close()
        
        recommendations = []
        for _, product in df.iterrows():
            recommendations.append({
                'product_id': int(product['ProductID']),
                'name': product['Name'],
                'brand': product['Brand'],
                'category': product['Category'],
                'price': product['Price'],
                'rating': product['Rating'],
                'popularity_score': float(product['Rating'] * np.log(product['ReviewCount'] + 1)),
                'recommendation_type': 'popular'
            })
        
        return recommendations
    
    def get_hybrid_recommendations(self, user_id=None, product_id=None, num_recommendations=10):
        """Get hybrid recommendations combining multiple approaches"""
        recommendations = []
        
        # Get content-based recommendations if product_id provided
        if product_id:
            content_recs = self.get_content_based_recommendations(product_id, num_recommendations//2)
            recommendations.extend(content_recs)
        
        # Get collaborative recommendations if user_id provided
        if user_id:
            collab_recs = self.get_collaborative_recommendations(user_id, num_recommendations//2)
            recommendations.extend(collab_recs)
        
        # Fill remaining slots with popular recommendations
        if len(recommendations) < num_recommendations:
            popular_recs = self.get_popular_recommendations(num_recommendations - len(recommendations))
            recommendations.extend(popular_recs)
        
        # Remove duplicates and limit results
        seen_products = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec['product_id'] not in seen_products:
                seen_products.add(rec['product_id'])
                unique_recommendations.append(rec)
                
                if len(unique_recommendations) >= num_recommendations:
                    break
        
        return unique_recommendations
    
    def save_recommendations_to_cache(self, user_id, recommendations):
        """Save recommendations to cache table"""
        conn = self.connect_to_database()
        cursor = conn.cursor()
        
        try:
            # Clear existing cache for user
            cursor.execute("DELETE FROM RecommendationCache WHERE UserID = ?", user_id)
            
            # Insert new recommendations
            for rec in recommendations:
                cursor.execute("""
                    INSERT INTO RecommendationCache (UserID, ProductID, RecommendationScore, RecommendationType)
                    VALUES (?, ?, ?, ?)
                """, user_id, rec['product_id'], 
                    rec.get('similarity_score', rec.get('predicted_score', rec.get('popularity_score', 0))),
                    rec['recommendation_type'])
            
            conn.commit()
            
        except Exception as e:
            print(f"Error saving recommendations to cache: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def train_all_models(self):
        """Train all recommendation models"""
        print("Training content-based model...")
        self.create_content_based_model()
        
        print("Training collaborative filtering model...")
        self.create_collaborative_filtering_model()
        
        print("All models trained successfully!")

def main():
    """Main function to test the recommendation engine"""
    engine = RecommendationEngine()
    
    # Train models
    engine.train_all_models()
    
    # Test content-based recommendations
    print("\n=== Content-Based Recommendations ===")
    content_recs = engine.get_content_based_recommendations(product_id=1, num_recommendations=5)
    for i, rec in enumerate(content_recs, 1):
        print(f"{i}. {rec['name']} - {rec['brand']} (Score: {rec['similarity_score']:.3f})")
    
    # Test collaborative recommendations
    print("\n=== Collaborative Filtering Recommendations ===")
    collab_recs = engine.get_collaborative_recommendations(user_id='user_1', num_recommendations=5)
    for i, rec in enumerate(collab_recs, 1):
        print(f"{i}. {rec['name']} - {rec['brand']} (Score: {rec['predicted_score']:.3f})")
    
    # Test hybrid recommendations
    print("\n=== Hybrid Recommendations ===")
    hybrid_recs = engine.get_hybrid_recommendations(user_id='user_1', product_id=1, num_recommendations=5)
    for i, rec in enumerate(hybrid_recs, 1):
        print(f"{i}. {rec['name']} - {rec['brand']} ({rec['recommendation_type']})")

if __name__ == "__main__":
    main()
