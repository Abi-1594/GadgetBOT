import os
import schedule
import time
from datetime import datetime
from web_scrapers import run_all_scrapers
from data_processor import DataProcessor
from recommendation_engine import RecommendationEngine
import pandas as pd
import json

class GadgetRecommendationApp:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.recommendation_engine = RecommendationEngine()
        
    def full_data_refresh(self):
        """Complete data refresh pipeline"""
        print(f"Starting full data refresh at {datetime.now()}")
        
        try:
            # Step 1: Scrape new data
            print("Step 1: Scraping data from all sources...")
            scraped_products = run_all_scrapers()
            
            if scraped_products:
                # Save scraped data
                df = pd.DataFrame(scraped_products)
                df.to_csv('scraped_products.csv', index=False)
                print(f"Scraped {len(scraped_products)} products")
                
                # Step 2: Process and clean data
                print("Step 2: Processing and cleaning data...")
                cleaned_df = self.data_processor.clean_product_data(scraped_products)
                
                # Step 3: Save to database
                print("Step 3: Saving to database...")
                self.data_processor.save_products_to_database(cleaned_df)
                
                # Step 4: Retrain recommendation models
                print("Step 4: Training recommendation models...")
                self.recommendation_engine.train_all_models()
                
                print("Full data refresh completed successfully!")
                
            else:
                print("No products scraped. Skipping data refresh.")
                
        except Exception as e:
            print(f"Error during data refresh: {e}")
    
    def get_recommendations_for_user(self, user_id, num_recommendations=10):
        """Get personalized recommendations for a user"""
        try:
            recommendations = self.recommendation_engine.get_hybrid_recommendations(
                user_id=user_id,
                num_recommendations=num_recommendations
            )
            
            # Save to cache
            self.recommendation_engine.save_recommendations_to_cache(user_id, recommendations)
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations for user {user_id}: {e}")
            return []
    
    def get_similar_products(self, product_id, num_recommendations=10):
        """Get similar products for a given product"""
        try:
            recommendations = self.recommendation_engine.get_content_based_recommendations(
                product_id=product_id,
                num_recommendations=num_recommendations
            )
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting similar products for product {product_id}: {e}")
            return []
    
    def search_products(self, query, category=None, min_price=None, max_price=None, min_rating=None):
        """Search products with filters"""
        import pyodbc
        
        conn = self.data_processor.connect_to_database()
        
        # Build search query
        sql = """
        SELECT ProductID, Name, Brand, Category, Price, Rating, Source, ProductURL
        FROM Products
        WHERE Name LIKE ?
        """
        params = [f'%{query}%']
        
        if category:
            sql += " AND Category = ?"
            params.append(category)
            
        if min_price:
            sql += " AND Price >= ?"
            params.append(min_price)
            
        if max_price:
            sql += " AND Price <= ?"
            params.append(max_price)
            
        if min_rating:
            sql += " AND Rating >= ?"
            params.append(min_rating)
        
        sql += " ORDER BY Rating DESC, Price ASC"
        
        try:
            df = pd.read_sql(sql, conn, params=params)
            conn.close()
            
            return df.to_dict('records')
            
        except Exception as e:
            print(f"Error searching products: {e}")
            conn.close()
            return []
    
    def get_trending_products(self, category=None, days=7):
        """Get trending products based on recent interactions"""
        import pyodbc
        from datetime import datetime, timedelta
        
        conn = self.data_processor.connect_to_database()
        
        sql = """
        SELECT p.ProductID, p.Name, p.Brand, p.Category, p.Price, p.Rating,
               COUNT(ui.InteractionID) as InteractionCount,
               AVG(ui.InteractionValue) as AvgInteractionValue
        FROM Products p
        LEFT JOIN UserInteractions ui ON p.ProductID = ui.ProductID
        WHERE ui.Timestamp >= ?
        """
        params = [datetime.now() - timedelta(days=days)]
        
        if category:
            sql += " AND p.Category = ?"
            params.append(category)
        
        sql += """
        GROUP BY p.ProductID, p.Name, p.Brand, p.Category, p.Price, p.Rating
        HAVING COUNT(ui.InteractionID) > 0
        ORDER BY InteractionCount DESC, AvgInteractionValue DESC
        """
        
        try:
            df = pd.read_sql(sql, conn, params=params)
            conn.close()
            
            return df.to_dict('records')
            
        except Exception as e:
            print(f"Error getting trending products: {e}")
            conn.close()
            return []
    
    def log_user_interaction(self, user_id, product_id, interaction_type, interaction_value=None):
        """Log user interaction for improving recommendations"""
        import pyodbc
        
        conn = self.data_processor.connect_to_database()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO UserInteractions (UserID, ProductID, InteractionType, InteractionValue)
                VALUES (?, ?, ?, ?)
            """, user_id, product_id, interaction_type, interaction_value)
            
            conn.commit()
            print(f"Logged interaction: {user_id} -> {product_id} ({interaction_type})")
            
        except Exception as e:
            print(f"Error logging interaction: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def get_product_details(self, product_id):
        """Get detailed information about a product"""
        import pyodbc
        
        conn = self.data_processor.connect_to_database()
        
        sql = """
        SELECT p.*, 
               STRING_AGG(pf.FeatureName + ': ' + pf.FeatureValue, '; ') as Features
        FROM Products p
        LEFT JOIN ProductFeatures pf ON p.ProductID = pf.ProductID
        WHERE p.ProductID = ?
        GROUP BY p.ProductID, p.Name, p.Brand, p.Category, p.SubCategory, 
                 p.Price, p.Rating, p.ReviewCount, p.Specifications, 
                 p.Description, p.ImageURL, p.ProductURL, p.Source, 
                 p.CreatedDate, p.UpdatedDate
        """
        
        try:
            df = pd.read_sql(sql, conn, params=[product_id])
            conn.close()
            
            if not df.empty:
                return df.iloc[0].to_dict()
            else:
                return None
                
        except Exception as e:
            print(f"Error getting product details: {e}")
            conn.close()
            return None
    
    def schedule_data_refresh(self):
        """Schedule automatic data refresh"""
        # Schedule daily data refresh at 2 AM
        schedule.every().day.at("02:00").do(self.full_data_refresh)
        
        print("Scheduled daily data refresh at 2:00 AM")
        
        # Keep the scheduler running
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
    
    def run_interactive_demo(self):
        """Run interactive demo of the recommendation system"""
        print("=== Gadget Recommendation Engine Demo ===")
        
        while True:
            print("\nOptions:")
            print("1. Get recommendations for user")
            print("2. Find similar products")
            print("3. Search products")
            print("4. Get trending products")
            print("5. Get product details")
            print("6. Log user interaction")
            print("7. Refresh data")
            print("8. Exit")
            
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                user_id = input("Enter user ID: ").strip()
                num_recs = int(input("Number of recommendations (default 10): ") or 10)
                
                recommendations = self.get_recommendations_for_user(user_id, num_recs)
                
                print(f"\nRecommendations for {user_id}:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec['name']} - {rec['brand']} (₹{rec['price']}) [{rec['recommendation_type']}]")
            
            elif choice == '2':
                product_id = int(input("Enter product ID: "))
                num_recs = int(input("Number of similar products (default 10): ") or 10)
                
                similar_products = self.get_similar_products(product_id, num_recs)
                
                print(f"\nSimilar products:")
                for i, rec in enumerate(similar_products, 1):
                    print(f"{i}. {rec['name']} - {rec['brand']} (Similarity: {rec['similarity_score']:.3f})")
            
            elif choice == '3':
                query = input("Enter search query: ").strip()
                category = input("Category (optional): ").strip() or None
                min_price = input("Min price (optional): ").strip()
                max_price = input("Max price (optional): ").strip()
                
                min_price = float(min_price) if min_price else None
                max_price = float(max_price) if max_price else None
                
                results = self.search_products(query, category, min_price, max_price)
                
                print(f"\nSearch results for '{query}':")
                for i, product in enumerate(results[:10], 1):
                    print(f"{i}. {product['Name']} - {product['Brand']} (₹{product['Price']}) - Rating: {product['Rating']}")
            
            elif choice == '4':
                category = input("Category (optional): ").strip() or None
                days = int(input("Days to look back (default 7): ") or 7)
                
                trending = self.get_trending_products(category, days)
                
                print(f"\nTrending products:")
                for i, product in enumerate(trending[:10], 1):
                    print(f"{i}. {product['Name']} - {product['Brand']} ({product['InteractionCount']} interactions)")
            
            elif choice == '5':
                product_id = int(input("Enter product ID: "))
                
                details = self.get_product_details(product_id)
                
                if details:
                    print(f"\nProduct Details:")
                    print(f"Name: {details['Name']}")
                    print(f"Brand: {details['Brand']}")
                    print(f"Category: {details['Category']}")
                    print(f"Price: ₹{details['Price']}")
                    print(f"Rating: {details['Rating']}")
                    print(f"Source: {details['Source']}")
                    if details.get('Features'):
                        print(f"Features: {details['Features']}")
                else:
                    print("Product not found.")
            
            elif choice == '6':
                user_id = input("Enter user ID: ").strip()
                product_id = int(input("Enter product ID: "))
                interaction_type = input("Interaction type (view/like/purchase): ").strip()
                interaction_value = input("Rating (1-5, optional): ").strip()
                
                interaction_value = float(interaction_value) if interaction_value else None
                
                self.log_user_interaction(user_id, product_id, interaction_type, interaction_value)
            
            elif choice == '7':
                print("Starting data refresh...")
                self.full_data_refresh()
            
            elif choice == '8':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")

def main():
    """Main application entry point"""
    app = GadgetRecommendationApp()
    
    print("Gadget Recommendation Engine")
    print("============================")
    
    # Initialize the system
    print("Initializing recommendation engine...")
    
    # Check if we have data, if not run initial setup
    try:
        # Try to train models
        app.recommendation_engine.train_all_models()
        print("Models loaded successfully!")
    except:
        print("No data found. Running initial data collection...")
        app.full_data_refresh()
    
    # Run interactive demo
    app.run_interactive_demo()

if __name__ == "__main__":
    main()
