import pyodbc
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseSetup:
    def __init__(self):
        self.connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={os.getenv('MSSQL_SERVER', 'localhost')};"
            f"DATABASE={os.getenv('MSSQL_DATABASE', 'GadgetRecommendations')};"
            f"UID={os.getenv('MSSQL_USERNAME', 'sa')};"
            f"PWD={os.getenv('MSSQL_PASSWORD', 'your_password')}"
        )
    
    def create_database_schema(self):
        """Create all necessary tables for the recommendation engine"""
        
        tables_sql = [
            # Products table
            """
            CREATE TABLE IF NOT EXISTS Products (
                ProductID INT IDENTITY(1,1) PRIMARY KEY,
                Name NVARCHAR(500) NOT NULL,
                Brand NVARCHAR(100),
                Category NVARCHAR(100),
                SubCategory NVARCHAR(100),
                Price DECIMAL(10,2),
                Rating DECIMAL(3,2),
                ReviewCount INT,
                Specifications NVARCHAR(MAX),
                Description NVARCHAR(MAX),
                ImageURL NVARCHAR(500),
                ProductURL NVARCHAR(500),
                Source NVARCHAR(50),
                CreatedDate DATETIME DEFAULT GETDATE(),
                UpdatedDate DATETIME DEFAULT GETDATE()
            )
            """,
            
            # User interactions table
            """
            CREATE TABLE IF NOT EXISTS UserInteractions (
                InteractionID INT IDENTITY(1,1) PRIMARY KEY,
                UserID NVARCHAR(100),
                ProductID INT,
                InteractionType NVARCHAR(50), -- view, like, purchase, search
                InteractionValue DECIMAL(5,2), -- rating or relevance score
                Timestamp DATETIME DEFAULT GETDATE(),
                FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
            )
            """,
            
            # Product features for ML
            """
            CREATE TABLE IF NOT EXISTS ProductFeatures (
                FeatureID INT IDENTITY(1,1) PRIMARY KEY,
                ProductID INT,
                FeatureName NVARCHAR(100),
                FeatureValue NVARCHAR(500),
                NumericValue DECIMAL(15,4),
                FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
            )
            """,
            
            # Recommendations cache
            """
            CREATE TABLE IF NOT EXISTS RecommendationCache (
                CacheID INT IDENTITY(1,1) PRIMARY KEY,
                UserID NVARCHAR(100),
                ProductID INT,
                RecommendationScore DECIMAL(5,4),
                RecommendationType NVARCHAR(50),
                CreatedDate DATETIME DEFAULT GETDATE(),
                FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
            )
            """,
            
            # Search logs
            """
            CREATE TABLE IF NOT EXISTS SearchLogs (
                LogID INT IDENTITY(1,1) PRIMARY KEY,
                UserID NVARCHAR(100),
                SearchQuery NVARCHAR(500),
                ResultCount INT,
                Timestamp DATETIME DEFAULT GETDATE()
            )
            """
        ]
        
        try:
            conn = pyodbc.connect(self.connection_string)
            cursor = conn.cursor()
            
            for sql in tables_sql:
                cursor.execute(sql)
                conn.commit()
                
            print("Database schema created successfully!")
            
            # Create indexes for better performance
            indexes_sql = [
                "CREATE INDEX IX_Products_Brand ON Products(Brand)",
                "CREATE INDEX IX_Products_Category ON Products(Category)",
                "CREATE INDEX IX_Products_Price ON Products(Price)",
                "CREATE INDEX IX_Products_Rating ON Products(Rating)",
                "CREATE INDEX IX_UserInteractions_UserID ON UserInteractions(UserID)",
                "CREATE INDEX IX_UserInteractions_ProductID ON UserInteractions(ProductID)",
                "CREATE INDEX IX_ProductFeatures_ProductID ON ProductFeatures(ProductID)"
            ]
            
            for index_sql in indexes_sql:
                try:
                    cursor.execute(index_sql)
                    conn.commit()
                except:
                    pass  # Index might already exist
                    
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error creating database schema: {e}")

if __name__ == "__main__":
    db_setup = DatabaseSetup()
    db_setup.create_database_schema()
