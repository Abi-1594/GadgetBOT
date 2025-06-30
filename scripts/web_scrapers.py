import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import json
import re

class GadgetScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def setup_selenium_driver(self):
        """Setup Selenium WebDriver for dynamic content"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        return webdriver.Chrome(options=chrome_options)

class Gadget360Scraper(GadgetScraper):
    def __init__(self):
        super().__init__()
        self.base_url = "https://gadgets.ndtv.com"
        
    def scrape_mobile_phones(self, pages=5):
        """Scrape mobile phones from Gadget360"""
        products = []
        
        for page in range(1, pages + 1):
            try:
                url = f"{self.base_url}/mobiles/page-{page}"
                response = self.session.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find product containers
                product_cards = soup.find_all('div', class_='rvw-imgbox')
                
                for card in product_cards:
                    try:
                        product = {}
                        
                        # Extract product name
                        name_elem = card.find('h3')
                        if name_elem:
                            product['name'] = name_elem.get_text().strip()
                        
                        # Extract price
                        price_elem = card.find('span', class_='price')
                        if price_elem:
                            price_text = price_elem.get_text().strip()
                            price_match = re.search(r'₹([\d,]+)', price_text)
                            if price_match:
                                product['price'] = float(price_match.group(1).replace(',', ''))
                        
                        # Extract rating
                        rating_elem = card.find('div', class_='rating')
                        if rating_elem:
                            rating_text = rating_elem.get_text().strip()
                            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                            if rating_match:
                                product['rating'] = float(rating_match.group(1))
                        
                        # Extract product URL
                        link_elem = card.find('a')
                        if link_elem:
                            product['url'] = self.base_url + link_elem.get('href')
                        
                        product['source'] = 'Gadget360'
                        product['category'] = 'Mobile Phones'
                        
                        if product.get('name'):
                            products.append(product)
                            
                    except Exception as e:
                        print(f"Error parsing product: {e}")
                        continue
                
                # Add delay to avoid being blocked
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"Error scraping page {page}: {e}")
                continue
                
        return products

class GSMArenaScaper(GadgetScraper):
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.gsmarena.com"
        
    def scrape_phones(self, pages=5):
        """Scrape phones from GSMArena"""
        products = []
        
        for page in range(1, pages + 1):
            try:
                url = f"{self.base_url}/makers.php3?sPage={page}"
                response = self.session.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find phone listings
                phone_links = soup.find_all('a', href=re.compile(r'.*\.php$'))
                
                for link in phone_links[:20]:  # Limit to avoid too many requests
                    try:
                        phone_url = self.base_url + '/' + link.get('href')
                        phone_response = self.session.get(phone_url)
                        phone_soup = BeautifulSoup(phone_response.content, 'html.parser')
                        
                        product = {}
                        
                        # Extract phone name
                        title_elem = phone_soup.find('h1', class_='specs-phone-name-title')
                        if title_elem:
                            product['name'] = title_elem.get_text().strip()
                        
                        # Extract specifications
                        specs = {}
                        spec_tables = phone_soup.find_all('table', cellspacing="0")
                        
                        for table in spec_tables:
                            rows = table.find_all('tr')
                            for row in rows:
                                cells = row.find_all(['td', 'th'])
                                if len(cells) >= 2:
                                    key = cells[0].get_text().strip()
                                    value = cells[1].get_text().strip()
                                    specs[key] = value
                        
                        product['specifications'] = json.dumps(specs)
                        product['source'] = 'GSMArena'
                        product['category'] = 'Mobile Phones'
                        product['url'] = phone_url
                        
                        if product.get('name'):
                            products.append(product)
                        
                        # Add delay
                        time.sleep(random.uniform(2, 4))
                        
                    except Exception as e:
                        print(f"Error scraping phone details: {e}")
                        continue
                
            except Exception as e:
                print(f"Error scraping GSMArena page {page}: {e}")
                continue
                
        return products

class AmazonScraper(GadgetScraper):
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.amazon.in"
        
    def scrape_electronics(self, category="smartphones", pages=3):
        """Scrape electronics from Amazon"""
        products = []
        driver = self.setup_selenium_driver()
        
        try:
            for page in range(1, pages + 1):
                search_url = f"{self.base_url}/s?k={category}&page={page}"
                driver.get(search_url)
                time.sleep(3)
                
                # Find product containers
                product_containers = driver.find_elements(By.CSS_SELECTOR, '[data-component-type="s-search-result"]')
                
                for container in product_containers:
                    try:
                        product = {}
                        
                        # Extract product name
                        name_elem = container.find_element(By.CSS_SELECTOR, 'h2 a span')
                        product['name'] = name_elem.text.strip()
                        
                        # Extract price
                        try:
                            price_elem = container.find_element(By.CSS_SELECTOR, '.a-price-whole')
                            price_text = price_elem.text.replace(',', '')
                            product['price'] = float(price_text)
                        except:
                            pass
                        
                        # Extract rating
                        try:
                            rating_elem = container.find_element(By.CSS_SELECTOR, '.a-icon-alt')
                            rating_text = rating_elem.get_attribute('textContent')
                            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                            if rating_match:
                                product['rating'] = float(rating_match.group(1))
                        except:
                            pass
                        
                        # Extract product URL
                        try:
                            link_elem = container.find_element(By.CSS_SELECTOR, 'h2 a')
                            product['url'] = self.base_url + link_elem.get_attribute('href')
                        except:
                            pass
                        
                        product['source'] = 'Amazon'
                        product['category'] = category.title()
                        
                        if product.get('name'):
                            products.append(product)
                            
                    except Exception as e:
                        print(f"Error parsing Amazon product: {e}")
                        continue
                
                time.sleep(random.uniform(2, 4))
                
        finally:
            driver.quit()
            
        return products

class FlipkartScraper(GadgetScraper):
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.flipkart.com"
        
    def scrape_mobiles(self, pages=3):
        """Scrape mobiles from Flipkart"""
        products = []
        driver = self.setup_selenium_driver()
        
        try:
            for page in range(1, pages + 1):
                url = f"{self.base_url}/search?q=smartphones&page={page}"
                driver.get(url)
                time.sleep(3)
                
                # Find product containers
                product_containers = driver.find_elements(By.CSS_SELECTOR, '._1AtVbE')
                
                for container in product_containers:
                    try:
                        product = {}
                        
                        # Extract product name
                        name_elem = container.find_element(By.CSS_SELECTOR, '._4rR01T')
                        product['name'] = name_elem.text.strip()
                        
                        # Extract price
                        try:
                            price_elem = container.find_element(By.CSS_SELECTOR, '._30jeq3')
                            price_text = price_elem.text.replace('₹', '').replace(',', '')
                            product['price'] = float(price_text)
                        except:
                            pass
                        
                        # Extract rating
                        try:
                            rating_elem = container.find_element(By.CSS_SELECTOR, '._3LWZlK')
                            product['rating'] = float(rating_elem.text)
                        except:
                            pass
                        
                        product['source'] = 'Flipkart'
                        product['category'] = 'Mobile Phones'
                        
                        if product.get('name'):
                            products.append(product)
                            
                    except Exception as e:
                        print(f"Error parsing Flipkart product: {e}")
                        continue
                
                time.sleep(random.uniform(2, 4))
                
        finally:
            driver.quit()
            
        return products

def run_all_scrapers():
    """Run all scrapers and return combined results"""
    all_products = []
    
    # Scrape from Gadget360
    print("Scraping Gadget360...")
    gadget360_scraper = Gadget360Scraper()
    gadget360_products = gadget360_scraper.scrape_mobile_phones(pages=2)
    all_products.extend(gadget360_products)
    print(f"Scraped {len(gadget360_products)} products from Gadget360")
    
    # Scrape from GSMArena
    print("Scraping GSMArena...")
    gsmarena_scraper = GSMArenaScaper()
    gsmarena_products = gsmarena_scraper.scrape_phones(pages=2)
    all_products.extend(gsmarena_products)
    print(f"Scraped {len(gsmarena_products)} products from GSMArena")
    
    # Scrape from Amazon
    print("Scraping Amazon...")
    amazon_scraper = AmazonScraper()
    amazon_products = amazon_scraper.scrape_electronics(pages=2)
    all_products.extend(amazon_products)
    print(f"Scraped {len(amazon_products)} products from Amazon")
    
    # Scrape from Flipkart
    print("Scraping Flipkart...")
    flipkart_scraper = FlipkartScraper()
    flipkart_products = flipkart_scraper.scrape_mobiles(pages=2)
    all_products.extend(flipkart_products)
    print(f"Scraped {len(flipkart_products)} products from Flipkart")
    
    print(f"Total products scraped: {len(all_products)}")
    return all_products

if __name__ == "__main__":
    products = run_all_scrapers()
    
    # Save to CSV for inspection
    df = pd.DataFrame(products)
    df.to_csv('scraped_products.csv', index=False)
    print("Products saved to scraped_products.csv")
