# src/api_ingestion.py
"""
Fetch large datasets from APIs incrementally with rate limiting and error handling.
Optimized for 100K+ records.
"""

import requests
import time
import json
from typing import List, Dict, Generator, Optional
from datetime import datetime
import os

class APIDataFetcher:
    """Fetch data from paginated APIs with retry logic."""
    
    def __init__(
        self, 
        base_url: str,
        api_key: Optional[str] = None,
        rate_limit_per_second: int = 10,
        batch_size: int = 100
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.rate_limit = rate_limit_per_second
        self.batch_size = batch_size
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def fetch_paginated(
        self, 
        endpoint: str,
        total_records: Optional[int] = None,
        max_retries: int = 3
    ) -> Generator[List[Dict], None, None]:
        """
        Fetch data in batches with pagination.
        Yields batches of records.
        """
        offset = 0
        records_fetched = 0
        
        while True:
            # Rate limiting
            time.sleep(1.0 / self.rate_limit)
            
            # Build URL with pagination
            url = f"{self.base_url}/{endpoint}"
            params = {
                'limit': self.batch_size,
                'offset': offset
            }
            
            # Retry logic
            for attempt in range(max_retries):
                try:
                    response = self.session.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract records (adjust based on your API structure)
                    records = data.get('results', data.get('data', []))
                    
                    if not records:
                        return  # No more data
                    
                    records_fetched += len(records)
                    print(f"✓ Fetched batch: {records_fetched} records total")
                    
                    yield records
                    
                    offset += self.batch_size
                    
                    # Check if we've reached the limit
                    if total_records and records_fetched >= total_records:
                        return
                    
                    break  # Success, exit retry loop
                    
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"⚠ Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        print(f"✗ Failed after {max_retries} attempts: {e}")
                        raise
    
    def fetch_all(self, endpoint: str, output_file: str = 'data/api_data.jsonl'):
        """Fetch all data and save to JSONL file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for batch in self.fetch_paginated(endpoint):
                for record in batch:
                    f.write(json.dumps(record) + '\n')
        
        print(f"\n✓ All data saved to {output_file}")


# Example usage for different API types
class CustomAPIFetcher:
    """Adapt this for your specific API structure."""
    
    @staticmethod
    def fetch_rest_api(url: str, api_key: str = None) -> Generator[Dict, None, None]:
        """Generic REST API fetcher."""
        fetcher = APIDataFetcher(url, api_key, rate_limit_per_second=5, batch_size=100)
        
        for batch in fetcher.fetch_paginated('items'):
            for record in batch:
                # Transform record to consistent format
                yield {
                    'id': record.get('id'),
                    'text': record.get('content', record.get('description', '')),
                    'metadata': {
                        'title': record.get('title', ''),
                        'url': record.get('url', ''),
                        'created_at': record.get('created_at', ''),
                        'category': record.get('category', '')
                    }
                }
    
    @staticmethod
    def fetch_from_database(connection_string: str) -> Generator[Dict, None, None]:
        """Fetch from SQL database in batches."""
        import sqlite3  # or psycopg2 for PostgreSQL
        
        conn = sqlite3.connect(connection_string)
        cursor = conn.cursor()
        
        batch_size = 1000
        offset = 0
        
        while True:
            cursor.execute(
                f"SELECT id, content, title, url FROM documents LIMIT {batch_size} OFFSET {offset}"
            )
            rows = cursor.fetchall()
            
            if not rows:
                break
            
            for row in rows:
                yield {
                    'id': row[0],
                    'text': row[1],
                    'metadata': {
                        'title': row[2],
                        'url': row[3]
                    }
                }
            
            offset += batch_size
            print(f"✓ Processed {offset} records from database")
        
        conn.close()


if __name__ == '__main__':
    # Example: Fetch from a test API
    print("Testing API fetcher...")
    
    # Replace with your actual API
    fetcher = APIDataFetcher(
        base_url='https://jsonplaceholder.typicode.com',
        batch_size=10
    )
    
    count = 0
    for batch in fetcher.fetch_paginated('posts', total_records=50):
        count += len(batch)
        print(f"Batch fetched: {len(batch)} items")
    
    print(f"\nTotal records fetched: {count}")