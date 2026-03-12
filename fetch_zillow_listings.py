"""
Fetch all Zillow listings in Seattle using the API with pagination
"""
import requests
import json
import time
import ssl
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_zillow_page(page_num, headers):
    """Fetch a single page of Zillow listings"""
    
    url = "https://api.hasdata.com/scrape/zillow/listing"
    
    params = {
        'keyword': 'Seattle, WA',
        'type': 'forSale',
        'sort': 'newest',
        'page': page_num,
        'homeTypes%5B%5D' : ['house', 'townhome']
    }
    
    print(f"\n  Fetching page {page_num}...")
    
    try:
        response = requests.get(url, headers=headers, params=params, verify=False, timeout=30)
        
        if response.status_code != 200:
            print(f"    ⚠ Error: Status code {response.status_code}")
            return None
        
        response_json = response.json()
        properties = response_json.get('properties', [])
        
        print(f"    ✓ Got {len(properties)} properties")
        
        return response_json
        
    except Exception as e:
        print(f"    ❌ Error fetching page {page_num}: {e}")
        return None


def fetch_zillow_listings(max_pages=10, delay_seconds=2):
    """
    Fetch multiple pages of Seattle listings from Zillow API
    
    Args:
        max_pages: Maximum number of pages to fetch (default: 10)
        delay_seconds: Delay between requests to avoid rate limiting (default: 2)
    """
    
    print("="*80)
    print("FETCHING ZILLOW LISTINGS FOR SEATTLE (MULTI-PAGE)")
    print("="*80)
    
    headers = {
        'x-api-key': "a33bc008-3e10-4785-a44f-3924e2391e0d",
        'Content-Type': "application/json"
    }
    
    print(f"\nConfiguration:")
    print(f"  Max pages: {max_pages}")
    print(f"  Delay between requests: {delay_seconds}s")
    
    all_properties = []
    total_results = None
    pages_fetched = 0
    
    try:
        for page_num in range(1, max_pages + 1):
            # Fetch page
            response_json = fetch_zillow_page(page_num, headers)
            
            if response_json is None:
                print(f"\n  Stopping at page {page_num} due to error")
                break
            
            # Extract properties
            properties = response_json.get('properties', [])
            
            if not properties:
                print(f"\n  No more properties found at page {page_num}")
                break
            
            # Add to collection
            all_properties.extend(properties)
            pages_fetched += 1
            
            # Get total results count (from first page)
            if total_results is None and 'searchInformation' in response_json:
                total_results = response_json['searchInformation'].get('totalResults', 'unknown')
                print(f"\n  Total results available: {total_results}")
            
            # Check if we've reached the end
            if total_results and isinstance(total_results, int):
                if len(all_properties) >= total_results:
                    print(f"\n  Fetched all available properties")
                    break
            
            # Delay before next request (except for last page)
            if page_num < max_pages:
                time.sleep(delay_seconds)
        
        print(f"\n{'='*80}")
        print(f"FETCH COMPLETE")
        print(f"{'='*80}")
        print(f"\n  Pages fetched: {pages_fetched}")
        print(f"  Total properties: {len(all_properties)}")
        print(f"  Total available: {total_results}")
        
        # Remove duplicates based on property ID
        unique_properties = {}
        for prop in all_properties:
            prop_id = prop.get('id')
            if prop_id and prop_id not in unique_properties:
                unique_properties[prop_id] = prop
        
        if len(unique_properties) < len(all_properties):
            print(f"  Removed {len(all_properties) - len(unique_properties)} duplicates")
            all_properties = list(unique_properties.values())
            print(f"  Unique properties: {len(all_properties)}")
        
        # Create combined response
        combined_response = {
            'requestMetadata': {
                'status': 'ok',
                'pages_fetched': pages_fetched,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'searchInformation': {
                'totalResults': total_results,
                'propertiesFetched': len(all_properties)
            },
            'properties': all_properties
        }
        
        # Save raw response
        output_file = 'data/zillow_seattle_listings.json'
        with open(output_file, 'w') as f:
            json.dump(combined_response, f, indent=2)
        
        print(f"\n✓ Saved combined response to {output_file}")
        
        # Summary statistics
        print(f"\nProperty Summary:")
        
        # Price statistics
        prices = [p.get('price') for p in all_properties if p.get('price')]
        if prices:
            print(f"  Price range: ${min(prices):,} to ${max(prices):,}")
            print(f"  Average price: ${sum(prices)/len(prices):,.0f}")
        
        # Property types
        home_types = {}
        for prop in all_properties:
            home_type = prop.get('homeType', 'UNKNOWN')
            home_types[home_type] = home_types.get(home_type, 0) + 1
        
        print(f"\n  Property types:")
        for home_type, count in sorted(home_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    {home_type}: {count}")
        
        # Days on Zillow
        days_on_zillow = [p.get('daysOnZillow') for p in all_properties if p.get('daysOnZillow') is not None]
        if days_on_zillow:
            print(f"\n  Days on Zillow:")
            print(f"    Average: {sum(days_on_zillow)/len(days_on_zillow):.1f}")
            print(f"    Range: {min(days_on_zillow)} to {max(days_on_zillow)}")
        
        # Sample property
        if all_properties:
            print(f"\n  Sample property fields:")
            sample = all_properties[0]
            for key in sorted(sample.keys())[:15]:
                value = sample[key]
                if isinstance(value, (dict, list)):
                    print(f"    {key}: {type(value).__name__}")
                else:
                    print(f"    {key}: {value}")
            if len(sample.keys()) > 15:
                print(f"    ... and {len(sample.keys()) - 15} more fields")
        
        return combined_response
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user")
        print(f"  Properties collected so far: {len(all_properties)}")
        
        if all_properties:
            # Save partial results
            partial_response = {
                'requestMetadata': {
                    'status': 'interrupted',
                    'pages_fetched': pages_fetched,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'searchInformation': {
                    'totalResults': total_results,
                    'propertiesFetched': len(all_properties)
                },
                'properties': all_properties
            }
            
            output_file = 'data/zillow_seattle_listings_partial.json'
            with open(output_file, 'w') as f:
                json.dump(partial_response, f, indent=2)
            
            print(f"  ✓ Saved partial results to {output_file}")
            return partial_response
        
        return None
        
    except Exception as e:
        print(f"\n❌ Error fetching listings: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys
    
    # Allow command line arguments for max_pages and delay
    max_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    delay_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    print(f"\nStarting fetch with max_pages={max_pages}, delay={delay_seconds}s")
    
    print(f"Press Ctrl+C to stop early and save partial results\n")
    
    response = fetch_zillow_listings(max_pages=max_pages, delay_seconds=delay_seconds)
