#!/usr/bin/env python3
"""
Example client for the Paddy Disease Classification API.

This script demonstrates how to use the API to predict disease from images.
"""

import argparse
import requests
from pathlib import Path


def predict_disease(image_path: str, api_url: str = "http://localhost:8000") -> str:
    """
    Send an image to the API and get disease prediction.
    
    Args:
        image_path: Path to the image file
        api_url: Base URL of the API (default: http://localhost:8000)
        
    Returns:
        Predicted disease name
        
    Raises:
        requests.RequestException: If API request fails
        FileNotFoundError: If image file doesn't exist
    """
    # Validate image path
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Prepare the request
    url = f"{api_url.rstrip('/')}/predict/"
    
    # Open and send the image
    with open(img_path, 'rb') as f:
        files = {'file': (img_path.name, f, 'image/jpeg')}
        response = requests.post(url, files=files)
    
    # Check for errors
    response.raise_for_status()
    
    # Parse and return prediction
    result = response.json()
    return result.get('prediction', 'Unknown')


def check_api_health(api_url: str = "http://localhost:8000") -> dict:
    """
    Check if the API is healthy and get model info.
    
    Args:
        api_url: Base URL of the API
        
    Returns:
        Dictionary with API status and model information
    """
    url = f"{api_url.rstrip('/')}/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="Predict paddy disease from an image using the API"
    )
    parser.add_argument(
        'image',
        help='Path to the image file'
    )
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='API base URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--check-health',
        action='store_true',
        help='Check API health before making prediction'
    )
    
    args = parser.parse_args()
    
    try:
        # Check health if requested
        if args.check_health:
            print("Checking API health...")
            health = check_api_health(args.url)
            print(f"API Status: {health.get('message', 'Unknown')}")
            print(f"Available classes: {health.get('classes', [])}")
            print()
        
        # Make prediction
        print(f"Predicting disease for: {args.image}")
        prediction = predict_disease(args.image, args.url)
        print(f"\n🔍 Prediction: {prediction}\n")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except requests.RequestException as e:
        print(f"API Error: {e}")
        print(f"\nMake sure the API is running at {args.url}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
