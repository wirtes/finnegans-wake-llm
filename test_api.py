#!/usr/bin/env python3
"""Test script for the Finnegans Wake API."""

import requests
import json

def test_api():
    """Test the translation API."""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Health: {response.json()}")
    
    # Test translation
    print("\nTesting translation...")
    test_texts = [
        "Hello, how are you today?",
        "The sun is shining brightly in the sky.",
        "I love reading books in the evening.",
        "Technology has changed our world dramatically."
    ]
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        
        response = requests.post(
            f"{base_url}/translate",
            json={"text": text}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Translated: {result['translated']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_api()