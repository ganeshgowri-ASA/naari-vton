"""
Jewelry Image Scraper - Using Pexels (free, working)
"""
import os
import requests
from pathlib import Path
import time

# Sample jewelry images from Pexels (direct URLs - no API needed)
SAMPLE_IMAGES = {
    "necklace": [
        "https://images.pexels.com/photos/1191531/pexels-photo-1191531.jpeg?w=400",
        "https://images.pexels.com/photos/10983783/pexels-photo-10983783.jpeg?w=400",
        "https://images.pexels.com/photos/8891779/pexels-photo-8891779.jpeg?w=400",
    ],
    "earring": [
        "https://images.pexels.com/photos/2735970/pexels-photo-2735970.jpeg?w=400",
        "https://images.pexels.com/photos/10151848/pexels-photo-10151848.jpeg?w=400",
        "https://images.pexels.com/photos/8891780/pexels-photo-8891780.jpeg?w=400",
    ],
    "bracelet": [
        "https://images.pexels.com/photos/1395306/pexels-photo-1395306.jpeg?w=400",
        "https://images.pexels.com/photos/2849742/pexels-photo-2849742.jpeg?w=400",
    ],
    "ring": [
        "https://images.pexels.com/photos/691046/pexels-photo-691046.jpeg?w=400",
        "https://images.pexels.com/photos/2697598/pexels-photo-2697598.jpeg?w=400",
    ],
}

def download_sample_images():
    base_dir = Path("dataset/jewelry_images")
    downloaded = 0
    
    for category, urls in SAMPLE_IMAGES.items():
        cat_dir = base_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        for i, url in enumerate(urls):
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    img_path = cat_dir / f"{category}_{i+1}.jpg"
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {img_path}")
                    downloaded += 1
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"Failed {category}_{i+1}: {e}")
    
    print(f"\n✓ Downloaded {downloaded} images to dataset/jewelry_images/")
    return downloaded

if __name__ == "__main__":
    download_sample_images()
