"""
Jewelry Image Scraper for Training Dataset
"""
import os
import requests
from pathlib import Path

UNSPLASH_SEARCH = "https://source.unsplash.com/400x400/?{query}"

def download_sample_images():
    base_dir = Path("dataset/jewelry_images")
    categories = ["necklace", "earring", "bracelet", "ring", "maangtika", "nosering"]
    
    for cat in categories:
        cat_dir = base_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(5):
            url = UNSPLASH_SEARCH.format(query=f"{cat}+jewelry+indian")
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img_path = cat_dir / f"{cat}_{i+1}.jpg"
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {img_path}")
            except Exception as e:
                print(f"Failed {cat}_{i+1}: {e}")
    
    print("\nSample dataset created in dataset/jewelry_images/")

if __name__ == "__main__":
    download_sample_images()
