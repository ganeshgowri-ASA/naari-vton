"""
Generate sample jewelry assets for testing.
Creates simple placeholder jewelry images with transparency.
"""

from PIL import Image, ImageDraw, ImageFont
import os


def create_necklace(output_path: str, width: int = 400, height: int = 200):
    """Create a sample necklace image with transparency."""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw a curved chain
    center_x = width // 2
    chain_color = (218, 165, 32, 255)  # Gold color

    # Draw the chain as connected circles (pearls/beads style)
    for i in range(-15, 16):
        x = center_x + i * 12
        # Parabolic curve for necklace drape
        y = 30 + int((i ** 2) * 0.3)
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=chain_color, outline=(139, 69, 19, 255))

    # Draw a pendant in the center
    pendant_x = center_x
    pendant_y = 100
    # Main pendant
    draw.ellipse([pendant_x - 25, pendant_y - 15, pendant_x + 25, pendant_y + 40],
                 fill=(255, 215, 0, 255), outline=(139, 69, 19, 255), width=3)
    # Inner design
    draw.ellipse([pendant_x - 15, pendant_y, pendant_x + 15, pendant_y + 25],
                 fill=(255, 0, 0, 200), outline=(139, 0, 0, 255), width=2)

    img.save(output_path)
    print(f"Created: {output_path}")


def create_earring(output_path: str, width: int = 80, height: int = 150):
    """Create a sample earring image with transparency."""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    center_x = width // 2
    gold = (218, 165, 32, 255)
    dark_gold = (139, 69, 19, 255)

    # Stud at top
    draw.ellipse([center_x - 10, 5, center_x + 10, 25], fill=gold, outline=dark_gold, width=2)

    # Chain/connector
    for i in range(4):
        y = 30 + i * 15
        draw.ellipse([center_x - 5, y, center_x + 5, y + 10], fill=gold, outline=dark_gold)

    # Main drop
    draw.ellipse([center_x - 20, 85, center_x + 20, 140], fill=(0, 128, 0, 255),
                 outline=dark_gold, width=3)
    draw.ellipse([center_x - 12, 95, center_x + 12, 130], fill=(50, 205, 50, 200))

    img.save(output_path)
    print(f"Created: {output_path}")


def create_maang_tikka(output_path: str, width: int = 120, height: int = 180):
    """Create a sample maang tikka image with transparency."""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    center_x = width // 2
    gold = (218, 165, 32, 255)
    dark_gold = (139, 69, 19, 255)
    red = (220, 20, 60, 255)

    # Top chain attachment
    draw.ellipse([center_x - 8, 5, center_x + 8, 21], fill=gold, outline=dark_gold)

    # Vertical chain
    for i in range(5):
        y = 25 + i * 18
        draw.ellipse([center_x - 6, y, center_x + 6, y + 12], fill=gold, outline=dark_gold)

    # Main pendant piece
    # Outer circle
    draw.ellipse([center_x - 35, 110, center_x + 35, 175], fill=gold, outline=dark_gold, width=3)
    # Middle circle
    draw.ellipse([center_x - 25, 120, center_x + 25, 165], fill=red, outline=dark_gold, width=2)
    # Inner circle (kundan style)
    draw.ellipse([center_x - 12, 133, center_x + 12, 155], fill=(255, 255, 255, 230),
                 outline=(200, 200, 200, 255), width=1)

    img.save(output_path)
    print(f"Created: {output_path}")


def create_bangle(output_path: str, width: int = 150, height: int = 150):
    """Create a sample bangle image with transparency."""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    center_x = width // 2
    center_y = height // 2
    gold = (218, 165, 32, 255)
    dark_gold = (139, 69, 19, 255)

    # Outer ring
    draw.ellipse([10, 10, width - 10, height - 10], fill=None, outline=gold, width=15)

    # Decorative elements around the bangle
    for angle in range(0, 360, 30):
        import math
        rad = math.radians(angle)
        r = 55
        x = center_x + int(r * math.cos(rad))
        y = center_y + int(r * math.sin(rad))
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(255, 0, 0, 200), outline=dark_gold)

    # Inner ring edge
    draw.ellipse([30, 30, width - 30, height - 30], fill=None, outline=dark_gold, width=3)

    img.save(output_path)
    print(f"Created: {output_path}")


def main():
    """Generate all sample jewelry assets."""
    assets_dir = "assets/jewelry"
    os.makedirs(assets_dir, exist_ok=True)

    print("Generating sample jewelry assets...")
    print("-" * 40)

    create_necklace(os.path.join(assets_dir, "necklace.png"))
    create_earring(os.path.join(assets_dir, "earring.png"))
    create_maang_tikka(os.path.join(assets_dir, "maang_tikka.png"))
    create_bangle(os.path.join(assets_dir, "bangle.png"))

    print("-" * 40)
    print("Done! Sample assets created in:", assets_dir)


if __name__ == "__main__":
    main()
