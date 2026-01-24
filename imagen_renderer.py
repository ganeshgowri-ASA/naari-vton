"""
Imagen Renderer Module

This module integrates Google Imagen API for photorealistic jewelry rendering.
Features:
- Google Imagen 3.0 integration with EDIT_MODE_INPAINT_INSERTION
- Intelligent prompt generation for each jewelry type
- Fallback to overlay method when API is unavailable
"""

import io
import os
import base64
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.warning("google-genai not available. Imagen API will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RenderMode(Enum):
    """Rendering modes available."""
    IMAGEN_INPAINT = "imagen_inpaint"
    OVERLAY = "overlay"
    HYBRID = "hybrid"


class JewelryPlacement(Enum):
    """Jewelry placement positions on body."""
    NECK = "neck"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    BOTH_EARS = "both_ears"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"


@dataclass
class RenderConfig:
    """Configuration for rendering."""
    mode: RenderMode = RenderMode.HYBRID
    imagen_model: str = "imagen-3.0-capability-001"
    blend_strength: float = 0.85
    shadow_opacity: float = 0.3
    highlight_intensity: float = 0.2
    color_match: bool = True
    use_antialiasing: bool = True


@dataclass
class RenderResult:
    """Result of rendering operation."""
    success: bool
    image: Optional[Image.Image]
    mode_used: RenderMode
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ImagenRenderer:
    """
    Advanced jewelry renderer using Google Imagen API with fallback support.
    """

    # Prompt templates for different jewelry types
    PROMPT_TEMPLATES = {
        "necklace": (
            "Photorealistic rendering of a person wearing {description}. "
            "The necklace sits naturally on the neck with realistic shadows and reflections. "
            "Professional jewelry photography lighting, high detail on metal and gemstones."
        ),
        "earring": (
            "Photorealistic rendering of a person wearing {description} as earrings. "
            "The earrings hang naturally from the ears with realistic light reflections. "
            "Professional portrait lighting emphasizing the jewelry details."
        ),
        "bangle": (
            "Photorealistic rendering of a person wearing {description} as a bangle on the wrist. "
            "The bangle fits naturally around the wrist with realistic metal reflections. "
            "Studio lighting highlighting the jewelry's craftsmanship."
        ),
        "bracelet": (
            "Photorealistic rendering of a person wearing {description} as a bracelet. "
            "The bracelet drapes naturally on the wrist with realistic shadows. "
            "Professional jewelry photography with attention to material details."
        ),
        "ring": (
            "Photorealistic rendering of a person wearing {description} as a ring. "
            "The ring fits naturally on the finger with realistic gemstone sparkle. "
            "Close-up professional lighting emphasizing the ring's details."
        ),
        "pendant": (
            "Photorealistic rendering of a person wearing {description} as a pendant necklace. "
            "The pendant hangs naturally at the chest with realistic chain and gemstone details. "
            "Professional portrait lighting with jewelry emphasis."
        ),
        "default": (
            "Photorealistic rendering of a person wearing {description}. "
            "Natural fit with realistic shadows, reflections, and material properties. "
            "Professional jewelry photography lighting."
        )
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[RenderConfig] = None
    ):
        """
        Initialize the Imagen renderer.

        Args:
            api_key: Google API key (or uses GOOGLE_API_KEY env var)
            config: Render configuration
        """
        self.config = config or RenderConfig()
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = None

        if GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Google Imagen API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Imagen client: {e}")
                self.client = None
        else:
            if not GENAI_AVAILABLE:
                logger.warning("google-genai package not installed")
            if not self.api_key:
                logger.warning("No Google API key provided")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _base64_to_image(self, base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))

    def generate_prompt(
        self,
        jewelry_type: str,
        description: str,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Generate an optimized prompt for Imagen based on jewelry type.

        Args:
            jewelry_type: Type of jewelry (necklace, earring, etc.)
            description: Natural language description of the jewelry
            additional_context: Optional additional context for the prompt

        Returns:
            Optimized prompt string
        """
        template = self.PROMPT_TEMPLATES.get(
            jewelry_type.lower(),
            self.PROMPT_TEMPLATES["default"]
        )

        prompt = template.format(description=description)

        if additional_context:
            prompt += f" {additional_context}"

        # Add quality modifiers
        prompt += (
            " Ultra high resolution, 8K quality, photorealistic, "
            "professional studio lighting, sharp focus on jewelry details."
        )

        return prompt

    def create_inpaint_mask(
        self,
        image_size: Tuple[int, int],
        region: Dict[str, Any],
        feather_radius: int = 10
    ) -> Image.Image:
        """
        Create an inpainting mask for the target region.

        Args:
            image_size: Size of the image (width, height)
            region: Dictionary with region coordinates and shape
            feather_radius: Radius for feathering the mask edges

        Returns:
            Mask image (white = area to inpaint)
        """
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)

        region_type = region.get('type', 'rectangle')
        coords = region.get('coords', [])

        if region_type == 'rectangle' and len(coords) >= 4:
            x1, y1, x2, y2 = coords[:4]
            draw.rectangle([x1, y1, x2, y2], fill=255)

        elif region_type == 'ellipse' and len(coords) >= 4:
            x1, y1, x2, y2 = coords[:4]
            draw.ellipse([x1, y1, x2, y2], fill=255)

        elif region_type == 'polygon' and len(coords) >= 6:
            # Coords should be flat list: [x1, y1, x2, y2, x3, y3, ...]
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            draw.polygon(points, fill=255)

        elif region_type == 'points' and len(coords) >= 4:
            # Draw convex hull around points
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            if len(points) >= 3:
                draw.polygon(points, fill=255)

        # Apply feathering
        if feather_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

        return mask

    async def render_with_imagen(
        self,
        person_image: Image.Image,
        mask: Image.Image,
        prompt: str
    ) -> RenderResult:
        """
        Render jewelry using Google Imagen API with inpainting.

        Args:
            person_image: Base image of the person
            mask: Inpainting mask (white = area to edit)
            prompt: Prompt describing the desired result

        Returns:
            RenderResult with the generated image
        """
        if not self.client:
            return RenderResult(
                success=False,
                image=None,
                mode_used=RenderMode.IMAGEN_INPAINT,
                error_message="Imagen client not available"
            )

        try:
            # Convert images to bytes
            person_bytes = io.BytesIO()
            person_image.save(person_bytes, format="PNG")
            person_bytes.seek(0)

            mask_bytes = io.BytesIO()
            mask.save(mask_bytes, format="PNG")
            mask_bytes.seek(0)

            # Create the edit request using Imagen 3.0
            response = await self.client.aio.models.edit_image(
                model=self.config.imagen_model,
                prompt=prompt,
                image=types.Image(image_bytes=person_bytes.getvalue()),
                mask=types.Image(image_bytes=mask_bytes.getvalue()),
                config=types.EditImageConfig(
                    edit_mode="EDIT_MODE_INPAINT_INSERTION",
                    number_of_images=1
                )
            )

            # Extract the generated image
            if response.generated_images and len(response.generated_images) > 0:
                generated_image = response.generated_images[0].image
                result_image = Image.open(io.BytesIO(generated_image.image_bytes))

                return RenderResult(
                    success=True,
                    image=result_image,
                    mode_used=RenderMode.IMAGEN_INPAINT,
                    metadata={"prompt": prompt}
                )
            else:
                return RenderResult(
                    success=False,
                    image=None,
                    mode_used=RenderMode.IMAGEN_INPAINT,
                    error_message="No images generated"
                )

        except Exception as e:
            logger.error(f"Imagen API error: {e}")
            return RenderResult(
                success=False,
                image=None,
                mode_used=RenderMode.IMAGEN_INPAINT,
                error_message=str(e)
            )

    def render_with_imagen_sync(
        self,
        person_image: Image.Image,
        mask: Image.Image,
        prompt: str
    ) -> RenderResult:
        """
        Synchronous version of render_with_imagen.

        Args:
            person_image: Base image of the person
            mask: Inpainting mask
            prompt: Prompt describing the desired result

        Returns:
            RenderResult with the generated image
        """
        if not self.client:
            return RenderResult(
                success=False,
                image=None,
                mode_used=RenderMode.IMAGEN_INPAINT,
                error_message="Imagen client not available"
            )

        try:
            # Convert images to bytes
            person_bytes = io.BytesIO()
            person_image.save(person_bytes, format="PNG")
            person_bytes.seek(0)

            mask_bytes = io.BytesIO()
            mask.save(mask_bytes, format="PNG")
            mask_bytes.seek(0)

            # Create the edit request using Imagen 3.0
            response = self.client.models.edit_image(
                model=self.config.imagen_model,
                prompt=prompt,
                image=types.Image(image_bytes=person_bytes.getvalue()),
                mask=types.Image(image_bytes=mask_bytes.getvalue()),
                config=types.EditImageConfig(
                    edit_mode="EDIT_MODE_INPAINT_INSERTION",
                    number_of_images=1
                )
            )

            # Extract the generated image
            if response.generated_images and len(response.generated_images) > 0:
                generated_image = response.generated_images[0].image
                result_image = Image.open(io.BytesIO(generated_image.image_bytes))

                return RenderResult(
                    success=True,
                    image=result_image,
                    mode_used=RenderMode.IMAGEN_INPAINT,
                    metadata={"prompt": prompt}
                )
            else:
                return RenderResult(
                    success=False,
                    image=None,
                    mode_used=RenderMode.IMAGEN_INPAINT,
                    error_message="No images generated"
                )

        except Exception as e:
            logger.error(f"Imagen API error: {e}")
            return RenderResult(
                success=False,
                image=None,
                mode_used=RenderMode.IMAGEN_INPAINT,
                error_message=str(e)
            )

    def render_with_overlay(
        self,
        person_image: Image.Image,
        jewelry_image: Image.Image,
        position: Tuple[int, int],
        scale: float = 1.0,
        rotation: float = 0.0,
        blend_mode: str = "normal"
    ) -> RenderResult:
        """
        Fallback rendering using image overlay with advanced blending.

        Args:
            person_image: Base image of the person
            jewelry_image: Processed jewelry image (with transparency)
            position: (x, y) position for jewelry placement
            scale: Scale factor for the jewelry
            rotation: Rotation angle in degrees
            blend_mode: Blending mode ('normal', 'multiply', 'screen', 'overlay')

        Returns:
            RenderResult with the composited image
        """
        try:
            # Ensure images are in correct mode
            if person_image.mode != 'RGBA':
                person_image = person_image.convert('RGBA')
            if jewelry_image.mode != 'RGBA':
                jewelry_image = jewelry_image.convert('RGBA')

            # Create a copy of the person image
            result = person_image.copy()

            # Scale jewelry if needed
            if scale != 1.0:
                new_size = (
                    int(jewelry_image.width * scale),
                    int(jewelry_image.height * scale)
                )
                jewelry_image = jewelry_image.resize(new_size, Image.Resampling.LANCZOS)

            # Rotate jewelry if needed
            if rotation != 0.0:
                jewelry_image = jewelry_image.rotate(
                    rotation,
                    resample=Image.Resampling.BICUBIC,
                    expand=True
                )

            # Apply shadow effect
            if self.config.shadow_opacity > 0:
                jewelry_with_shadow = self._add_shadow(
                    jewelry_image,
                    opacity=self.config.shadow_opacity
                )
            else:
                jewelry_with_shadow = jewelry_image

            # Apply highlight effect for metallic look
            if self.config.highlight_intensity > 0:
                jewelry_with_effects = self._add_highlights(
                    jewelry_with_shadow,
                    intensity=self.config.highlight_intensity
                )
            else:
                jewelry_with_effects = jewelry_with_shadow

            # Color match if enabled
            if self.config.color_match:
                jewelry_with_effects = self._color_match(
                    jewelry_with_effects,
                    person_image,
                    position
                )

            # Composite based on blend mode
            if blend_mode == "normal":
                result = self._composite_normal(result, jewelry_with_effects, position)
            elif blend_mode == "multiply":
                result = self._composite_multiply(result, jewelry_with_effects, position)
            elif blend_mode == "screen":
                result = self._composite_screen(result, jewelry_with_effects, position)
            elif blend_mode == "overlay":
                result = self._composite_overlay(result, jewelry_with_effects, position)
            else:
                result = self._composite_normal(result, jewelry_with_effects, position)

            # Apply antialiasing if enabled
            if self.config.use_antialiasing:
                # Subtle sharpening for cleaner edges
                result = result.filter(ImageFilter.SMOOTH_MORE)

            return RenderResult(
                success=True,
                image=result,
                mode_used=RenderMode.OVERLAY,
                metadata={
                    "position": position,
                    "scale": scale,
                    "rotation": rotation,
                    "blend_mode": blend_mode
                }
            )

        except Exception as e:
            logger.error(f"Overlay rendering error: {e}")
            return RenderResult(
                success=False,
                image=None,
                mode_used=RenderMode.OVERLAY,
                error_message=str(e)
            )

    def _add_shadow(
        self,
        image: Image.Image,
        opacity: float = 0.3,
        offset: Tuple[int, int] = (5, 5),
        blur_radius: int = 8
    ) -> Image.Image:
        """Add a drop shadow to the jewelry image."""
        # Create shadow from alpha channel
        alpha = image.split()[3]
        shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))

        # Create shadow layer
        shadow_layer = Image.new('RGBA', image.size, (0, 0, 0, int(255 * opacity)))
        shadow_layer.putalpha(alpha)

        # Blur the shadow
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(blur_radius))

        # Create result with shadow
        result_size = (
            image.width + abs(offset[0]),
            image.height + abs(offset[1])
        )
        result = Image.new('RGBA', result_size, (0, 0, 0, 0))

        # Paste shadow with offset
        shadow_pos = (max(0, offset[0]), max(0, offset[1]))
        result.paste(shadow_layer, shadow_pos, shadow_layer)

        # Paste original image
        image_pos = (max(0, -offset[0]), max(0, -offset[1]))
        result.paste(image, image_pos, image)

        return result

    def _add_highlights(
        self,
        image: Image.Image,
        intensity: float = 0.2
    ) -> Image.Image:
        """Add subtle highlights for metallic effect."""
        enhancer = ImageEnhance.Brightness(image)
        brightened = enhancer.enhance(1.0 + intensity * 0.5)

        enhancer = ImageEnhance.Contrast(brightened)
        result = enhancer.enhance(1.0 + intensity * 0.3)

        return result

    def _color_match(
        self,
        jewelry: Image.Image,
        background: Image.Image,
        position: Tuple[int, int]
    ) -> Image.Image:
        """
        Adjust jewelry colors to match the lighting of the background.
        """
        # Sample background colors around the placement position
        bg_array = np.array(background)

        # Define sampling region
        x, y = position
        sample_size = 50
        x1 = max(0, x - sample_size)
        y1 = max(0, y - sample_size)
        x2 = min(background.width, x + sample_size)
        y2 = min(background.height, y + sample_size)

        if x2 > x1 and y2 > y1:
            sample_region = bg_array[y1:y2, x1:x2]
            avg_brightness = np.mean(sample_region) / 255.0

            # Adjust jewelry brightness to match
            if avg_brightness < 0.5:
                # Dark background - slightly reduce jewelry brightness
                enhancer = ImageEnhance.Brightness(jewelry)
                jewelry = enhancer.enhance(0.9)
            elif avg_brightness > 0.7:
                # Bright background - slightly increase jewelry brightness
                enhancer = ImageEnhance.Brightness(jewelry)
                jewelry = enhancer.enhance(1.1)

        return jewelry

    def _composite_normal(
        self,
        background: Image.Image,
        foreground: Image.Image,
        position: Tuple[int, int]
    ) -> Image.Image:
        """Normal alpha composite."""
        result = background.copy()
        result.paste(foreground, position, foreground)
        return result

    def _composite_multiply(
        self,
        background: Image.Image,
        foreground: Image.Image,
        position: Tuple[int, int]
    ) -> Image.Image:
        """Multiply blend mode."""
        result = background.copy()

        # Extract region
        fg_array = np.array(foreground).astype(float)
        x, y = position
        w, h = foreground.size

        # Ensure bounds
        x2 = min(x + w, background.width)
        y2 = min(y + h, background.height)
        fg_w = x2 - x
        fg_h = y2 - y

        if fg_w > 0 and fg_h > 0:
            bg_array = np.array(result).astype(float)
            bg_region = bg_array[y:y2, x:x2]

            # Multiply blend
            alpha = fg_array[:fg_h, :fg_w, 3:4] / 255.0
            fg_rgb = fg_array[:fg_h, :fg_w, :3]
            bg_rgb = bg_region[:, :, :3]

            blended = (fg_rgb * bg_rgb / 255.0)
            result_rgb = blended * alpha + bg_rgb * (1 - alpha)

            bg_array[y:y2, x:x2, :3] = result_rgb
            result = Image.fromarray(bg_array.astype(np.uint8))

        return result

    def _composite_screen(
        self,
        background: Image.Image,
        foreground: Image.Image,
        position: Tuple[int, int]
    ) -> Image.Image:
        """Screen blend mode."""
        result = background.copy()

        fg_array = np.array(foreground).astype(float)
        x, y = position
        w, h = foreground.size

        x2 = min(x + w, background.width)
        y2 = min(y + h, background.height)
        fg_w = x2 - x
        fg_h = y2 - y

        if fg_w > 0 and fg_h > 0:
            bg_array = np.array(result).astype(float)
            bg_region = bg_array[y:y2, x:x2]

            alpha = fg_array[:fg_h, :fg_w, 3:4] / 255.0
            fg_rgb = fg_array[:fg_h, :fg_w, :3]
            bg_rgb = bg_region[:, :, :3]

            # Screen blend: 1 - (1-a)(1-b)
            blended = 255 - ((255 - fg_rgb) * (255 - bg_rgb) / 255.0)
            result_rgb = blended * alpha + bg_rgb * (1 - alpha)

            bg_array[y:y2, x:x2, :3] = result_rgb
            result = Image.fromarray(bg_array.astype(np.uint8))

        return result

    def _composite_overlay(
        self,
        background: Image.Image,
        foreground: Image.Image,
        position: Tuple[int, int]
    ) -> Image.Image:
        """Overlay blend mode."""
        result = background.copy()

        fg_array = np.array(foreground).astype(float)
        x, y = position
        w, h = foreground.size

        x2 = min(x + w, background.width)
        y2 = min(y + h, background.height)
        fg_w = x2 - x
        fg_h = y2 - y

        if fg_w > 0 and fg_h > 0:
            bg_array = np.array(result).astype(float)
            bg_region = bg_array[y:y2, x:x2]

            alpha = fg_array[:fg_h, :fg_w, 3:4] / 255.0
            fg_rgb = fg_array[:fg_h, :fg_w, :3] / 255.0
            bg_rgb = bg_region[:, :, :3] / 255.0

            # Overlay blend
            mask = bg_rgb < 0.5
            blended = np.where(
                mask,
                2 * fg_rgb * bg_rgb,
                1 - 2 * (1 - fg_rgb) * (1 - bg_rgb)
            ) * 255

            result_rgb = blended * alpha + bg_region[:, :, :3] * (1 - alpha)

            bg_array[y:y2, x:x2, :3] = result_rgb
            result = Image.fromarray(bg_array.astype(np.uint8))

        return result

    def render(
        self,
        person_image: Image.Image,
        jewelry_image: Image.Image,
        jewelry_type: str,
        jewelry_description: str,
        placement_region: Dict[str, Any],
        position: Optional[Tuple[int, int]] = None,
        scale: float = 1.0,
        rotation: float = 0.0
    ) -> RenderResult:
        """
        Main render method with automatic fallback.

        Args:
            person_image: Base image of the person
            jewelry_image: Processed jewelry image
            jewelry_type: Type of jewelry
            jewelry_description: Description for prompt generation
            placement_region: Region dict for inpainting mask
            position: Position for overlay fallback
            scale: Scale factor
            rotation: Rotation angle

        Returns:
            RenderResult with the rendered image
        """
        # Generate prompt
        prompt = self.generate_prompt(jewelry_type, jewelry_description)

        # Try Imagen first if configured for hybrid or inpaint mode
        if self.config.mode in [RenderMode.IMAGEN_INPAINT, RenderMode.HYBRID]:
            if self.client:
                # Create inpaint mask
                mask = self.create_inpaint_mask(
                    person_image.size,
                    placement_region
                )

                # Try Imagen rendering
                result = self.render_with_imagen_sync(person_image, mask, prompt)

                if result.success:
                    logger.info("Successfully rendered with Imagen API")
                    return result
                else:
                    logger.warning(f"Imagen failed: {result.error_message}")

        # Fallback to overlay
        if self.config.mode in [RenderMode.OVERLAY, RenderMode.HYBRID]:
            if position is None:
                # Calculate position from region
                coords = placement_region.get('coords', [])
                if len(coords) >= 4:
                    x = coords[0]
                    y = coords[1]
                    position = (x, y)
                else:
                    position = (person_image.width // 4, person_image.height // 4)

            result = self.render_with_overlay(
                person_image,
                jewelry_image,
                position,
                scale,
                rotation
            )

            if result.success:
                logger.info("Successfully rendered with overlay fallback")
                return result

        return RenderResult(
            success=False,
            image=None,
            mode_used=self.config.mode,
            error_message="All rendering methods failed"
        )


def create_renderer(
    api_key: Optional[str] = None,
    mode: str = "hybrid"
) -> ImagenRenderer:
    """
    Factory function to create an ImagenRenderer.

    Args:
        api_key: Google API key
        mode: Render mode ('imagen_inpaint', 'overlay', 'hybrid')

    Returns:
        Configured ImagenRenderer instance
    """
    render_mode = {
        'imagen_inpaint': RenderMode.IMAGEN_INPAINT,
        'overlay': RenderMode.OVERLAY,
        'hybrid': RenderMode.HYBRID
    }.get(mode.lower(), RenderMode.HYBRID)

    config = RenderConfig(mode=render_mode)
    return ImagenRenderer(api_key=api_key, config=config)


if __name__ == "__main__":
    # Example usage
    print("Imagen Renderer Module")
    print("=" * 50)
    print(f"Google GenAI available: {GENAI_AVAILABLE}")

    renderer = create_renderer(mode="hybrid")
    print(f"Renderer created with mode: {renderer.config.mode.value}")

    # Test prompt generation
    test_prompt = renderer.generate_prompt(
        "necklace",
        "elegant gold necklace with diamond pendant"
    )
    print(f"\nSample prompt:\n{test_prompt}")
