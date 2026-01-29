"""
Naari Studio - Virtual Try-On Application
HuggingFace Spaces compatible Gradio interface.

Features:
- Garment Virtual Try-On (placeholder for IDM-VTON integration)
- Jewelry Virtual Try-On (Necklace, Earrings, Maang Tikka, Nose Ring, Bangles, Rings)
- AI-Powered Jewelry Generation via Replicate trained model

Powered by:
- cvzone PoseModule and MediaPipe Face Mesh for accurate landmark detection
- Replicate trained model (ganeshgowri-asa/naari-jewelry-vton:f6b844b4) for AI generation
"""

import gradio as gr
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace Spaces GPU decorator
try:
    import spaces
    SPACES_AVAILABLE = True
    logger.info("HuggingFace Spaces module loaded")
except ImportError:
    SPACES_AVAILABLE = False
    logger.warning("HuggingFace Spaces module not available - running without @spaces.GPU decorator")
    # Create a dummy decorator that does nothing
    class spaces:
        @staticmethod
        def GPU(duration=60):
            def decorator(func):
                return func
            return decorator

from jewelry_engine import (
    apply_jewelry,
    remove_jewelry_background,
    jewelry_tryon_api,
    get_available_options,
    get_generation_engine,
    JEWELRY_TYPES,
    METAL_TYPES,
    STONE_TYPES,
    STYLE_OPTIONS,
    REPLICATE_AVAILABLE
)


# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

MAX_IMAGE_DIMENSION = 4096


def resize_image_if_needed(image: Optional[np.ndarray], max_dim: int = MAX_IMAGE_DIMENSION) -> Optional[np.ndarray]:
    """
    Resize image if any dimension exceeds max_dim, maintaining aspect ratio.

    Args:
        image: Input image as numpy array (RGB/RGBA)
        max_dim: Maximum allowed dimension (default 4096)

    Returns:
        Resized image as numpy array, or original if no resize needed
    """
    if image is None:
        return None

    height, width = image.shape[:2]

    # Check if resize is needed
    if width <= max_dim and height <= max_dim:
        return image

    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = max_dim
        new_height = int(height * (max_dim / width))
    else:
        new_height = max_dim
        new_width = int(width * (max_dim / height))

    logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height} (max dimension: {max_dim})")

    # Convert to PIL, resize with LANCZOS, convert back
    pil_image = Image.fromarray(image)
    resized_pil = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return np.array(resized_pil)


# Theme configuration
THEME = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="pink",
    neutral_hue="slate",
)

# CSS for better styling
CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}
.tab-nav button {
    font-size: 16px !important;
    font-weight: 600 !important;
}
.result-image {
    min-height: 400px;
}
.main-tabs > .tab-nav {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px 10px 0 0;
    padding: 5px;
}
.main-tabs > .tab-nav button {
    color: white !important;
    font-size: 18px !important;
}
.main-tabs > .tab-nav button.selected {
    background: rgba(255,255,255,0.2) !important;
    border-radius: 5px;
}
.jewelry-section {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
footer {
    visibility: hidden;
}
"""


# ============================================================================
# JEWELRY TRY-ON FUNCTIONS
# ============================================================================

@spaces.GPU(duration=60)
def process_necklace(person_image: Optional[np.ndarray],
                     jewelry_image: Optional[np.ndarray],
                     opacity: float) -> Tuple[Optional[np.ndarray], str]:
    """Process necklace try-on request with GPU acceleration."""
    if person_image is None:
        return None, "Please upload a person photo."
    if jewelry_image is None:
        return None, "Please upload a necklace image."

    # Resize images if needed to prevent "image too large" errors
    person_image = resize_image_if_needed(person_image)
    jewelry_image = resize_image_if_needed(jewelry_image)

    # Convert numpy to PIL
    person_pil = Image.fromarray(person_image)
    jewelry_pil = Image.fromarray(jewelry_image)

    # Apply jewelry
    result, message = apply_jewelry(person_pil, jewelry_pil, "necklace", opacity)

    if result is not None:
        return np.array(result.convert('RGB')), message
    return None, message


@spaces.GPU(duration=60)
def process_earrings(person_image: Optional[np.ndarray],
                     jewelry_image: Optional[np.ndarray],
                     opacity: float) -> Tuple[Optional[np.ndarray], str]:
    """Process earrings try-on request with GPU acceleration."""
    if person_image is None:
        return None, "Please upload a person photo."
    if jewelry_image is None:
        return None, "Please upload an earring image."

    # Resize images if needed to prevent "image too large" errors
    person_image = resize_image_if_needed(person_image)
    jewelry_image = resize_image_if_needed(jewelry_image)

    person_pil = Image.fromarray(person_image)
    jewelry_pil = Image.fromarray(jewelry_image)

    result, message = apply_jewelry(person_pil, jewelry_pil, "earrings", opacity)

    if result is not None:
        return np.array(result.convert('RGB')), message
    return None, message


@spaces.GPU(duration=60)
def process_maang_tikka(person_image: Optional[np.ndarray],
                        jewelry_image: Optional[np.ndarray],
                        opacity: float) -> Tuple[Optional[np.ndarray], str]:
    """Process maang tikka try-on request with GPU acceleration."""
    if person_image is None:
        return None, "Please upload a person photo."
    if jewelry_image is None:
        return None, "Please upload a maang tikka image."

    # Resize images if needed to prevent "image too large" errors
    person_image = resize_image_if_needed(person_image)
    jewelry_image = resize_image_if_needed(jewelry_image)

    person_pil = Image.fromarray(person_image)
    jewelry_pil = Image.fromarray(jewelry_image)

    result, message = apply_jewelry(person_pil, jewelry_pil, "maang_tikka", opacity)

    if result is not None:
        return np.array(result.convert('RGB')), message
    return None, message


@spaces.GPU(duration=60)
def process_nose_ring(person_image: Optional[np.ndarray],
                      jewelry_image: Optional[np.ndarray],
                      opacity: float,
                      side: str,
                      style: str) -> Tuple[Optional[np.ndarray], str]:
    """Process nose ring try-on request with GPU acceleration."""
    if person_image is None:
        return None, "Please upload a person photo."
    if jewelry_image is None:
        return None, "Please upload a nose ring image."

    # Resize images if needed to prevent "image too large" errors
    person_image = resize_image_if_needed(person_image)
    jewelry_image = resize_image_if_needed(jewelry_image)

    person_pil = Image.fromarray(person_image)
    jewelry_pil = Image.fromarray(jewelry_image)

    result, message = apply_jewelry(person_pil, jewelry_pil, "nose_ring", opacity,
                                    side=side, ring_style=style)

    if result is not None:
        return np.array(result.convert('RGB')), message
    return None, message


@spaces.GPU(duration=30)
def remove_background(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Remove background from jewelry image using rembg with GPU acceleration."""
    if image is None:
        return None

    pil_image = Image.fromarray(image)
    result = remove_jewelry_background(pil_image)

    return np.array(result.convert('RGBA'))


# ============================================================================
# AI-POWERED JEWELRY TRY-ON (Replicate Model)
# ============================================================================

@spaces.GPU(duration=120)
def process_ai_jewelry_tryon(
    person_image: Optional[np.ndarray],
    jewelry_prompt: str,
    jewelry_type: str,
    metal_type: str,
    stones: str,
    style: str,
    opacity: float
) -> Tuple[Optional[np.ndarray], str]:
    """
    Process AI-powered jewelry try-on using the trained Replicate model.

    Args:
        person_image: Person photo as numpy array
        jewelry_prompt: Text prompt for jewelry generation
        jewelry_type: Type of jewelry (necklace, earrings, etc.)
        metal_type: Metal type (gold, silver, etc.)
        stones: Stone type (diamond, ruby, etc.)
        style: Style variant
        opacity: Overlay opacity

    Returns:
        Tuple of (result image, status message)
    """
    if person_image is None:
        return None, "Please upload a person photo."

    if not jewelry_prompt or jewelry_prompt.strip() == "":
        jewelry_prompt = "beautiful jewelry"

    # Resize image if needed
    person_image = resize_image_if_needed(person_image)

    try:
        # Call the jewelry try-on API
        result = jewelry_tryon_api(
            person_image=person_image,
            jewelry_prompt=jewelry_prompt,
            jewelry_type=jewelry_type,
            metal_type=metal_type,
            stones=stones,
            style=style if style else None,
            opacity=opacity
        )

        if result["success"] and result["image"] is not None:
            result_array = np.array(result["image"].convert('RGB'))
            return result_array, result["message"]
        else:
            return None, result["message"]

    except Exception as e:
        logger.error(f"AI jewelry try-on error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"


def get_styles_for_jewelry_type(jewelry_type: str) -> list:
    """Get available styles for a given jewelry type."""
    jewelry_type = jewelry_type.lower().replace(" ", "_").replace("-", "_")
    return STYLE_OPTIONS.get(jewelry_type, ["default"])


def update_style_dropdown(jewelry_type: str):
    """Update style dropdown choices based on jewelry type selection."""
    styles = get_styles_for_jewelry_type(jewelry_type)
    return gr.Dropdown(choices=styles, value=styles[0] if styles else "default")


# ============================================================================
# API ENDPOINT FUNCTION (for programmatic access)
# ============================================================================

def api_jewelry_tryon(
    person_image_path: str,
    jewelry_prompt: str,
    jewelry_type: str = "necklace",
    metal_type: str = "gold",
    stones: str = "none",
    style: str = None
) -> Dict[str, Any]:
    """
    API endpoint for jewelry try-on.

    This function provides a programmatic interface for the /api/jewelry-tryon endpoint.
    It can be called via Gradio's API mode.

    Args:
        person_image_path: Path to person image file
        jewelry_prompt: Text description of desired jewelry
        jewelry_type: Type of jewelry (necklace, earrings, bangles, rings, maang_tikka, nose_ring)
        metal_type: Metal type (gold, silver, rose gold, platinum, oxidized silver, antique gold)
        stones: Stone type (diamond, ruby, emerald, sapphire, pearl, kundan, polki, none)
        style: Style variant (depends on jewelry type)

    Returns:
        Dictionary with success status, result image path, and message
    """
    try:
        result = jewelry_tryon_api(
            person_image=person_image_path,
            jewelry_prompt=jewelry_prompt,
            jewelry_type=jewelry_type,
            metal_type=metal_type,
            stones=stones,
            style=style
        )

        # Convert PIL image to numpy for Gradio
        if result["success"] and result["image"]:
            return {
                "success": True,
                "image": np.array(result["image"].convert('RGB')),
                "message": result["message"],
                "prompt_used": result["prompt_used"]
            }
        else:
            return {
                "success": False,
                "image": None,
                "message": result["message"],
                "prompt_used": result.get("prompt_used", "")
            }

    except Exception as e:
        logger.error(f"API endpoint error: {e}")
        return {
            "success": False,
            "image": None,
            "message": f"Error: {str(e)}",
            "prompt_used": ""
        }


# ============================================================================
# GARMENT TRY-ON FUNCTIONS (Placeholder for IDM-VTON integration)
# ============================================================================

@spaces.GPU(duration=120)
def process_garment_tryon(person_image: Optional[np.ndarray],
                          garment_image: Optional[np.ndarray],
                          garment_type: str,
                          denoise_steps: int,
                          seed: int) -> Tuple[Optional[np.ndarray], str]:
    """
    Process garment virtual try-on request.

    This is a placeholder for IDM-VTON integration.
    In production, this would call the IDM-VTON model for garment try-on.
    """
    if person_image is None:
        return None, "Please upload a person photo."
    if garment_image is None:
        return None, "Please upload a garment image."

    # Resize images if needed to prevent "image too large" errors
    person_image = resize_image_if_needed(person_image)
    garment_image = resize_image_if_needed(garment_image)

    # Placeholder response - replace with actual IDM-VTON integration
    return None, "Garment try-on is coming soon! This feature requires IDM-VTON model integration."


# ============================================================================
# UI COMPONENTS
# ============================================================================

def create_jewelry_tab(jewelry_type: str,
                       jewelry_label: str,
                       description: str,
                       process_fn,
                       show_side: bool = False,
                       show_style: bool = False):
    """Create a jewelry try-on tab with consistent layout."""

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(f"### {description}")

            person_input = gr.Image(
                label="Person Photo",
                type="numpy",
                sources=["upload", "webcam"],
                height=300
            )

            jewelry_input = gr.Image(
                label=f"{jewelry_label} Image",
                type="numpy",
                sources=["upload"],
                height=300
            )

            with gr.Row():
                remove_bg_btn = gr.Button("Remove Background", variant="secondary", size="sm")

            opacity_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=1.0,
                step=0.1,
                label="Opacity"
            )

            # Optional side selector for nose rings
            side_dropdown = None
            style_dropdown = None

            if show_side:
                side_dropdown = gr.Dropdown(
                    choices=["left", "right", "septum"],
                    value="left",
                    label="Placement Side"
                )

            if show_style:
                style_dropdown = gr.Dropdown(
                    choices=["stud", "hoop", "nath"],
                    value="stud",
                    label="Ring Style"
                )

            try_on_btn = gr.Button(f"Try On {jewelry_label}", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Result",
                type="numpy",
                height=500,
                elem_classes="result-image"
            )
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )

    # Connect the remove background button
    remove_bg_btn.click(
        fn=remove_background,
        inputs=[jewelry_input],
        outputs=[jewelry_input]
    )

    # Connect the try-on button based on available options
    if show_side and show_style:
        try_on_btn.click(
            fn=process_fn,
            inputs=[person_input, jewelry_input, opacity_slider, side_dropdown, style_dropdown],
            outputs=[output_image, status_text]
        )
    elif show_side:
        # Wrap function to add default style
        def wrapped_fn(person, jewelry, opacity, side):
            return process_fn(person, jewelry, opacity, side, "stud")
        try_on_btn.click(
            fn=wrapped_fn,
            inputs=[person_input, jewelry_input, opacity_slider, side_dropdown],
            outputs=[output_image, status_text]
        )
    else:
        try_on_btn.click(
            fn=process_fn,
            inputs=[person_input, jewelry_input, opacity_slider],
            outputs=[output_image, status_text]
        )


def create_ai_jewelry_tab():
    """Create the AI-powered jewelry generation tab with customization options."""

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ### AI Jewelry Generation

            Generate custom jewelry on your photo using AI!
            Select jewelry type, customize options, and describe your desired piece.

            **Powered by:** Replicate trained model (ganeshgowri-asa/naari-jewelry-vton)
            """)

            person_input = gr.Image(
                label="Person Photo",
                type="numpy",
                sources=["upload", "webcam"],
                height=300
            )

            # Jewelry type selection
            jewelry_type_dropdown = gr.Dropdown(
                choices=list(JEWELRY_TYPES.keys()),
                value="necklace",
                label="Jewelry Type",
                info="Select the type of jewelry to generate"
            )

            # Customization options in an accordion
            with gr.Accordion("Customization Options", open=True):
                with gr.Row():
                    metal_type_dropdown = gr.Dropdown(
                        choices=METAL_TYPES,
                        value="gold",
                        label="Metal Type"
                    )
                    stones_dropdown = gr.Dropdown(
                        choices=STONE_TYPES,
                        value="none",
                        label="Stones"
                    )

                style_dropdown = gr.Dropdown(
                    choices=STYLE_OPTIONS.get("necklace", ["default"]),
                    value=STYLE_OPTIONS.get("necklace", ["default"])[0],
                    label="Style"
                )

            # Text prompt for additional customization
            jewelry_prompt = gr.Textbox(
                label="Jewelry Description (optional)",
                placeholder="Describe additional details... e.g., 'intricate floral pattern', 'minimalist design'",
                lines=2
            )

            opacity_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=1.0,
                step=0.1,
                label="Opacity"
            )

            # API status indicator
            api_status = gr.Markdown(
                f"**Replicate API Status:** {'Available' if REPLICATE_AVAILABLE else 'Not configured (set REPLICATE_API_TOKEN)'}"
            )

            generate_btn = gr.Button(
                "Generate Jewelry",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Result",
                type="numpy",
                height=500,
                elem_classes="result-image"
            )
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3
            )

    # Update style dropdown when jewelry type changes
    jewelry_type_dropdown.change(
        fn=update_style_dropdown,
        inputs=[jewelry_type_dropdown],
        outputs=[style_dropdown]
    )

    # Connect the generate button
    generate_btn.click(
        fn=process_ai_jewelry_tryon,
        inputs=[
            person_input,
            jewelry_prompt,
            jewelry_type_dropdown,
            metal_type_dropdown,
            stones_dropdown,
            style_dropdown,
            opacity_slider
        ],
        outputs=[output_image, status_text]
    )


def create_garment_tab():
    """Create the garment virtual try-on tab."""

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ### Garment Virtual Try-On

            Upload a person photo and a garment image to see how it looks!

            **Coming Soon:** This feature will use IDM-VTON for realistic garment try-on.
            """)

            person_input = gr.Image(
                label="Person Photo",
                type="numpy",
                sources=["upload", "webcam"],
                height=300
            )

            garment_input = gr.Image(
                label="Garment Image",
                type="numpy",
                sources=["upload"],
                height=300
            )

            garment_type = gr.Dropdown(
                choices=["upper_body", "lower_body", "full_body"],
                value="upper_body",
                label="Garment Type"
            )

            with gr.Accordion("Advanced Settings", open=False):
                denoise_steps = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=30,
                    step=5,
                    label="Denoise Steps"
                )

                seed = gr.Slider(
                    minimum=-1,
                    maximum=2147483647,
                    value=42,
                    step=1,
                    label="Seed (-1 for random)"
                )

            try_on_btn = gr.Button("Try On Garment", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Result",
                type="numpy",
                height=500,
                elem_classes="result-image"
            )
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )

    # Connect the try-on button
    try_on_btn.click(
        fn=process_garment_tryon,
        inputs=[person_input, garment_input, garment_type, denoise_steps, seed],
        outputs=[output_image, status_text]
    )


def create_app():
    """Create the Gradio application with both garment and jewelry tabs."""

    with gr.Blocks(theme=THEME, css=CSS, title="Naari Studio - Virtual Try-On") as app:
        # Header
        gr.Markdown("""
        # Naari Studio - Virtual Try-On

        Experience AI-powered virtual try-on for garments and jewelry!

        **Tips for best results:**
        - Use a well-lit, front-facing photo
        - Ensure face/shoulders are clearly visible
        - Use jewelry images with transparent backgrounds for best results
        - Click "Remove Background" to auto-remove jewelry image backgrounds
        """)

        # Main tabs for Garment vs Jewelry
        with gr.Tabs(elem_classes="main-tabs") as main_tabs:

            # ================================================================
            # GARMENT TAB
            # ================================================================
            with gr.Tab("Garment Try-On", id="garment"):
                gr.Markdown("""
                ## Garment Virtual Try-On

                Try on clothes virtually using AI! Upload your photo and a garment image.
                """)
                create_garment_tab()

            # ================================================================
            # JEWELRY TAB
            # ================================================================
            with gr.Tab("Jewelry Try-On", id="jewelry"):
                gr.Markdown("""
                ## Jewelry Virtual Try-On

                Try on various types of jewelry using AI generation or image overlay.
                - **AI Generate**: Create custom jewelry with text prompts
                - **Upload & Overlay**: Use your own jewelry images
                """)

                # Jewelry sub-tabs
                with gr.Tabs():
                    # AI Generation Tab (NEW)
                    with gr.Tab("AI Generate"):
                        create_ai_jewelry_tab()

                    with gr.Tab("Necklace"):
                        create_jewelry_tab(
                            jewelry_type="necklace",
                            jewelry_label="Necklace",
                            description="Try on necklaces - works best with visible shoulders and neck area. Uses pose detection landmarks 9, 10, 11, 12 for accurate positioning.",
                            process_fn=process_necklace
                        )

                    with gr.Tab("Earrings"):
                        create_jewelry_tab(
                            jewelry_type="earrings",
                            jewelry_label="Earrings",
                            description="Try on earrings - upload a single earring image (will be mirrored for both ears). Uses face mesh earlobe landmarks for accurate positioning.",
                            process_fn=process_earrings
                        )

                    with gr.Tab("Maang Tikka"):
                        create_jewelry_tab(
                            jewelry_type="maang_tikka",
                            jewelry_label="Maang Tikka",
                            description="Try on traditional Indian forehead jewelry - works best with visible forehead. Uses face mesh hairline landmarks for accurate positioning.",
                            process_fn=process_maang_tikka
                        )

                    with gr.Tab("Nose Ring"):
                        create_jewelry_tab(
                            jewelry_type="nose_ring",
                            jewelry_label="Nose Ring",
                            description="Try on nose rings/nath - select placement side and style. Uses face mesh nostril landmarks for accurate positioning.",
                            process_fn=process_nose_ring,
                            show_side=True,
                            show_style=True
                        )

            # ================================================================
            # API DOCUMENTATION TAB
            # ================================================================
            with gr.Tab("API", id="api"):
                gr.Markdown("""
                ## API Documentation

                ### /api/jewelry-tryon Endpoint

                The jewelry try-on functionality is available as a programmatic API.
                You can call it using Gradio's API client or direct HTTP requests.

                #### Parameters:

                | Parameter | Type | Required | Description |
                |-----------|------|----------|-------------|
                | `person_image` | Image | Yes | Person photo (uploaded file) |
                | `jewelry_prompt` | string | Yes | Text description of desired jewelry |
                | `jewelry_type` | string | No | Type: necklace, earrings, bangles, rings, maang_tikka, nose_ring (default: necklace) |
                | `metal_type` | string | No | Metal: gold, silver, rose gold, platinum, oxidized silver, antique gold (default: gold) |
                | `stones` | string | No | Stones: diamond, ruby, emerald, sapphire, pearl, kundan, polki, none (default: none) |
                | `style` | string | No | Style variant (depends on jewelry type) |

                #### Example Python Usage:

                ```python
                from gradio_client import Client

                client = Client("GaneshGowri/naari-avatar")
                result = client.predict(
                    person_image="path/to/image.jpg",
                    jewelry_prompt="elegant bridal necklace",
                    jewelry_type="necklace",
                    metal_type="gold",
                    stones="kundan",
                    style="choker",
                    api_name="/api/jewelry-tryon"
                )
                ```

                #### Style Options by Jewelry Type:

                | Jewelry Type | Available Styles |
                |--------------|-----------------|
                | Necklace | choker, princess, matinee, opera, statement, layered, pendant |
                | Earrings | studs, drops, hoops, chandeliers, jhumkas, cuffs |
                | Bangles | traditional, modern, kada, charm, cuff, tennis |
                | Rings | solitaire, band, cluster, eternity, cocktail, stackable |
                | Maang Tikka | bridal, simple, elaborate, kundan, pearl |
                | Nose Ring | stud, hoop, nath, septum |

                #### Response:

                Returns a dictionary with:
                - `success`: boolean indicating operation success
                - `image`: Result image (numpy array)
                - `message`: Status message
                - `prompt_used`: Full prompt sent to the model
                """)

                # API test interface
                gr.Markdown("### Try the API")

                with gr.Row():
                    with gr.Column():
                        api_person_image = gr.Image(
                            label="Person Image",
                            type="numpy",
                            sources=["upload"]
                        )
                        api_prompt = gr.Textbox(
                            label="Jewelry Prompt",
                            value="elegant gold necklace with diamonds"
                        )
                        api_jewelry_type = gr.Dropdown(
                            choices=list(JEWELRY_TYPES.keys()),
                            value="necklace",
                            label="Jewelry Type"
                        )
                        api_metal = gr.Dropdown(
                            choices=METAL_TYPES,
                            value="gold",
                            label="Metal Type"
                        )
                        api_stones = gr.Dropdown(
                            choices=STONE_TYPES,
                            value="diamond",
                            label="Stones"
                        )
                        api_test_btn = gr.Button("Test API", variant="primary")

                    with gr.Column():
                        api_output = gr.Image(label="API Result", type="numpy")
                        api_status = gr.Textbox(label="API Response", lines=4)

                def test_api(person_image, prompt, jewelry_type, metal, stones):
                    if person_image is None:
                        return None, "Error: Please upload a person image"
                    result = api_jewelry_tryon(
                        person_image_path=person_image,
                        jewelry_prompt=prompt,
                        jewelry_type=jewelry_type,
                        metal_type=metal,
                        stones=stones
                    )
                    return result.get("image"), json.dumps({
                        "success": result["success"],
                        "message": result["message"],
                        "prompt_used": result["prompt_used"]
                    }, indent=2)

                api_test_btn.click(
                    fn=test_api,
                    inputs=[api_person_image, api_prompt, api_jewelry_type, api_metal, api_stones],
                    outputs=[api_output, api_status]
                )

        # Footer
        gr.Markdown("""
        ---
        **Naari Studio** - AI-Powered Virtual Try-On

        Built with:
        - cvzone PoseModule & MediaPipe Face Mesh for landmark detection
        - Replicate trained model (ganeshgowri-asa/naari-jewelry-vton:f6b844b4) for AI jewelry generation
        - Gradio for the web interface

        [GitHub](https://github.com/ganeshgowri/naari-vton) | Powered by HuggingFace Spaces
        """)

    return app


# Create and launch the app
app = create_app()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
