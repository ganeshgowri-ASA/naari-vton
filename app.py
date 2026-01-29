"""
Naari Studio - Virtual Try-On Application
HuggingFace Spaces compatible Gradio interface.

Features:
- Garment Virtual Try-On (placeholder for IDM-VTON integration)
- Jewelry Virtual Try-On (Necklace, Earrings, Maang Tikka, Nose Ring)

Powered by cvzone PoseModule and MediaPipe Face Mesh for accurate landmark detection.
"""

import gradio as gr
from PIL import Image
import numpy as np
from typing import Optional, Tuple
import logging

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

from jewelry_engine import apply_jewelry, remove_jewelry_background


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

                Try on various types of jewelry using pose and face detection.
                Select a jewelry type below to get started.
                """)

                # Jewelry sub-tabs
                with gr.Tabs():
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

        # Footer
        gr.Markdown("""
        ---
        **Naari Studio** - AI-Powered Virtual Try-On

        Built with cvzone PoseModule, MediaPipe Face Mesh, and Gradio

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
