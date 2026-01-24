"""
Naari Studio - Virtual Try-On Application
Jewelry Try-On powered by OpenCV face detection
"""

import gradio as gr
import numpy as np
from PIL import Image
import os

from jewelry_engine import JewelryTryOnEngine, apply_jewelry_pil


# Initialize the jewelry engine
engine = JewelryTryOnEngine()


def process_jewelry_tryon(
    person_image: Image.Image,
    jewelry_image: Image.Image,
    jewelry_type: str,
    opacity: float,
    hand_selection: str = "Both Hands"
) -> tuple:
    """
    Process jewelry try-on request.

    Args:
        person_image: PIL Image of the person
        jewelry_image: PIL Image of the jewelry
        jewelry_type: Type of jewelry
        opacity: Opacity level
        hand_selection: For bangles - which hand(s)

    Returns:
        Tuple of (result_image, status_message)
    """
    if person_image is None:
        return None, "Please upload a person image"

    if jewelry_image is None:
        return None, "Please upload a jewelry image"

    # Map UI jewelry type to engine type
    type_mapping = {
        "Necklace": "necklace",
        "Earrings": "earrings",
        "Maang Tikka": "maang_tikka",
        "Bangles": "bangles"
    }

    jewelry_type_key = type_mapping.get(jewelry_type, "necklace")

    # Map hand selection
    hand_mapping = {
        "Both Hands": "both",
        "Left Hand Only": "left",
        "Right Hand Only": "right"
    }
    hand = hand_mapping.get(hand_selection, "both")

    # Apply jewelry
    result, info = apply_jewelry_pil(
        person_image,
        jewelry_image,
        jewelry_type_key,
        opacity,
        hand=hand
    )

    if info["success"]:
        status = f"Success! Applied {jewelry_type} with {int(opacity * 100)}% opacity"
        if info.get("face_rect"):
            x, y, w, h = info["face_rect"]
            status += f"\nFace detected at: ({x}, {y}) - Size: {w}x{h}"
    else:
        status = f"Error: {info['message']}"

    return result, status


def load_sample_jewelry(jewelry_type: str) -> Image.Image:
    """Load a sample jewelry image based on type."""
    assets_dir = "assets/jewelry"

    type_files = {
        "Necklace": "necklace.png",
        "Earrings": "earring.png",
        "Maang Tikka": "maang_tikka.png",
        "Bangles": "bangle.png"
    }

    filename = type_files.get(jewelry_type)
    if filename:
        filepath = os.path.join(assets_dir, filename)
        if os.path.exists(filepath):
            return Image.open(filepath)

    return None


def update_hand_visibility(jewelry_type: str):
    """Show/hide hand selection based on jewelry type."""
    return gr.update(visible=(jewelry_type == "Bangles"))


def create_demo_interface():
    """Create the Gradio demo interface."""

    # Create theme - compatible with both Gradio 4.x and 5.x+
    try:
        theme = gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="pink"
        )
    except Exception:
        theme = "soft"

    with gr.Blocks(title="Naari Studio - Jewelry Try-On", theme=theme) as demo:
        gr.Markdown("""
        # Naari Studio - Jewelry Virtual Try-On

        Upload your photo and try on beautiful jewelry virtually!

        **Supported Jewelry Types:**
        - **Necklace** - Positioned at neck area
        - **Earrings** - Positioned at ear level
        - **Maang Tikka** - Positioned at forehead center
        - **Bangles** - Positioned at wrist area

        **Tips for best results:**
        - Use a clear, front-facing photo
        - Good lighting helps with face detection
        - Use jewelry images with transparent backgrounds (PNG)
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Images")

                person_input = gr.Image(
                    label="Your Photo",
                    type="pil",
                    height=350
                )

                jewelry_input = gr.Image(
                    label="Jewelry Image (PNG with transparency)",
                    type="pil",
                    height=200
                )

                gr.Markdown("### Settings")

                jewelry_type = gr.Dropdown(
                    choices=["Necklace", "Earrings", "Maang Tikka", "Bangles"],
                    value="Necklace",
                    label="Jewelry Type"
                )

                hand_selection = gr.Radio(
                    choices=["Both Hands", "Left Hand Only", "Right Hand Only"],
                    value="Both Hands",
                    label="Hand Selection (for Bangles)",
                    visible=False
                )

                opacity_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                    label="Opacity"
                )

                apply_btn = gr.Button(
                    "Apply Jewelry",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Result")

                result_output = gr.Image(
                    label="Try-On Result",
                    type="pil",
                    height=450
                )

                status_output = gr.Textbox(
                    label="Status",
                    lines=3
                )

        # Event handlers
        jewelry_type.change(
            fn=update_hand_visibility,
            inputs=[jewelry_type],
            outputs=[hand_selection]
        )

        apply_btn.click(
            fn=process_jewelry_tryon,
            inputs=[
                person_input,
                jewelry_input,
                jewelry_type,
                opacity_slider,
                hand_selection
            ],
            outputs=[result_output, status_output]
        )

        # Example section
        gr.Markdown("---")
        gr.Markdown("### How to Use")
        gr.Markdown("""
        1. **Upload your photo** - A clear front-facing photo works best
        2. **Upload jewelry image** - PNG images with transparent background work best
        3. **Select jewelry type** - Choose the type that matches your jewelry
        4. **Adjust opacity** - Lower opacity for a subtle effect
        5. **Click Apply** - See your virtual try-on result!

        **Note:** For bangles, you can select which hand(s) to apply them to.
        """)

        gr.Markdown("---")
        gr.Markdown("""
        <center>

        **Naari Studio** - AI-Powered Virtual Try-On

        Built with OpenCV Face Detection | Powered by Gradio

        </center>
        """)

    return demo


# Create and launch the app
demo = create_demo_interface()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
