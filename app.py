"""
Naari Studio - Jewelry Virtual Try-On
Gradio Web Interface for HuggingFace Spaces

Upload a person photo and jewelry image to virtually try on:
- Necklaces
- Earrings
- Maang Tikka (Indian forehead jewelry)
- Nose Rings

Powered by MediaPipe for accurate landmark detection.
"""

import gradio as gr
from PIL import Image
import numpy as np
from typing import Optional, Tuple

from jewelry_engine import apply_jewelry, remove_jewelry_background


# Theme configuration
THEME = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="pink",
    neutral_hue="slate",
)

# CSS for better styling
CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.tab-nav button {
    font-size: 16px !important;
    font-weight: 600 !important;
}
.result-image {
    min-height: 400px;
}
footer {
    visibility: hidden;
}
"""


def process_necklace(person_image: Optional[np.ndarray],
                     jewelry_image: Optional[np.ndarray],
                     opacity: float) -> Tuple[Optional[np.ndarray], str]:
    """Process necklace try-on request."""
    if person_image is None:
        return None, "Please upload a person photo."
    if jewelry_image is None:
        return None, "Please upload a necklace image."

    # Convert numpy to PIL
    person_pil = Image.fromarray(person_image)
    jewelry_pil = Image.fromarray(jewelry_image)

    # Apply jewelry
    result, message = apply_jewelry(person_pil, jewelry_pil, "necklace", opacity)

    if result is not None:
        return np.array(result.convert('RGB')), message
    return None, message


def process_earrings(person_image: Optional[np.ndarray],
                     jewelry_image: Optional[np.ndarray],
                     opacity: float) -> Tuple[Optional[np.ndarray], str]:
    """Process earrings try-on request."""
    if person_image is None:
        return None, "Please upload a person photo."
    if jewelry_image is None:
        return None, "Please upload an earring image."

    person_pil = Image.fromarray(person_image)
    jewelry_pil = Image.fromarray(jewelry_image)

    result, message = apply_jewelry(person_pil, jewelry_pil, "earrings", opacity)

    if result is not None:
        return np.array(result.convert('RGB')), message
    return None, message


def process_maang_tikka(person_image: Optional[np.ndarray],
                        jewelry_image: Optional[np.ndarray],
                        opacity: float) -> Tuple[Optional[np.ndarray], str]:
    """Process maang tikka try-on request."""
    if person_image is None:
        return None, "Please upload a person photo."
    if jewelry_image is None:
        return None, "Please upload a maang tikka image."

    person_pil = Image.fromarray(person_image)
    jewelry_pil = Image.fromarray(jewelry_image)

    result, message = apply_jewelry(person_pil, jewelry_pil, "maang_tikka", opacity)

    if result is not None:
        return np.array(result.convert('RGB')), message
    return None, message


def process_nose_ring(person_image: Optional[np.ndarray],
                      jewelry_image: Optional[np.ndarray],
                      opacity: float,
                      side: str) -> Tuple[Optional[np.ndarray], str]:
    """Process nose ring try-on request."""
    if person_image is None:
        return None, "Please upload a person photo."
    if jewelry_image is None:
        return None, "Please upload a nose ring image."

    person_pil = Image.fromarray(person_image)
    jewelry_pil = Image.fromarray(jewelry_image)

    result, message = apply_jewelry(person_pil, jewelry_pil, "nose_ring", opacity, side=side)

    if result is not None:
        return np.array(result.convert('RGB')), message
    return None, message


def remove_background(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Remove background from jewelry image."""
    if image is None:
        return None

    pil_image = Image.fromarray(image)
    result = remove_jewelry_background(pil_image)

    return np.array(result.convert('RGBA'))


def create_jewelry_tab(jewelry_type: str,
                       jewelry_label: str,
                       description: str,
                       process_fn,
                       show_side: bool = False):
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

            if show_side:
                side_dropdown = gr.Dropdown(
                    choices=["left", "right"],
                    value="left",
                    label="Side"
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

    # Connect the try-on button
    if show_side:
        try_on_btn.click(
            fn=process_fn,
            inputs=[person_input, jewelry_input, opacity_slider, side_dropdown],
            outputs=[output_image, status_text]
        )
    else:
        try_on_btn.click(
            fn=process_fn,
            inputs=[person_input, jewelry_input, opacity_slider],
            outputs=[output_image, status_text]
        )


def create_app():
    """Create the Gradio application."""

    with gr.Blocks(theme=THEME, css=CSS, title="Naari Studio - Jewelry VTON") as app:
        # Header
        gr.Markdown("""
        # Naari Studio - Jewelry Virtual Try-On

        Upload a clear, front-facing photo and a jewelry image to see how it looks on you!

        **Tips for best results:**
        - Use a well-lit, front-facing photo
        - Ensure face/shoulders are clearly visible
        - Use jewelry images with transparent backgrounds for best results
        """)

        # Tabs for different jewelry types
        with gr.Tabs():
            with gr.Tab("Necklace"):
                create_jewelry_tab(
                    jewelry_type="necklace",
                    jewelry_label="Necklace",
                    description="Try on necklaces - works best with visible shoulders and neck area.",
                    process_fn=process_necklace
                )

            with gr.Tab("Earrings"):
                create_jewelry_tab(
                    jewelry_type="earrings",
                    jewelry_label="Earrings",
                    description="Try on earrings - upload a single earring image (will be mirrored for both ears).",
                    process_fn=process_earrings
                )

            with gr.Tab("Maang Tikka"):
                create_jewelry_tab(
                    jewelry_type="maang_tikka",
                    jewelry_label="Maang Tikka",
                    description="Try on traditional Indian forehead jewelry - works best with visible forehead.",
                    process_fn=process_maang_tikka
                )

            with gr.Tab("Nose Ring"):
                create_jewelry_tab(
                    jewelry_type="nose_ring",
                    jewelry_label="Nose Ring",
                    description="Try on nose rings/nath - select which side to place the jewelry.",
                    process_fn=process_nose_ring,
                    show_side=True
                )

        # Footer
        gr.Markdown("""
        ---
        **Naari Studio** - Virtual Try-On for Indian Jewelry

        Built with MediaPipe and Gradio | [GitHub](https://github.com/ganeshgowri/naari-vton)
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
