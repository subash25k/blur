import logging
import os

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps, ImageFilter
from transformers import pipeline
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackgroundBlur:
    def __init__(self, model_path: str):
        """Initialize the background blur class with a model path."""
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()

    def load_model(self) -> None:
        """Load the Hugging Face segmentation model."""
        try:
            self.model = pipeline(
                "image-segmentation",
                model=self.model_path,
                device=0 if self.device == "cuda" else -1
            )
            logger.info(f"Model loaded successfully on {self.device}")
            st.write(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            st.error(f"Error loading model: {str(e)}")

    def refine_mask(self, mask: Image.Image, mask_array=None) -> Image.Image:
        try:
            # Ensure that mask is not None and is converted to an array
            if mask is None:
                raise ValueError("The mask is None. Cannot refine mask.")

            # Convert mask to numpy array (if not already done)
            mask_array = np.array(mask) if mask_array is None else mask_array

            # Apply threshold to make the mask more decisive
            threshold: int = 128
            mask_array = np.where(mask_array > threshold, 255, 0).astype(np.uint8)

            # Convert back to PIL Image
            refined_mask = Image.fromarray(mask_array)

            # Apply slight Gaussian blur to refine the mask
            refined_mask = refined_mask.filter(ImageFilter.GaussianBlur(radius=0.5))
            return refined_mask

        except Exception as e:
            logger.error(f"Error refining mask: {str(e)}")
            raise e

    def process_image(self, image: Image.Image, blur_level: int) -> Tuple[Optional[Image.Image], Optional[str]]:
        try:
            # Run the model on the image
            result = self.model(inputs=image)

            # Debugging: Log the result to see what we get from the model
            logger.info(f"Model output: {result}")

            # Get the segmentation mask from the model output
            mask = result[0].get('mask')  # Use .get() to safely retrieve the mask

            # Debugging: Check if the mask is None or not
            if mask is None:
                logger.error("No mask found in the model output.")
                return None, "No mask found in the model output."

            # Refine the mask
            refined_mask = self.refine_mask(mask)

            # Invert the refined mask
            mask_inverted = ImageOps.invert(refined_mask)

            # Create a blurred background
            background = image.copy()
            background = background.filter(ImageFilter.GaussianBlur(radius=blur_level))

            # Combine the original image with the blurred background using the inverted mask
            final_image = Image.composite(image, background, mask_inverted)

            return final_image, None

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None, str(e)


def main():
    st.set_page_config(page_title="Background Blur App", layout="wide")

    st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .stImage {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None

    model_path = "mattmdjaga/segformer_b2_clothes"
    st.write(f"Using model: {model_path}")

    processor = BackgroundBlur(model_path)

    st.title("Subash Project 5: Background Blur App")
    st.markdown("Upload an image to blur its background while keeping the subject in focus.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Image")
        uploaded_image = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

        if uploaded_image:
            original_image = Image.open(uploaded_image)
            st.image(original_image, caption="Original Image", use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        if uploaded_image:
            blur_level = st.slider("Blur Intensity", min_value=0, max_value=30, value=15, step=1)
            process_button = st.button("Process Image")

            if process_button:
                with st.spinner('Processing image...'):
                    final_image, error = processor.process_image(original_image, blur_level)

                    if error:
                        st.error(f"Error processing image: {error}")
                    else:
                        st.session_state.processed_image = final_image
                        st.image(final_image, caption="Processed Image", use_column_width=True)
                        if final_image:
                            import io
                            buf = io.BytesIO()
                            final_image.save(buf, format="PNG")
                            byte_im = buf.getvalue()

                            st.download_button(
                                label="Download Processed Image",
                                data=byte_im,
                                file_name="processed_image.png",
                                mime="image/png"
                            )

if __name__ == "__main__":
    main()
