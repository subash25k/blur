import logging
import os

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps, ImageFilter

from transformers import pipeline
from typing import Tuple, Optional

#configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackgroundBlur:
    def __init__(self, model_path: str):
        """initialize the background class with model path"""
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()

    def load_model(self) -> None:
        """load segmentation model"""
        try:
            self.model = pipeline("image-segmentation",
                                  model=self.model_path,
                                 device=self.device)
            logger.info(f"Model loaded successfully on {self.device}")
            # Use a pipeline as a high-level helper

        except Exception as e:
            logger.error(f"Error loading model:{str(e)}")
            st.error("Failed to laod the segmentation model.please check the path")

    def refine_mask(self, mask: Image.Image, mask_array=None) -> Image.Image:

        #convert mask to numpy array(mask)

        #apply threshold to make the mask more decisive
        threshold: int = 128

        mask_array = np.where(mask_array > threshold, 255, 0).astype(np.uint8)

        #convert
        refined_mask = Image.fromarray(mask_array)
        refined_mask = refined_mask.filter(ImageFilter.GaussianBlur(radius=0.5))
        return refined_mask

    def process_image(self, image: Image.Image, blur_level: int) -> tuple[Image, None] | tuple[str, None]:
        try:
            result = self.model(Images=image)
            mask = result[0]['mask']

            refined_mask = self.refine_mask(mask)

            mask_inverted = ImageOps.invert(refined_mask)

            background = image.copy()
            background = background.filter(ImageFilter.GaussianBlur(radius=blur_level))

            final_image = Image.composite(image, background, mask_inverted)

            return final_image, None

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return str(e), None


def main():
    st.write("App has started successfully.")  # Add this at the beginning of the main()

    st.set_page_config(page_title="Background blur app", layout="wide")

    st.markdown("""
    <style>
    .stButton>button {
    width: 100%;
    }
    .stImage {
    display:flex;
    justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None

        model_path = os.path.join("..", "Model", "models--mattmdjaga--segformer-b2-clothes",
                                  "snapshots", "fc92b3abe7b123c814ca7910683151f2b7b7281e")

        st.write(f"Model path: {model_path}")

        processor = BackgroundBlur(model_path)

        st.title("subash project 5 : background blur app")
        st.markdown("upload an image to blur its background while keeping the subject in focus")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("upload image")
        uploaded_image = st.file_uploader("choose a photo", type=["jpg", "jpeg", "png"])

        if uploaded_image:
            original_image = Image.open(uploaded_image)
            st.image(original_image, caption="original image", use_column_width=True)

    with col2:
        st.subheader("processed Image ")
        if uploaded_image:
            blur_level = st.slider("blur intensity",
                                   min_value=0,
                                   max_value=30,
                                   value=15,
                                   step=1,
                                   help="adjust the intensity of the background blur")
            process_button = st.button("process Image")

            if process_button or st.session_state.processed_image is None:
                with st.spinner('processing image...'):
                    final_image, error = processor.process_image(original_image, blur_level)

                    if error:
                        st.error(f"error processing image:{error}")
                    else:
                        st.session_state.processed_image = final_image
                        st.image(final_image, caption="processed image", use_column_width=True)

                        if final_image:
                            import io
                            buf = io.BytesIO()
                            final_image.save(buf, format="PNG")
                            byte_im = buf.getvalue()

                            st.download_button(
                                label="downlaoded processed image",
                                data=byte_im,
                                file_name="processed_image.png",
                                mime="image/png"
                            )
