import numpy as np
import streamlit as st
from PIL import Image
from utils import MaskMaker, PolkaDotMaker

model_path = './tflitemodels/mobilenet-float-multiplier-050-stride16-float16.tflite'
maker = MaskMaker(model_path)

from PIL import Image

def resize_image(image: Image.Image, max_size):
    """
    This function takes in a PIL.Image variable called image and an int variable called max_size,
    and resizes the image so that the width or height (whichever is larger) does not exceed max_size.
    """
    # Obtain original width and height of image.
    width, height = image.size

    # Get the larger dimension to determine how much to scale the image down by.
    if width > height:
        scale_factor = max_size / width
    else:
        scale_factor = max_size / height

    # If neither dimension exceeds the maximum size, return the original image.
    if scale_factor >= 1:
        return image

    # Otherwise, resize the image.
    new_width = int(round(width * scale_factor))
    new_height = int(round(height * scale_factor))
    resized_image = image.resize((new_width, new_height))

    return resized_image

with st.sidebar:
    th = st.slider('Threshold value', 0.3, 0.95, 0.75)
    min_r = st.slider('min_r', 5, 50, 30)
    seed = st.slider('seed', 0, 2048, 0)

st.title('水玉コラ生成クソアプリ')

uploaded_image = st.file_uploader(
    "画像を以下からアップロードしてください",
    type=['png', 'jpg', 'bmp'])

run_button = st.button("実行", key=1)

col1, col2= st.columns(2)
col1.header("元画像")
col2.header("コラ画像")

img_array = None

with col1:
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image = resize_image(image, 800)
        img_array = np.array(image, dtype='uint8')
        st.image(img_array, use_column_width=None)

if run_button:
    if img_array is not None:
        masks = maker.run(img_array, th)
        dot_mask = PolkaDotMaker(seed).run(*masks, min_r=min_r)
        image = PolkaDotMaker.chroma_key(img_array.copy(), dot_mask, (128, 255, 0))
        col2.image(image, use_column_width=None)
    else:
        st.error('画像を入力してください')
