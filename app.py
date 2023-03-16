import numpy as np
import streamlit as st
import cv2
from PIL import Image
from utils import MaskMaker, PolkaDotMaker
from streamlit_extras.buy_me_a_coffee import button

model_path = './tflitemodels/mobilenet-float-multiplier-050-stride16-float16.tflite'
maker = MaskMaker(model_path)


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
    st.header('各種パラメータ')
    th = st.slider('身体範囲推定のしきい値', 0.3, 0.95, 0.75)
    min_r = st.slider('水玉の最小半径', 5, 50, 15)
    seed = st.slider('乱数シード', 0, 2048, 0)
    hide_method = st.radio('隠し方', ['固定色', '固定肌色+すりガラス', 'inpaint+すりガラス'])
    st.text('固定色')
    color_R = st.slider('R', 0, 255, 188)
    color_G = st.slider('G', 0, 255, 226)
    color_B = st.slider('B', 0, 255, 232)

st.title('水玉コラ自動生成ツール')
st.subheader('絵に対してはできないのであしからず。')

text, coffee = st.columns(2)
with text:
    st.text('パラメータで色々調整できます')
    st.text('乱数ガチャで最高にえっちな画像を作ろう')
with coffee:
    button(username="apiss", floating=False, width=221)

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
        try:
            masks = maker.run(img_array, th)
            dot_mask = PolkaDotMaker(seed).run(*masks, min_r=min_r)
            if hide_method == '固定色':
                image = PolkaDotMaker.chroma_key(img_array, dot_mask, (color_R, color_G, color_B))
            elif hide_method == '固定肌色+すりガラス':
                dot_mask = np.repeat(np.expand_dims(dot_mask, -1), 3, -1)
                skin_color = img_array[masks[-1]==1].mean(axis=0).astype('uint8')
                blur = PolkaDotMaker.chroma_key(img_array, masks[1], skin_color)
                blur = cv2.GaussianBlur(blur, (51, 51), 70)
                image = np.where(dot_mask==1, blur, img_array)
            elif hide_method == 'inpaint+すりガラス':
                dot_mask = np.repeat(np.expand_dims(dot_mask, -1), 3, -1)
                blur = cv2.inpaint(img_array, masks[1], 3, cv2.INPAINT_TELEA)
                blur = cv2.GaussianBlur(blur, (51, 51), 70)
                image = np.where(dot_mask==1, blur, img_array)
            col2.image(image, use_column_width=None)
        except Exception as e:
            st.error(f'不明なエラーが発生しました: {e}')
    else:
        st.error('画像を入力してください')
