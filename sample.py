from utils import MaskMaker, PolkaDotMaker
from bodypix.image import load_image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model_path = './tflitemodels/mobilenet-float-multiplier-050-stride16-float16.tflite'
    maker = MaskMaker(model_path)

    img_path = './sample_images/img_1.jpg'
    image_array = load_image(img_path, max_size=800)
    masks = maker.run(image_array, 0.80)

    dot_mask = PolkaDotMaker().run(*masks, min_r=10)
    image = PolkaDotMaker.chroma_key(image_array.copy(), dot_mask, (128, 255, 0))
    plt.imshow(image)
    plt.show()

