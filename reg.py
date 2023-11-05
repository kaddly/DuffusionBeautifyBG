import torch
import onnxruntime
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import os


model_path = '/root/autodl-tmp/models/huge.onnx'
res = 1024

# Initialize session and get prediction
session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def get_rg(image: Image.Image):
    im_h, im_w = image.size
    image_org = image
    image = image.resize((res,res), Image.BICUBIC)
    image = np.array(image)
    image = image / 255.0 - 0.5
    image = image.transpose(2,0,1)
    image = np.expand_dims(image, axis = 0).astype('float32')

    alpha = session.run([output_name], {input_name: image})

    matte = np.zeros((im_w, im_h, 4), dtype=np.uint8)
    alpha = alpha[0][0][0]
    # alpha = alpha.transpose(1, 2, 0)
    alpha = cv2.resize(alpha, [im_h, im_w])

    alpha = alpha * 255


    # alpha = alpha.astype(np.uint8)
    # th, alpha_in = cv2.threshold(alpha,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # print(th)
    # alpha_ouline = cv2.adaptiveThreshold(alpha, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,0)

    # alpha = alpha_in + alpha_ouline
    # alpha = alpha.clip(0,255)

    # kernel = np.ones((3, 3), np.uint8)
    # alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=3)
    # alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=3)

    alpha[alpha>10]=255
    alpha[alpha<=10]=0

    matte[:, :, :3] = image_org
    matte[:, :, 3] = alpha

    return matte, Image.fromarray(alpha).convert('L')



def get_img_mask(img):
    return get_rg(img)


def save_mask_img(img_dir_path):
    os.makedirs(os.path.join(img_dir_path, "img_mask"), exist_ok=True)
    for img_name in tqdm(os.listdir(img_dir_path)):
        if img_name == "img_mask" or os.path.exists(os.path.join(img_dir_path, "img_mask", img_name)):
            continue
        img_path = os.path.join(img_dir_path, img_name)
        img = Image.open(img_path)
        _, mask = get_img_mask(img)
        mask.save(os.path.join(img_dir_path, "img_mask", img_name))



# if __name__ == '__main__':
#     gr.Interface(get_rg, inputs=[gr.Image(type='pil')], outputs=[gr.Image(), gr.Image(type='pil')]).launch(server_name='0.0.0.0', server_port=7878)
#     # save_mask_img('/home/oneway/cxz_workspace/LLMData/IC/ECommerce-IC/img')
#     # save_mask_img('/home/oneway/cxz_workspace/LLMData/IC/ECommerce-IC/val_img')
