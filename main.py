import cv2
import numpy as np
import os
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import argparse

def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser()
    
    # Add arguments for the script
    parser.add_argument("--path", type=str, default="", help='folder path')
    parser.add_argument("--img_path", type=str, default="", help="image_path [jpg,jpeg,png]")
    parser.add_argument("--outdir", type=str, default="result", help="save results to /folder")
    
    # Parse known args
    args= parser.parse_args()
    return args

def make_outdir(args):
    # Create output directories if they don't exist
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    if not os.path.exists(args.outdir + "/GO"):
        os.mkdir(args.outdir + "/GO")
    if not os.path.exists(args.outdir + "/NG"):
        os.mkdir(args.outdir + "/NG")

def crop_img(path):
    # Function to crop the image based on its contours
    img = cv2.imread(path)
    if img.mean() < 5:
        new_img = img * 2
    else:
        new_img = img
    
    blur = cv2.GaussianBlur(new_img, (7, 7), 1)
    canny = cv2.Canny(blur, 100, 300)  
    kernel = np.ones((40, 40), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=6)
    cnt, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(cnt) > 0:
        x, y, w, h = cv2.boundingRect(cnt[0])
        img = img[y:y+h, x:x+w]
    cv2.imwrite("cropped.png", img)

def image_embedding(img_path):
    # Function to generate image embedding using ResNet50
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    pred = model.predict(x)
    return pred

def GO_similiarity_score(img_path):
    # Function to calculate similarity score with GO (Good Quality) embeddings
    GO = np.load("GO.npy")
    emebdding = image_embedding(img_path)
    similiarity = cosine_similarity(GO, emebdding)
    return similiarity

def NG_similiarity_score(img_path):
    # Function to calculate similarity score with NG (Not Good Quality) embeddings
    NG = np.load("NG.npy")
    emebdding = image_embedding(img_path)
    similiarity = cosine_similarity(NG, emebdding)
    return similiarity
 
def quality_decision(img_path, GO_thres=0.839, NG_thres=0.76):
    # Function to decide image quality based on similarity scores
    crop_img(img_path)
    GO_similiarity = GO_similiarity_score("cropped.png")
    NG_similiarity = NG_similiarity_score("cropped.png")
    os.remove("cropped.png")
    
    if GO_similiarity >= GO_thres and NG_similiarity >= NG_thres:
        quality = "GO"
    elif GO_similiarity >= GO_thres:
        quality = "GO"
    elif NG_similiarity >= NG_thres:
        quality = "NG"
    else:
        quality = "None"
    
    return quality

def run():
    args = parse_args()
    make_outdir(args)
    
    quality_map = {
        "GO": args.outdir + "/GO",
        "NG": args.outdir + "/NG"
    }
    
    if args.path != "":
        # Process images from a folder
        try:
            img_path = os.listdir(args.path)
            for img in img_path:
                quality = quality_decision(os.path.join(args.path, img))
                print(img + ": " + quality)
                if quality in quality_map:
                    shutil.copyfile(os.path.join(args.path, img), os.path.join(quality_map[quality], img))
        except Exception as e:
            print(e)

    elif args.img_path != "":
        # Process a single image
        try:
            quality = quality_decision(args.img_path)
            print(args.img_path + ": " + quality)
            if quality in quality_map:
                shutil.copyfile(args.img_path, os.path.join(quality_map[quality], "result.png"))
        except Exception as e:
            print(e)
    
    else:
        print("No input images!")

if __name__ == "__main__":
    run()


