# Documentation

## Objective
The script aims to classify the quality of input images as either "Good Quality" (GO), "Not Good Quality" (NG), or "None" based on their similarity scores with precomputed embeddings of GO and NG images using the ResNet50 model.

## Solution
1. Cropping Image
The first stage is to crop an input image based on its contours and save the cropped image as "cropped.png".

2. Image Embedding:c
The next step is to generate image embeddings for each input image using the pre-trained ResNet50 model. Image embedding is the process of representing images as fixed-length feature vectors. The ResNet50 model is a deep convolutional neural network that has been trained on a large dataset for image recognition tasks. The model is used here with the "pooling='avg'" argument, which means it outputs a 2048-dimensional feature vector for each input image.

3. Loading Precomputed Embeddings:
The script assumes that the embeddings of GO and NG images have been precomputed and stored in "GO.npy" and "NG.npy" files, respectively. These files contain the image embeddings generated using the same ResNet50 model. The script loads these embeddings into memory for similarity comparison with the input images.

4. Similarity Calculation:
For each input image, the script calculates its similarity scores with the GO and NG embeddings using cosine similarity. Cosine similarity is a measure of similarity between two non-zero vectors. It ranges from -1 to 1, where 1 indicates identical vectors, -1 indicates opposite vectors, and 0 indicates no similarity.

5. Quality Classification:
The similarity scores are then compared against pre-defined threshold values for GO and NG classifications. 
- If the similarity score with GO embeddings is higher than the GO threshold and the similarity score with NG embeddings is higher than the NG threshold, the image is classified as "Good Quality" (GO).
- If the similarity score with GO embeddings is higher than the GO threshold but not with NG embeddings, the image is also classified as "Good Quality" (GO).
- If the similarity score with NG embeddings is higher than the NG threshold but not with GO embeddings, the image is classified as "Not Good Quality" (NG).
- Otherwise, the image is classified as "None."

6. Output and Result Storage:
The script creates output directories named "GO" and "NG" within the specified output directory. For each input image, it copies the image to the appropriate output directory based on its quality classification (GO or NG). If the image is classified as "None," it is not copied to any output directory.

## Usage

The script can be run from the command line with the following optional arguments:

--path: Path to the folder containing input images for batch processing.
--img_path: Path to a single input image for processing.
--outdir: Output directory to save results (default: "result").
The script will create output directories for "GO" and "NG" classifications within the main output directory.

For each input image, the script will calculate its quality (GO, NG, or None) and copy the image to the respective output directory based on the classification.

## Example Usage:

Process images in a folder:
```bash
python main.py --path input_folder --outdir output_folder
```

Process a single image:
```bash
python main.py --img_path input_image.jpg --outdir output_folder
```

**note** : Before running the script, ensure that the necessary libraries in the `requirements.txt` are installed in the Python environment.

## Solutions to Improve the Performance

1. Use More Data:
Acquiring more labeled data, especially diverse and representative samples of "GO" and "NG" Tip, can significantly improve the model's ability to generalize. Larger datasets allow the model to learn more robust features and patterns, leading to better detection results.

2. Domain Knowledge for Labeling:
In scenarios where labeled data is scarce, domain experts can help label the data more accurately. Their expertise can ensure that the dataset contains correct and meaningful labels, minimizing labeling errors and improving the model's performance.

3. Segmentation Method for Crack Percentage:
Employing image segmentation techniques to calculate the percentage of the image covered by cracks can provide valuable additional information about the severity of cracks. By incorporating this information into the classification process, the model can make more nuanced decisions, such as categorizing cracks based on their size or severity.

