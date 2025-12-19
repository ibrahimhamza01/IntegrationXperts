# IntegrationXperts

## Overview
This project performs **gender, emotion, and age detection** using deep learning models.  
It combines **YOLOv8**, **VGG16**, and pretrained face/emotion models to process videos and images.


## Project Workflow

### 1. Gender Classification (Males, Females, Kids)
- Initial dataset: **UTKFace + custom images**  
- Models used: **YOLOv8** and **VGG16**  
- Issue: Original models trained on **face-cropped images**, cannot predict full-length images  
- Solution: Use face detection to crop images → predict gender (still limited features)

### 2. Full-Length Male & Female Classification
- Collected full-length images from **stock videos**  
- Annotated using **labelImg**  
- Trained YOLOv8:
  - Initial training: 206 epochs (stopped due to plateau)
  - Improved dataset: 250 epochs → good results  
- Confusion matrices and test results generated on unseen images and videos

### 3. Best Face Selection in Videos
- Workflow:
  1. Detect gender using YOLOv8
  2. Crop face region using MTCNN
  3. Compute facial embeddings using InceptionResnetV1
  4. Compare embeddings using cosine similarity → save unique faces
  5. Save cropped face images in appropriate class directories

### 4. Emotion and Age Detection
- Used **DeepFace** pretrained model for emotion/age
- Workflow:
  - Detect faces in video frames using MTCNN
  - Predict emotion & age using DeepFace
- Combined results for multiple videos

### 5. Final Integration
- Combined gender, emotion, and age detection into **one pipeline**
- Notes:
  - Gender detection fast on CPU
  - Emotion/age detection slow due to frame-by-frame redundancy
  - Optimized by using Hugging Face model `dima806/facial_emotions_image_detection`


## Repository Structure

IntegrationXperts/
├── code/ # Scripts and YOLO/VGG code
├── notebooks/ # Jupyter notebooks
├── configs/ # Configuration files
├── README.md # Project description + links
└── .gitignore # Ignored files (weights, reports, ppt)


## Large Files

The following large files are hosted on **Hugging Face**:

- 🤖 YOLO Weights: [Download](https://huggingface.co/ibrahimhamza01/yolo-integrationxperts/tree/main/weights)  
- 📄 Final Report: [Download](https://huggingface.co/ibrahimhamza01/yolo-integrationxperts/blob/main/final%20report.docx)  
- 📊 Presentation: [Download](https://huggingface.co/ibrahimhamza01/yolo-integrationxperts/blob/main/presentation.pptx)  



## Usage

1. Clone this repo:

```bash
git clone https://github.com/USERNAME/IntegrationXperts.git
cd IntegrationXperts

```

2. Install required dependencies (example):

```bash
pip install -r requirements.txt
```

3. Run the code on your own images or videos.

4. Download YOLO weights from Hugging Face before running training/inference.

##Notes

1. Models trained on full-length images after dataset improvement

2. Accuracy improves with more consistent data

3. Emotion/age detection may be slow due to frame-by-frame processing
