# Fashion Classifier: End-to-End AI & Data Pipeline 👕🤖

An end-to-end Computer Vision pipeline designed to process, augment, and classify fashion items from raw images. This project demonstrates a full data lifecycle: from raw metadata parsing and cleaning using Pandas, to training a Deep Learning model, and preparing the architecture for web dashboard integration.

## 🎯 Objective
To build a scalable image classification model capable of distinguishing between 5 different clothing categories. The primary focus of this repository is to showcase robust data engineering practices, modular Python architecture, and transfer learning capabilities.

## 🛠️ Tech Stack
* **Data Engineering & Analysis:** Python, Pandas, NumPy, OpenCV.
* **Deep Learning:** TensorFlow, Keras (Transfer Learning via VGG16).
* **Data Visualization:** Matplotlib, Seaborn.
* **Environment:** Python 3.12, Virtual Environments (`.venv`).

## 🧠 The Pipeline
1. **Data Preparation:** Raw metadata parsing and data cleaning using **Pandas** to structure the dataset before training.
2. **Model Architecture (`pipeline_utils.py`):** Leveraging VGG16 pre-trained on ImageNet. The base layers are frozen, and custom dense layers with Dropout and Batch Normalization are added to prevent overfitting.
3. **Training & Execution (`main.ipynb`):** Clean, modular execution environment using `ImageDataGenerator` for real-time data augmentation.
4. **Integration:** The model outputs are structured to be easily consumed by a frontend web dashboard interface.

## 📊 Results & Diagnostics
The model successfully validates the end-to-end technical pipeline. However, as an engineering exercise, it currently runs on a highly constrained "dummy" dataset. 

**Limitations & Future Improvements:**
* **Dataset Size:** The validation set currently relies on a minimal sample size. This causes a "staircase" effect in the accuracy learning curves, as each individual prediction heavily skews the percentage.
* **Class Confusion:** While the model achieves perfect recall on distinct shapes like trousers and footwear, the confusion matrix reveals a bias where upper-body garments (shirts) are frequently misclassified as coats.
* **Next Steps:** Scaling the raw data ingestion pipeline to process a larger, balanced dataset (e.g., DeepFashion) to improve generalization and fine-tune the upper layers of the VGG16 base.

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Fashion-Classifier.git](https://github.com/your-username/Fashion-Classifier.git)
   cd Fashion-Classifier
2. Set up the virtual environment (Python 3.12 recommended):

Bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
Install dependencies:

Bash
pip install tensorflow pandas opencv-python matplotlib seaborn scikit-learn
Prepare your data:
Place your categorized images inside the data/train/ and data/validation/ directories, maintaining subfolders for each class.

Execute:
Run all cells sequentially in main.ipynb.

Developed by Manuel Deserti - Computer Engineering Student
