# 🫁 Lung Disease Detection from Chest X-Rays using GANs

A final year B.Tech project built using a hybrid deep learning architecture combining **GANs, CapsNet, and VGG-16** to detect and classify lung diseases from X-ray images. Features a Django-based web app with a text-to-speech interface for result narration.

---

## 🚀 Features
- Predicts **lung diseases** like Pneumonia, Lung Cancer, and other thoracic conditions
- Uses **GANs** to synthetically balance the dataset and improve CNN accuracy
- Hybrid deep learning model combining **CapsNet + VGG-16**
- Trained on the **NIH Chest X-ray Dataset** (112,120 images)
- Easy-to-use **Django web interface** with **text-to-speech output**
- Visual disease localization on uploaded X-rays

---

## 🧠 Technologies Used
- Python, Django, TensorFlow, Keras, OpenCV
- CapsNet, VGG-16 (transfer learning)
- pyttsx3 (text-to-speech)
- Google Colab for training with free GPU access

---

## 🗂️ Dataset
**NIH Chest X-ray Dataset**
- 112,120 X-rays from 30,805 patients
- 14 disease categories + "No findings"
- Dataset used from: [NIH Kaggle Link](https://www.kaggle.com/datasets/nih-chest-xrays/data)

---

## 🛠 Installation
```bash
# Clone the repo
git clone https://github.com/abhinavpatel202/lung-disease-detection.git
cd lung-disease-detection

# Activate virtual environment (recommended)
conda activate lungenv

# Install requirements
pip install -r requirements.txt

# Run the Django app
python manage.py runserver
```


---

## 🧪 How the System Works – Step-by-Step

1. **User uploads a chest X-ray image** through the Django-based web interface.
2. The backend loads the pre-trained **hybrid CapsNet + VGG-16 CNN model**.
3. **GAN-generated samples** help balance class distribution for robust classification.
4. Model predicts whether the image indicates:
   - **Normal lung**
   - **Pneumonia**
   - **Lung Cancer**
5. The result is not just displayed — it is **read aloud using pyttsx3** for accessibility.
6. The infected region is optionally **highlighted** to aid in visual inspection by doctors.

---

## 📸 Screenshots & Demo (Optional)

> _Add screenshots, architecture diagrams, or demo videos here._  
> _Examples: `model_accuracy.png`, `architecture.png`, `sample_output.png`_

---

## 📦 Model File (Trained Weights)

Due to GitHub's 100MB limit, the trained model `NN.h5` is not included in this repo.  
📁 **[Download from Google Drive](https://your-drive-link.com)** ← _Replace with your link_

- Model was trained on **NIH Chest X-ray Dataset** with GAN-augmented samples
- Achieves classification accuracy ~99% with robust generalization

---

## 👨‍🔬 Authors

- **Abhinav Patel** – [@abhinavpatel202](https://github.com/abhinavpatel202)  
- **Hardik Patel**  
- **Janmey Patel**

> Guided by **Prof. Richa Sharma**  
> Department of Information Technology  
> Dwarkadas J. Sanghvi College of Engineering  
> University of Mumbai

---

