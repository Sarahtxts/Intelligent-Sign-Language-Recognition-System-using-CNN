# Intelligent-Sign-Language-Recognition-System-using-CNN 

### 🏫 Mini Project – B.Tech Artificial Intelligence and Data Science  
**Rajalakshmi Institute of Technology, Chennai**  
Developed by **Sarah S V (2117230070142)**  

---

## 📖 Overview  

**Intelligent Sign Language Recognition Using CNN** is a **deep learning–based project** designed to bridge the communication gap between the hearing- and speech-impaired community and non-signers.  

The system leverages **Convolutional Neural Networks (CNNs)** and **computer vision** to recognize **American Sign Language (ASL)** gestures in real time.  
Trained on the **ASL Alphabet Dataset from Kaggle**, the model classifies hand gestures corresponding to alphabets and special signs, enabling real-time assistive communication.

---

## 🎯 Objectives  

- Recognize multiple ASL gestures with high accuracy.  
- Achieve **real-time classification** using webcam input.  
- Ensure robustness under varied **lighting, background, and orientation** conditions.  
- Build a **lightweight and scalable model** suitable for desktops and embedded platforms.  
- Promote **inclusive communication** through AI-driven accessibility tools.  

---

## 🧩 Features  

✅ High classification accuracy (95%+)  
✅ Real-time recognition using webcam (OpenCV)  
✅ Lightweight CNN architecture optimized for low-resource devices  
✅ Robust preprocessing (resizing, normalization, augmentation)  
✅ Scalable design retrainable for other sign language datasets  
✅ Promotes inclusivity through AI-powered communication  

---

## 🧠 Dataset Details  

**Dataset Name:** [American Sign Language (ASL) Alphabet Dataset – Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  

**Dataset Summary:**  
- Total Images: **87,000 labeled RGB images**  
- Classes: **29 (A–Z + space, del, nothing)**  
- Image Size: **200×200 px** (resized to 64×64 during preprocessing)  

**Data Split:**  
| Dataset | Images | Description |
|----------|---------|-------------|
| Training | 69,600 | Used for model training |
| Validation | 17,400 | Used for model tuning and evaluation |

---

## ⚙️ Technologies Used  

| Category | Tools & Libraries |
|-----------|------------------|
| Programming Language | Python 3.7+ |
| Deep Learning Framework | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib |
| Development Environment | Google Colab / Jupyter Notebook |

---

## 🧮 Model Architecture  

**Input:** 64×64 RGB images  

**Architecture Overview:**  
- `Conv2D → ReLU → MaxPooling2D`  
- `Conv2D → ReLU → MaxPooling2D`  
- `Flatten → Dense (128 units, ReLU) → Dropout(0.5)`  
- `Dense (29 units, Softmax)`  

**Training Details:**  
- Optimizer: `Adam`  
- Loss Function: `Categorical Crossentropy`  
- Batch Size: `32`  
- Epochs: `10–15`  

---

## 🔧 System Architecture  

1. **Data Acquisition:** Capture static images or webcam input.  
2. **Preprocessing:** Resize, normalize, and augment data.  
3. **Model Training:** Train CNN for feature extraction and classification.  
4. **Real-Time Prediction:** Classify gestures via webcam with instant feedback.  

---

## 🧑‍💻 Implementation Steps  

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/<your-username>/Intelligent-Sign-Language-Recognition-CNN.git
cd Intelligent-Sign-Language-Recognition-CNN
```

###2️⃣ Install Dependencies

```
pip install tensorflow keras opencv-python numpy pandas matplotlib
```

3️⃣ Download the Dataset

Download the ASL Alphabet Dataset from Kaggle and place it in the /dataset folder.
```
kaggle datasets download -d grassknoted/asl-alphabet
```

4️⃣ Train the Model
```
python train_model.py
```

5️⃣ Run Real-Time Recognition

```
python real_time_recognition.py
```

# ASL Alphabet Recognition Project

📊 **Results**

| Metric              | Value          |
|--------------------|---------------|
| Accuracy            | 95.2%         |
| Precision           | 94.1%         |
| Recall              | 93.6%         |
| F1-Score            | 93.85%        |
| FPS                 | 20–30         |
| Prediction Latency  | ~0.12 sec/frame |

🔍 **Highlights**
- Stable recognition across various lighting conditions.
- Minor confusion for visually similar signs (e.g., M vs N).
- Consistent real-time predictions on standard laptop hardware.

🧪 **Sample Outputs**
- **Dataset Prediction:** CNN model accurately classifies alphabets (A–Z) and special signs.
- **Real-Time Prediction:** Webcam feed integrated via OpenCV — predicted label displayed live.

🚀 **Future Enhancements**
- Support dynamic gesture recognition using RNN/LSTM.
- Expand to Indian and British Sign Languages.
- Develop a mobile/web app for accessibility.
- Integrate text-to-speech (TTS) for audio output.
- Improve dataset diversity (hand sizes, skin tones, backgrounds).

🤝 **Contribution**
Contributions are always welcome!  
```bash
# Fork the repository
git checkout -b feature-name

# Commit your changes
git commit -m "Added new feature"

# Push to your branch
git push origin feature-name

# Open a Pull Request
```

## 👩‍💻 Author
**Sarah S V**  
B.Tech – Artificial Intelligence and Data Science  
Rajalakshmi Institute of Technology, Chennai  
📧 [Add your email or LinkedIn profile link]

---

## 📚 References
- ASL Alphabet Dataset – [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet)  
- François Chollet, *Deep Learning with Python* (Manning, 2017)  
- Ian Goodfellow et al., *Deep Learning* (MIT Press, 2016)  
- [TensorFlow & Keras Documentation](https://www.tensorflow.org/)  
- [OpenCV Official Documentation](https://opencv.org/)

---

## 💬 Closing Note
This project demonstrates how Deep Learning and Computer Vision can make technology more inclusive and human-centered.
