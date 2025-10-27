# 🔥 FIRES (Forest Ignition and Risk Evaluation System):

**FIRES** is a machine learning–powered web application designed to **predict forest fires from satellite imagery**. By analyzing environmental and spatial data, FIRES identifies areas at high risk for wildfires, enabling early prevention and saving lives, ecosystems, and infrastructure.

---

## 🌍 Overview

FIRES allows users to **upload satellite images** of a specific region to determine whether that area is at risk of a wildfire.  
The system uses a **custom-trained convolutional neural network (MobileNetV2 + MLP)** that achieves **97% accuracy** on validation data, significantly outperforming traditional risk prediction models.

After processing the input image, FIRES provides:
- A **prediction** (yes/no) on wildfire risk.  
- **Actionable safety tips** on mitigating or preventing wildfires.  
- An **analytics dashboard** to review past predictions and outcomes.

---

## 🧠 Features

-  **Satellite Image Analysis:** Upload an image of a region to detect potential wildfire risks.  
-  **High-Accuracy Predictions:** 97% detection accuracy with MobileNetV2 + Multi-Layer Perceptron architecture.  
-  **Smart Prevention Tips:** Offers tailored recommendations to mitigate wildfire effects.  
-  **Analytics Section:** Review historical inputs, outputs, and detection trends.  
-  **Web-Based Interface:** Simple, dark-themed, and responsive design for accessibility across devices.

---

## ⚙️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| **Frontend** | CSS / Python (Streamlit) |
| **Backend** | Python |
| **Machine Learning** | TensorFlow, Keras, NumPy, OpenCV |
| **Model Architecture** | MobileNetV2 + MLP Classifier |
| **Deployment** | Localhost / Streamlit Cloud |

---

## 📈 Model Performance

| Metric | Result |
|--------|--------|
| Accuracy | 97% |
| Loss | 0.08 |
| Model Type | Binary Classification (Fire / No Fire) |

---

## How It Works

1. **Upload Satellite Image**  
   The user drags and drops a file for analysis.  

2. **Model Prediction**  
   The trained model processes the image to determine if the area is at risk of a wildfire.  

3. **Results Display**  
   FIRES provides the prediction result and customized tips based on the analysis.  

4. **Analytics Dashboard**  
   Users can view consolidated data from previous predictions to track changes over time.

---

## Future Improvements

- Integrate **additional data sources** (temperature, humidity, vegetation index) to improve accuracy.  
- Add **LLM integration (e.g., ChatGPT API)** for enhanced feedback and interpretability.  
- Build a **mobile version** with real-time push notifications for early warnings.  
- Deploy FIRES on a **public cloud platform** for broader accessibility.

---

## Inspiration

FIRES was inspired by the increasing frequency of wildfires and the devastating effects of climate-related disasters. After personally experiencing severe environmental damage in my own community, I realized the need for an accessible, predictive tool to help prevent tragedies before they happen.

---

## Author

**Developer:** Varun Nukala  
**Email:** [varun.k.nukala@gmail.com@example.com]  
**Project:** Congressional App Challenge Submission 2025  
**Location:** Helotes, Texas

---
