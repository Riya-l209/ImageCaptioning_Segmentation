# 🖼️ AI Image Captioning & Segmentation

This project performs **automatic image captioning** and **object segmentation** using state-of-the-art deep learning models.  
Upload an image, and the app will:

✅ **Generate a meaningful caption** describing the image.  
✅ **Highlight objects** using segmentation masks.

---

## 🌐 **Live Demo**
Try the app here: [🔗 **Click to Open**](https://imagecaptioningsegmentation-atczy55nxgx2jjqo42rsqc.streamlit.app/)

---

## 🚀 **Features**
- **Captioning Model** → [ViT-GPT2 Image Captioning (nlpconnect/vit-gpt2)](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)  
- **Segmentation Model** → [Mask R-CNN (torchvision)](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html)  
- User-friendly **Streamlit web app** with clean UI.

---

## 🛠 **Tech Stack**
- **Python**, **Streamlit** (for deployment)  
- **PyTorch**, **Transformers** (Hugging Face models)  
- **OpenCV, NumPy, Matplotlib**

---

## 📂 **Project Structure**

Image_Captioning_Segmentation/
├── app.py # Streamlit web app
├── requirements.txt # Project dependencies
├── captioning_segmentation.ipynb # Colab notebook (training & experimentation)
└── README.md # Project documentation

---

## ▶ **Run Locally**
1. Clone the repo:  
```bash
git clone https://github.com/Riya-l209/ImageCaptioning_Segmentation.git
cd image-captioning-segmentation

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py
```

🎓 Learning Outcomes
Hands-on experience with Computer Vision + NLP integration.
Understanding of pre-trained transformer models for vision tasks.
Deployment of ML models using Streamlit Community Cloud.

👩‍💻 Author
Riya Bhardwaj
Data Science & Computer Vision Enthusiast
