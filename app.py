import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import numpy as np

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Image Captioning & Segmentation",
    page_icon="🖼️",
    layout="wide"
)

# ============ CUSTOM CSS ============
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #e0f7fa, #fff3e0);
            color: #333;
            font-family: 'Trebuchet MS', sans-serif;
        }
        .caption-box {
            background-color: #fff8e1;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #ffb300;
            font-size: 18px;
            color: #5d4037;
            font-weight: bold;
        }
        .seg-title {
            background-color: #f1f8e9;
            padding: 8px;
            border-radius: 5px;
            color: #33691e;
            font-weight: bold;
            text-align: center;
        }
        .uploaded-img {
            border: 2px solid #90caf9;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ============ SIDEBAR ============
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=100)
st.sidebar.title("⚙️ About This Project")
st.sidebar.info("""
**Image Captioning & Segmentation App**  
- **Captioning** → BLIP (Pre-trained Transformer Model)  
- **Segmentation** → Mask R-CNN (Pre-trained COCO model)  
- Developed by **Riya Bhardwaj** ❤️
""")
st.sidebar.markdown("---")
st.sidebar.write("💡 **Tip**: Upload a clear image for better results.")

# ============ MODELS ============
@st.cache_resource
def load_models():
    cap_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    seg_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    seg_model.eval()
    return cap_processor, cap_model, seg_model

processor, caption_model, seg_model = load_models()
transform = T.Compose([T.ToTensor()])

# ============ FUNCTIONS ============
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def get_segmented_image(image, threshold=0.5):
    img_tensor = transform(image)
    with torch.no_grad():
        prediction = seg_model([img_tensor])
    masks = prediction[0]['masks']
    scores = prediction[0]['scores']
    img_np = np.array(image)

    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            img_np[mask > 128] = img_np[mask > 128] * 0.5 + color * 0.5
    return img_np

# ============ MAIN UI ============
st.title("🖼️ **AI Image Captioning & Segmentation**")
st.write("### Let AI **understand** and **highlight** objects in your images ✨")

uploaded_file = st.file_uploader("📤 **Upload an image (jpg/png):**", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Two Columns Layout
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True, output_format="PNG", channels="RGB")
    with col2:
        st.markdown('<p class="seg-title">🎨 Segmented Output</p>', unsafe_allow_html=True)

    if st.button("✨ Generate Caption & Segmentation"):
        with st.spinner("⏳ Analyzing your image..."):
            caption = generate_caption(image)
            seg_image = get_segmented_image(image)
        col2.image(seg_image, use_column_width=True)

        st.markdown(f'<p class="caption-box">📝 Caption: {caption}</p>', unsafe_allow_html=True)
