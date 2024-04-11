import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Function to predict mango disease
def predict_mango(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Function to predict tomato disease
def predict_tomato(image, model):
    # Predict the disease
    prediction = model.predict(image)
    return prediction

# Function to preprocess tomato image
def preprocess_tomato_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1  # Normalize to [-1, 1] range
    return normalized_image_array.reshape((1, 224, 224, 3))

# Function to display mango prediction
# List of mango remedies
mango_remedies = [
    "Prune affected plant parts to remove infected areas.Apply fungicides containing copper or chlorothalonil following label instructions.Improve air circulation around plants by proper spacing and pruning.",
    "Prune infected branches at least 6 inches below the visible symptoms.Apply copper-based bactericides during dormant periods.Avoid overhead watering to reduce moisture on foliage.",
    "Inspect cuttings before planting and remove any larvae or adult weevils.Use insecticidal soap or neem oil to control adult weevils.Practice good sanitation to eliminate breeding sites.",
    "Prune affected branches to healthy tissue, making cuts at least 6 inches below visible symptoms.Ensure proper watering and fertilization to maintain plant vigor.Avoid mechanical injuries to plant tissue.",
    "Remove and destroy infected plant parts.Apply insecticidal soap or neem oil to control larvae.Encourage natural predators like parasitic wasps.",
    "No remedy required (Healthy)",
    "Apply fungicides containing sulfur, potassium bicarbonate, or neem oil following label instructions.Improve air circulation by proper plant spacing and pruning.Water plants at the base to avoid wetting foliage.",
    "Address the underlying pest infestation, such as aphids or scale insects, as sooty mold usually develops on their honeydew.Control the pest population using insecticidal soap or neem oil.Physically remove sooty mold with a gentle stream of water or wipe with a damp cloth."
]

# List of tomato remedies
tomato_remedies = [
    "Plant is Healthy. No remedy required",
    "Apply copper-based fungicides and practice crop rotation to manage bacterial spot in tomato plants",
    "Remove infected leaves, apply fungicides containing chlorothalonil, and ensure proper spacing for air circulation to prevent early blight",
    "Use fungicides containing chlorothalonil or copper-based products, practice crop rotation, and avoid overhead irrigation to control late blight in tomato plants.",
    "Increase ventilation, space plants properly, and apply fungicides containing copper or chlorothalonil to manage leaf mold in tomato plants.",
    "Remove infected leaves, apply fungicides containing chlorothalonil, and water at the base of plants to prevent the spread of septoria leaf spot.",
    "Use insecticidal soap or neem oil, increase humidity levels, and employ natural predators like ladybugs to control spider mites on tomato plants.",
    "Remove infected leaves, apply fungicides containing chlorothalonil, and avoid overhead irrigation to manage target spot in tomato plants.",
    "There is no cure for tomato mosaic virus. Remove and destroy infected plants, practice strict sanitation measures, and plant virus-resistant varieties to prevent its spread.",
    "Use reflective mulches, employ insecticidal sprays to control whiteflies, and remove infected plants to manage tomato yellow leaf curl virus."
]

disease_info = {
    'Anthracnose': {
        'description': "Anthracnose is a fungal disease that affects many tropical fruits, including mangoes. It causes dark lesions on leaves, stems, and fruit.",
        'impact': "Anthracnose-infected fruits may develop dark lesions, making them unattractive and reducing market value."
    },
    'Bacterial Canker': {
        'description': "Bacterial Canker is a common bacterial disease in mango trees. It causes dark lesions on leaves and stems.",
        'impact': "Bacterial Canker can cause fruit drop and premature ripening, leading to lower yields and reduced marketability."
    },
    'Cutting Weevil': {
        'description': "Cutting Weevil is a pest that damages the stems and leaves of mango trees. It can cause wilting and stunted growth.",
        'impact': "Cutting Weevil-infested trees may produce fewer and smaller fruits, affecting market supply and quality."
    },
    'Die Back': {
        'description': "Die Back is a fungal disease that causes the dieback of branches and leaves. It can lead to severe damage if left untreated.",
        'impact': "Die Back can weaken mango trees, reducing fruit production and affecting market availability."
    },
    'Gall Midge': {
        'description': "Gall Midge is an insect pest that affects young leaves and shoots of mango trees. It causes swelling and distortion of leaves.",
        'impact': "Gall Midge damage can reduce the aesthetic appeal of fruits, affecting market demand."
    },
    'Healthy': {
        'description': "No disease detected. Your mango plant appears to be healthy.",
        'impact': "Healthy trees produce high-quality fruits, enhancing market value and consumer satisfaction."
    },
    'Powdery Mildew': {
        'description': "Powdery Mildew is a fungal disease that affects mango leaves, causing a white powdery growth on the surface.",
        'impact': "Powdery Mildew-infected leaves may drop prematurely, reducing tree vigor and affecting fruit quality."
    },
    'Sooty Mould': {
        'description': "Sooty Mould is a fungal disease that grows on honeydew secretions of insects like aphids. It appears as a black coating on leaves and stems.",
        'impact': "Sooty Mould can reduce photosynthesis in leaves, affecting tree growth and fruit development."
    }
}


# Function to display mango prediction and remedies
def display_mango_prediction(prediction):
    class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy',
                   'Powdery Mildew', 'Sooty Mould']
    detected_disease = class_names[np.argmax(prediction)]
    confidence_score = prediction[0][np.argmax(prediction)]
    remedy = mango_remedies[np.argmax(prediction)]
    info = disease_info.get(detected_disease,{"description": "No information available.", "impact": "No impact information available."})
    st.subheader("Mango Disease Prediction")
    st.write(f"**Detected Disease:** {detected_disease}")
    st.write(f"**Confidence Score:**", confidence_score)
    st.write(f"**Remedy:** {remedy}")
    st.write(f"**Description:** {info['description']}")
    st.write(f"**Impact on Final Produce:** {info['impact']}")

# Function to display tomato prediction and remedies
def display_tomato_prediction(prediction):
    class_names = open("tomato_labels.txt", "r").readlines()
    class_name = class_names[np.argmax(prediction)].strip()  # Remove any trailing whitespace
    confidence_score = prediction[0][np.argmax(prediction)]
    remedy = tomato_remedies[np.argmax(prediction)]

    # Basic information about tomato diseases
    tomato_disease_info = {
        'Healthy': {
            'description': "The plant appears to be healthy with no signs of disease.",
            'impact': "Healthy plants produce high-quality fruits suitable for the market, contributing to better market value."
        },
        'Tomato___Bacterial_spot': {
            'description': "Bacterial Spot is a common bacterial disease in tomatoes. It causes dark, water-soaked spots on leaves and fruits.",
            'impact': "Bacterial Spot-infected fruits may develop lesions, making them unattractive and reducing market value."
        },
        'Tomato___Early_blight': {
            'description': "Early Blight is a fungal disease that affects tomato plants. It causes dark spots on leaves and can lead to defoliation.",
            'impact': "Early Blight can reduce tomato yields and affect fruit quality, leading to lower market prices."
        },
        'Tomato___Late_blight': {
            'description': "Late Blight is a serious fungal disease that affects tomatoes, causing brown lesions on leaves and fruits.",
            'impact': "Late Blight can devastate tomato crops, leading to significant yield losses and reduced market availability."
        },
        'Tomato___Leaf_Mold': {
            'description': "Leaf Mold is a fungal disease that affects tomato foliage, causing yellowing and wilting of leaves.",
            'impact': "Leaf Mold-infected plants may produce fewer and smaller fruits, affecting market supply and quality."
        },
        'Tomato__Septoria_leaf_spot': {
            'description': "Septoria Leaf Spot is a fungal disease that affects tomato leaves, causing small, dark spots with white centers.",
            'impact': "Septoria Leaf Spot can defoliate tomato plants, reducing photosynthesis and affecting fruit development."
        },
        'Tomato___Spider_mites Two-spotted_spider_mite': {
            'description': "Spider Mites are tiny pests that feed on tomato leaves, causing stippling and webbing on the undersides.",
            'impact': "Spider Mite infestations can weaken tomato plants, reducing yields and affecting market quality."
        },
        'Tomato_Target_Spot': {
            'description': "Target Spot is a fungal disease that affects tomato leaves, causing circular lesions with concentric rings.",
            'impact': "Target Spot can defoliate tomato plants, reducing photosynthesis and affecting fruit development."
        },
        'Tomato___Tomato_mosaic_virus': {
            'description': "Tomato Mosaic Virus is a viral disease that affects tomatoes, causing mosaic patterns on leaves and stunted growth.",
            'impact': "Tomato Mosaic Virus can reduce tomato yields and affect fruit quality, leading to lower market prices."
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
            'description': "Tomato Yellow Leaf Curl Virus is a viral disease transmitted by whiteflies. It causes curling and yellowing of leaves.",
            'impact': "Tomato Yellow Leaf Curl Virus can severely damage tomato crops, leading to significant yield losses and reduced market availability."
        }
    }

    # Fetch basic information about the detected disease
    info = tomato_disease_info.get(class_name, {"description": "No information available.",
                                                "impact": "No impact information available."})

    # Display information
    st.subheader("Tomato Disease Prediction")
    st.write(f"**Detected Disease:** {class_name}")
    st.write(f"**Confidence Score:**",confidence_score)
    st.write(f"**Remedy:** {remedy}")
    st.write(f"**Description:** {info['description']}")
    st.write(f"**Impact on Final Produce:** {info['impact']}")


# Main function
def main():
    st.title("Plant Leaf Disease Detection")

    # Sidebar navigation
    page = st.sidebar.selectbox("Select Disease Detection Page", ("Home", "Mango", "Tomato"))

    if page == "Home":
        st.write("Welcome to Plant Leaf Disease Detection App!")
        st.write("This app helps in detecting diseases in plant leaves, particularly in mango and tomato plants.")
        st.write("Please select a disease detection page from the sidebar.")
        st.image("download (2).jpeg", use_column_width=True)

    elif page == "Mango":
        st.write("Mango Disease Detection")
        uploaded_image = st.file_uploader("Upload an image of mango leaf", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            mango_model = load_model("mango_model.h5")
            prediction = predict_mango(image, mango_model)
            display_mango_prediction(prediction)

    elif page == "Tomato":
        st.write("Tomato Disease Detection")
        uploaded_image = st.file_uploader("Upload an image of tomato leaf", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            preprocessed_image = preprocess_tomato_image(image)
            tomato_model = load_model("tm.h5")
            prediction = predict_tomato(preprocessed_image, tomato_model)  # Pass preprocessed image to predict_tomato
            display_tomato_prediction(prediction)

if __name__ == "__main__":
    main()
