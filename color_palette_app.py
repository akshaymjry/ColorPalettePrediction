import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import streamlit as st

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

def predict_image(uploaded_file):
    model = ResNet50(weights='imagenet')

    img = Image.open(uploaded_file)

    # Resize and preprocess the image as per ResNet50 standards
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded_predictions = decode_predictions(preds, top=1)[0]
    prediction = decoded_predictions[0][1]

    return prediction

def main():
    st.title("Color Palette and Image Prediction")
    st.write(
        "Upload an image, generate a color palette using K-Means clustering, and get a prediction of what the image possibly contains.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)

        # Resize the image to fit within the column width
        image = resize_image(image)
        image_array = np.array(image)

        # Reshape the array to (height * width, num_channels)
        pixels = image_array.reshape(-1, 3)

        # Slider for number of palette colors
        num_colors = st.slider("Select the number of colors in the palette:", 2, 10, 6)

        # Training K-Means model
        kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)

        # Extract cluster centers (palette colors)
        palette_colors = kmeans.cluster_centers_

        # Predicting animation, call image prediction function
        with st.spinner("Predicting..."):
            prediction = predict_image(uploaded_file)

        # Display the prediction output
        st.markdown(f"<h3 style='text-align: center; color: white;'>Prediction: {prediction}</h3>", unsafe_allow_html=True)

        # Display the original image, color palette, and image prediction
        st.image(image, caption="Uploaded Image", use_column_width=True)
        display_color_palette(palette_colors, kmeans.labels_)

def resize_image(image):
    max_width = 800
    width_ratio = max_width / image.width
    new_width = int(image.width * width_ratio)
    new_height = int(image.height * width_ratio)
    return image.resize((new_width, new_height))

def display_color_palette(palette_colors, labels):
    st.write("Color Palette:")
    fig, ax = plt.subplots(1, len(palette_colors), figsize=(12, 2))

    for idx, color in enumerate(palette_colors):
        color_block = np.zeros((1, 1, 3), dtype=np.uint8)
        color_block[0, 0, :] = color
        ax[idx].imshow(color_block)
        ax[idx].axis('off')
        hex_color = "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
        ax[idx].text(0.5, -0.1, hex_color, color='black', fontsize=8, horizontalalignment='center', transform=ax[idx].transAxes)

    plt.subplots_adjust(wspace=0, hspace=0)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
