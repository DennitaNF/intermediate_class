import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model("amcc.keras")
    return model

def preprocessing_image(image):
    target_size = (64, 64)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype('float32') /255.0
    return image_array

def predict(model, image):
    return model.predict(image, batch_size=1)

def interpret_predict(prediction):
    if prediction.shape[-1] == 1:
        score = prediction[0][0]
        prediction_class = 0 if score >= 0.5 else 1
        confidence_score = [score, 1-score, 0]
    else:
        confidence_score = prediction[0]
        prediction_class = np.argmax(confidence_score)
    return prediction_class, confidence_score

def main():
    st.set_page_config(
        page_title = "Pet Classifier",
        layout = 'centered'
    )

    st.title("Classifier for Cats and Dogs")

    try:
        model = load_model()
        st.sidebar.write("<h1>About</h1>", unsafe_allow_html=True)
        st.sidebar.write("Classify your image trough this web")
        st.sidebar.write("model output shape", model.output_shape)
        st.sidebar.markdown("### Important Points:")
        st.sidebar.markdown("""
        - 🐶: Can classify cat or dog.
        - 🐱: The image must be png / jpg / jpeg.
        - 😇: Can show the confidence score for the result.
        """)
    except Exception as err:
        st.error(f"error: {str(err)}")
        return
    
    uploader = st.file_uploader("Select the image for classification:", type=['jpg', 'jpeg', 'png'])

    if uploader is not None:
        try:
            col1, col2 = st.columns([2,1])
            with col1:
                image = Image.open(uploader)
                st.image(image, caption="Your image", use_column_width=True)

            with col2:
                if st.button("Classify", use_container_width=True):
                    with st.spinner('loading..'):
                        processed_image = preprocessing_image(image)
                        prediction = predict(model, processed_image)
                        predicted_class, confidence_score = interpret_predict(prediction)
                        class_names = ['Cat', 'Dog']
                        result = class_names[predicted_class]
                        st.success(f"Prediction : {result}")

                        st.write("<h4>Prediction Score:</h4>", unsafe_allow_html=True)

                        for i, class_name in enumerate(class_names):
                            st.write(f"{class_name}: {confidence_score[i] * 100:.2f}%")
                            st.progress((float(confidence_score[i])), text=None)


        except Exception as err:
            st.error(f"error : {str(err)}")
            st.write("Choose the right file!")
            st.write(f"The error : {str(err)}")

if __name__ == "__main__":
    main()