# import streamlit as st
# import tensorflow as tf
# import numpy as np

# # Assume 'class_names' is a list of class names in the order that your model outputs probabilities
# # For the binary dataset, it might be something like:
# binary_class_names = ["Pizza", "Steak"]
# # For the 10-class dataset, you mentioned the names already:
# multi_class_names = ["Chicken Curry", "Chicken Wings", "Fried Rice", "Grilled Salmon", "Hamburger", "Ice Cream", "Pizza", "Ramen", "Steak", "Sushi"]

# st.set_page_config(layout="wide")

# # Function to load and prepare the image
# def load_and_prep_image(image, img_shape=224):
#     # Process the image to model's expected format
#     img = tf.image.decode_image(image.read(), channels=3)
#     img = tf.image.resize(img, [img_shape, img_shape])
#     img = img / 255.
#     return img


# # Placeholder for model loading function
# def loading_model(model_name):
#     # Load and return the model (to be implemented)
#     model = tf.keras.models.load_model(f'Models/{model_name}')
#     return model


# # Using columns to layout the foods and the main application
# col1, col_main, col3 = st.columns([1, 2, 1])

# # Displaying the food types in the first column
# with col1:
#     st.header("Binary Dataset")
#     st.write("This dataset contains images of two food types:")
#     st.write("- Pizza")
#     st.write("- Steak")

# # Main functionality in the middle column
# with col_main:
#     st.title("Food Classification App")

#     # Dataset selection
#     dataset = st.selectbox("Choose the dataset:", ["Select", "Pizza/Steak (Binary)", "10 Food Types (Multi-class)"],
#                            key="dataset")

#     # Model selection based on dataset
#     if dataset != "Select":
#         model_choice = st.selectbox("Choose the model:", ["Model 1", "Model 2", "Model 3", "Model 4",
#                                                           "Model 5"] if dataset == "Pizza/Steak (Binary)" else [
#             "Model 6", "Model 7", "Model 8"], key="model_choice")
#         # Image upload
#         image = st.file_uploader("Upload an image of food", type=["jpg", "jpeg", "png"], key="image_uploader",)

#         # When the user uploads an image and selects a model
#         if image is not None and model_choice is not None:
#             # Display the uploaded image
#             st.image(image, caption="Uploaded Image")

#             # Load and prepare the image
#             prepared_image = load_and_prep_image(image)
#             prepared_image = tf.expand_dims(prepared_image, axis=0)  # Add batch dimension

#             # Load the selected model (ensure this function is properly implemented)
#             model = loading_model(model_choice)

#             # Make a prediction
#             pred = model.predict(prepared_image)

#             # Determine the class label with the highest probability
#             if dataset == "Pizza/Steak (Binary)":
#                 # For binary classification, you might only get one output neuron
#                 # In that case, you can check if the value is closer to 0 or 1
#                 pred_class = binary_class_names[int(tf.round(pred)[0][0])]
#             else:
#                 # For multi-class classification, find the index of the max probability
#                 pred_class_index = np.argmax(pred, axis=1)[0]  # Assuming your model outputs a batch
#                 pred_class = multi_class_names[pred_class_index]

#             # Display the prediction
#             st.write(f"Prediction: {pred_class}")

# # Displaying the food types in the third column
# with col3:
#     st.header("10 Food Types")
#     st.write("This dataset contains images of ten different food types:")
#     st.write("- Chicken Curry")
#     st.write("- Chicken Wings")
#     st.write("- Fried Rice")
#     st.write("- Grilled Salmon")
#     st.write("- Hamburger")
#     st.write("- Ice Cream")
#     st.write("- Pizza")
#     st.write("- Ramen")
#     st.write("- Steak")
#     st.write("- Sushi")

import streamlit as st
import tensorflow as tf
import numpy as np

# Assume 'class_names' is a list of class names in the order that your model outputs probabilities
binary_class_names = ["Pizza", "Steak"]
multi_class_names = ["Chicken Curry", "Chicken Wings", "Fried Rice", "Grilled Salmon", "Hamburger", "Ice Cream", "Pizza", "Ramen", "Steak", "Sushi"]

st.set_page_config(layout="wide", page_title="Food Classification App", page_icon="🍕")

# Function to load and prepare the image
def load_and_prep_image(image, img_shape=224):
    img = tf.image.decode_image(image.read(), channels=3)
    img = tf.image.resize(img, [img_shape, img_shape])
    img = img / 255.
    return img

# Placeholder for model loading function
def loading_model(model_name):
    model = tf.keras.models.load_model(f'Models/{model_name}')
    return model

# Sidebar for dataset and model selection
with st.sidebar:
    st.title("Food Classification App")
    st.write("Select the dataset and model, then upload an image to classify.")

    dataset = st.selectbox("Choose the dataset:", ["Select", "Pizza/Steak (Binary)", "10 Food Types (Multi-class)"])
    model_choice = st.selectbox("Choose the model:", ["Select"] + ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"] if dataset == "Pizza/Steak (Binary)" else ["Model 6", "Model 7", "Model 8"])
    image = st.file_uploader("Upload an image of food", type=["jpg", "jpeg", "png"])

    st.markdown("---")

# Main content
if image is not None and dataset != "Select" and model_choice != "Select":
    st.title("Food Classification Results")
    
    # Display the uploaded image with a fade-in effect
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load and prepare the image
    prepared_image = load_and_prep_image(image)
    prepared_image = tf.expand_dims(prepared_image, axis=0)  # Add batch dimension

    # Load the selected model
    model = loading_model(model_choice)

    # Make a prediction
    pred = model.predict(prepared_image)

    # Determine the class label with the highest probability
    if dataset == "Pizza/Steak (Binary)":
        pred_class = binary_class_names[int(tf.round(pred)[0][0])]
    else:
        pred_class_index = np.argmax(pred, axis=1)[0]
        pred_class = multi_class_names[pred_class_index]

    # Display the prediction with an animated effect
    st.subheader("Prediction:")
    st.write(f"**{pred_class}**")
    
    # Add a loading spinner
    with st.spinner("Processing..."):
        st.balloons()

# Display food types in a collapsible section
with st.expander("Food Types"):
    if dataset == "Pizza/Steak (Binary)":
        st.write("This dataset contains images of two food types:")
        st.write("- Pizza")
        st.write("- Steak")
    else:
        st.write("This dataset contains images of ten different food types:")
        st.write("- Chicken Curry")
        st.write("- Chicken Wings")
        st.write("- Fried Rice")
        st.write("- Grilled Salmon")
        st.write("- Hamburger")
        st.write("- Ice Cream")
        st.write("- Pizza")
        st.write("- Ramen")
        st.write("- Steak")
        st.write("- Sushi")

# Add custom CSS for dark background and styling
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #e0e0e0;
        font-family: 'Arial', sans-serif;
    }
    .css-1v3fvcr {
        background-color: #2d2d2d;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        padding: 20px;
    }
    .stImage {
        border: 5px solid #444;
        border-radius: 10px;
    }
    .stSpinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    .stExpander {
        border: 1px solid #444;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Add footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("Made with ❤️ by [Your Name]", unsafe_allow_html=True)



