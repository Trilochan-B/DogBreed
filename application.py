import streamlit as st
import joblib as jb
import numpy as np
import cv2
from PIL import Image

from src.predict.modelPrediction import prediction

model = prediction()


st.set_page_config(
    page_title='DogBreed',
    page_icon='🐶',
)

st.title('Breed Predection 🐶')
st.subheader("Upload your image file to predict")
file = st.file_uploader(label='dog',type=['jpg','png'], label_visibility='hidden')
st.write('''* Disclaimer : This model is trained on only 10 breeds i.e 
         Beagle,Bulldog,Boxer,Dachshund,German shepherd,Golden retriver,
         Labrador retriver,Poodle,Rottweiler,yorkshire terrier''')

col1, col2 = st.columns(2)

with col1:
    if file != None :
        image = np.array(Image.open(file))
        name = model.predict(image)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        st.title(f'Predicted : {name}')
        with col2:
            st.write("Has it predicted right ?")
            y = st.button('Yes')
            n = st.button('No')
            st.write('Thanks for your feedback !')
            
               




