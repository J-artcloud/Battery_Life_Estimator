import streamlit as st 
import joblib
import numpy as np
import pandas as pd
import time

# --- Load The Model and Scaler ---
@st.cache_resource
def load_model():
    model = joblib.load('.\models\model1.joblib')
    #scaler = joblib.load('scaler.joblib')
    return model #,scaler

model = load_model()

st.set_page_config(
    page_title="Battery Life App",
    page_icon=" 🔋⚡🔋",
    layout="wide"
)


st.title('Battery life Estimator App 🔋')
st.header("Analyze how long your phone can stay on before going off")
st.write(
    """
    This app predicts **how long a phone can stay on before going off** based on  differnt features of your phones.
    Enter the details of your phone, and the model will estimate the how long your phone stays on.
    This is based on a Linear Regression model trained on the phone_battery_dataset.
    """
)
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("Ready to know your battery life...Let's dive in !")
st.sidebar.write("Give the differennt features of your phone ")
st.sidebar.header("Input Features")

def user_input_features():
    Brand = st.sidebar.selectbox('phone mark', ('Motorola','Noxia','Xiaomi','Google','Oppo','OnePlus','Sony','Samsung','Apple','Huawei'))
    Screen_Size_in = st.sidebar.slider('Screen_size', 4.0, 8.0, 4.5)
    RAM_GB = st.sidebar.slider('RAM size in GB', 2.0, 15.0 ,2.0)
    Battery_mAh = st.sidebar.slider('Battery Capacity', 2000.0,7000.0 , 2000.0)
    Processor_GHz = st.sidebar.slider('processor speed in GHz', 2.0, 4.0, 1.5)
    Num_Cores = st.sidebar.slider('Number of cores(processor chips)',  1.0, 4.0, 1.5 )
    Has_5G = st.sidebar.selectbox('Is phone 5G or not (0: No , 1: Yes) ', (0,1))
    Weight_g = st.sidebar.slider('Phone\'s weigh in grams', 100.0, 230.0, 150.0)
    Price_USD = st.sidebar.slider('phone\'s price in USD ', 200.0, 900.0, 450.0)
    OS_number = st.sidebar.selectbox('select your phone\'s operating system (0:Android , 1: iOS) ', (0,1)),
    Res_FHD = st.sidebar.selectbox('Is the Resolution of the screen FHD ?  (0: No , 1: Yes) ', (0,1)),
    Res_HD = st.sidebar.selectbox('Is the Resolution of the screen HD ? (0: No , 1: Yes) ', (0,1)),
    Res_QHD = st.sidebar.selectbox('Is the Resolution of the screen QHD ?(0: No , 1: Yes) ', (0,1))
    
    data = {
         'Screen_Size_in': Screen_Size_in,'RAM_GB': RAM_GB, 'Battery_mAh': Battery_mAh ,
        'Processor_GHz': Processor_GHz, 'Num_Cores':  Num_Cores ,
        'Has_5G': Has_5G, 'Weight_g': Weight_g, 'Price_USD': Price_USD, 'OS_number': OS_number, 
        'Res_FHD' : Res_FHD, 'Res_HD': Res_HD, 
        'Res_QHD' : Res_QHD
    }

    return pd.DataFrame(data, index=[0])


input_df = user_input_features()

# --- Main Panel ---
st.header("Your Input")
st.dataframe(input_df)

if st.sidebar.button("Predict Battery Life⚡ "):
    with st.spinner("Calculating Battery Life prediction.⚡..⚡..⚡.."):
        time.sleep(2.3)  # Simulate processing delay
        #scaled_input = fit.transform(input_df)
        prediction = model.predict(input_df)
        predicted_Battery_life = prediction[0] 
    
    st.markdown(f"<div class='prediction-result'>Predicted Battery Life: {predicted_Battery_life:,.2f}hrs</div>", unsafe_allow_html=True)


st.markdown("---")
