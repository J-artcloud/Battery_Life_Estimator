import streamlit as st 
import joblib
import numpy as np
import pandas as pd
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "model1.joblib"

# --- Load The Model and Scaler ---
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    #scaler = joblib.load('scaler.joblib')
    return model #,scaler

model = load_model()

st.set_page_config(
    page_title="Battery Life App",
    page_icon=" 🔋⚡🔋",
    layout="wide"
)


st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">

<style>

/* Global font */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* App background */
.stApp {
    background: linear-gradient(135deg,#020617,#0f172a);
    color:white;
}

/* Main Title */
.main-title{
    font-family: 'Orbitron', sans-serif;
    font-size:60px;
    font-weight:800;
    text-align:center;
    background: linear-gradient(90deg,#38bdf8,#22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom:5px;
}

/* Subtitle */
.subtitle{
    text-align:center;
    font-size:18px;
    color:#94a3b8;
    margin-bottom:35px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#020617,#020617,#0f172a);
    border-right:1px solid rgba(56,189,248,0.1);
}

/* Sidebar labels */
section[data-testid="stSidebar"] label{
    font-size:15px;
    font-weight:500;
    color:#38bdf8;
}

/* Inputs */
.stSelectbox div[data-baseweb="select"]{
    background:#020617;
    border-radius:10px;
    border:1px solid rgba(56,189,248,0.3);
}

/* Sliders */
.stSlider > div{
    color:#38bdf8;
}

/* Input container */
.stSelectbox, .stSlider{
    border-radius:10px;
    padding:5px;
}

/* Buttons */
.stButton>button{
    background:linear-gradient(90deg,#38bdf8,#22c55e);
    color:white;
    font-size:18px;
    font-weight:600;
    border:none;
    border-radius:12px;
    padding:12px 30px;
    transition:0.3s;
    box-shadow:0px 0px 12px rgba(56,189,248,0.4);
}

.stButton>button:hover{
    background:linear-gradient(90deg,#0ea5e9,#16a34a);
    transform:scale(1.05);
    box-shadow:0px 0px 20px rgba(34,197,94,0.6);
}

/* Prediction card */
.prediction-result{
    font-family: 'Orbitron', sans-serif;
    font-size:34px;
    text-align:center;
    background:linear-gradient(90deg,#22c55e,#4ade80);
    padding:22px;
    border-radius:14px;
    margin-top:25px;
    color:black;
    box-shadow:0px 0px 20px rgba(34,197,94,0.5);
}

/* Dataframe */
[data-testid="stDataFrame"]{
    border-radius:10px;
}

/* Battery container */
.battery-container{
    width:220px;
    height:85px;
    border:4px solid #38bdf8;
    border-radius:12px;
    position:relative;
    margin:auto;
    background:#020617;
}

/* Battery tip */
.battery-container::after{
    content:"";
    position:absolute;
    right:-14px;
    top:26px;
    width:12px;
    height:32px;
    background:#38bdf8;
    border-radius:3px;
}

/* Battery level */
.battery-level{
    height:100%;
    background:linear-gradient(90deg,#22c55e,#4ade80);
    width:0%;
    border-radius:8px;
    transition:width 1.8s ease-in-out;
    box-shadow:0px 0px 15px rgba(34,197,94,0.6);
}

/* Phone mockup */
.phone{
    width:230px;
    height:430px;
    border-radius:35px;
    border:3px solid #1e293b;
    background:#020617;
    margin:auto;
    padding:20px;
    box-shadow:0px 0px 30px rgba(56,189,248,0.25);
    position:relative;
}

/* Phone screen glow */
.phone::before{
    content:"";
    position:absolute;
    inset:0;
    border-radius:35px;
    box-shadow:inset 0px 0px 25px rgba(56,189,248,0.1);
}

/* Battery text */
.phone-battery{
    font-size:34px;
    text-align:center;
    margin-top:140px;
    font-family:'Orbitron',sans-serif;
    color:#22c55e;
}

/* Glass card effect */
.glass-card{
    background:rgba(255,255,255,0.05);
    border-radius:15px;
    padding:20px;
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)


st.markdown("""<h1 class="title">Battery life Estimator App 🔋 </h1>""",unsafe_allow_html=True)
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
    battery_percent = min(predicted_Battery_life * 10, 100)

    st.markdown(f"""
    <div class="battery-container">
        <div class="battery-level" style="width:{battery_percent}%"></div>
    </div>

    <div class="phone-battery">
    🔋 {predicted_Battery_life:.1f} hrs
    </div>
    """, unsafe_allow_html=True)
   
st.markdown("---")
