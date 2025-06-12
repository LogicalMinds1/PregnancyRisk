import os
# os.environ["STREAMLIT_WATCHDOG_USE_POLLING"] = "true" 
import streamlit as st
import base64
from qdrant_rag import initialize_system, process_csv_to_qdrant, rag_query
import pandas as pd
import numpy as np
import re
import joblib
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def ask_gpt(question, context, risk):
    try:
        messages = [
            {"role": "system", "content": f'''You are a helpful and knowledgeable pregnancy assistant.
            Patient Information: {context}
            Risk in pregnancy: {risk}
            Based on the above context, provide a concise and accurate answer to the following question'''},
            {"role": "user", "content": question}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Sorry, I couldn’t generate a proper response. Error:\n\n{str(e)}"

def get_explaination(context,risk):
    question = f"""
        I've provided you with the context of a patient.
        Based on the context, please provide a concise and meaningfull explanation for the patient to be at {risk} risk in their pregnancy.
    """
    return ask_gpt(question, context, risk=risk)
# Predefined login credentials (You can later replace these with a secure method, e.g., database)
USERNAME = "ragadmin"
PASSWORD = "ragadmin@321"

# Function to check credentials
def check_credentials(username, password):
    return username == USERNAME and password == PASSWORD

# Function to display login page
def login_page():
    st.title("Login Page")
    
    with st.form(key="login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        
        if login_button:
            if check_credentials(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.session_state.page = "main_page"  # Set page state to main page
                st.rerun()  # Force the app to rerun
            else:
                st.error("Invalid username or password")

avrage_mapping = {
    0: "low",
    1: "medium",
    2: "high"
}
output_mapping = {
    "low": 0,
    "medium": 1,
    "high": 2
}


@st.cache_resource
def cached_initialize_system_and_process():
    initialize_system()
    process_csv_to_qdrant("filtered_merged_output.csv")
    model = joblib.load("stats/xgb_model.pkl")
    le_y = joblib.load("stats/label_encoder_y.pkl")
    label_encoder = joblib.load("stats/label_encoders_features.pkl")
    return model, le_y, label_encoder


model, le_y, label_encoder = cached_initialize_system_and_process()


def classify_risk(input_dict: dict) -> str:
    """
    Predict risk level for a single input sample.
    Args:
        input_dict (dict): Dictionary with feature names as keys and user input as values.
    Returns:
        str: Predicted risk level.
    """
    # Convert to DataFrame with a single row
    df = pd.DataFrame([input_dict])

    # Fill and clean
    df['HEIGHT'] = df['HEIGHT'].fillna(160)
    df['AGE'] = df['AGE'].fillna(25)
    df['WEIGHT'] = df['WEIGHT'].fillna(60)
    df['BLOOD_GRP'] = df['BLOOD_GRP'].fillna('O+')
    df['GESTANTIONAL_DIA'] = df.get('GESTANTIONAL_DIA', 0)
    df['KNOWN_EPILEPTIC'] = df.get('KNOWN_EPILEPTIC', 0)
    df['CONVULSION_SEIZURES'] = df.get('CONVULSION_SEIZURES', 0)
    df['FOLIC_ACID'] = df.get('FOLIC_ACID', 0)

    # OGTT Handling
    avg_ogtt = (98 + 100 + 110) / 3
    df["OGTT_2_HOURS"] = pd.to_numeric(df.get("OGTT_2_HOURS", avg_ogtt), errors='coerce')
    df["OGTT_2_HOURS"] = df["OGTT_2_HOURS"].fillna(avg_ogtt)

    # Uterus size
    df["UTERUS_SIZE"] = pd.to_numeric(df.get("UTERUS_SIZE", np.nan), errors='coerce').fillna(70)

    # Normalize binary categorical fields
    binary_cols = [
        'URINE_SUGAR', 'URINE_ALBUMIN', 'THYROID', 'RH_NEGATIVE', 'HIV', 'HIV_RESULT',
        'IFA_TABLET', 'CALCIUM', 'PHQ_ACTION', 'GAD_ACTION',
        'ANC1FLG', 'ANC2FLG', 'ANC3FLG', 'ANC4FLG',
        'MISSANC1FLG', 'MISSANC2FLG', 'MISSANC3FLG', 'MISSANC4FLG',
        'IS_PREV_PREG', "HUSBAND_BLOOD_GROUP"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: (
                'No' if pd.isna(x) or str(x).strip().lower() in ['no', 'n', 'not give counselling', 'absent', 'non-reactive']
                else 'Yes' if str(x).strip().lower() == 'y'
                else x
            ))

    # Convert any numerics from string
    numeric_cols = ['BP', 'BP1', 'HEMOGLOBIN', 'HEART_RATE', 'BLOOD_SUGAR', 'FEVER',
                    'IFA_QUANTITY', 'NO_OF_WEEKS', 'PHQ_SCORE', 'GAD_SCORE']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())

    # Mode-based defaults
    if 'PULSE_RATE' in df.columns:
        df['PULSE_RATE'] = df['PULSE_RATE'].fillna(72)
    if 'RESPIRATORY_RATE' in df.columns:
        df['RESPIRATORY_RATE'] = df['RESPIRATORY_RATE'].fillna(16)

    # Use saved label encoders
    for col, le in label_encoder.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    # Align features with training model
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Ensure all columns are numeric types
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Final safety check
    assert all(np.issubdtype(dt, np.number) for dt in df.dtypes), "Non-numeric dtype found in model input."

    # Predict
    pred = model.predict(df)[0]
    return le_y.inverse_transform([pred])[0]


def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# === Validation Functions ===
def validate_date_format(date_str):
    return bool(re.match(r"^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/\\d{4}$", date_str))

def validate_pregnancy_logic(gravida, parity, abortions, live):
    return parity + abortions + live <= gravida


def main_page():

    # === User input form ===
    st.subheader("Enter Patient Details")
    classify_result = False
    prediction_start = False
    if st.session_state.get("use_chat", False):
        input_data = st.text_area("Paste or type patient's details here...")
        if st.button("🔍 Predict from Text"):
            prediction_start = True
    
    else:
        mandatory_fields = ['AGE', 'HEIGHT', 'WEIGHT', 'BLOOD_GRP', 'HUSBAND_BLOOD_GROUP', 'RH_NEGATIVE',
        'GRAVIDA', 'PARITY', 'ABORTIONS', 'PREVIOUS_ABORTION', 'LIVE', 'DEATH',
        'KNOWN_EPILEPTIC', 'TWIN_PREGNANCY', 'GESTANTIONAL_DIA', 'CONVULSION_SEIZURES',
        'BP', 'BP1', 'HEMOGLOBIN', 'PULSE_RATE', 'RESPIRATORY_RATE', 'HEART_RATE', 'FEVER']
        input_data = {}

        col1, col2, col3 = st.columns(3)
        if 'error' in st.session_state:
            st.error(st.session_state.error)
            del st.session_state.error

        with col1:
            age_years = st.number_input("Age (Years) :red[*]", min_value=10, max_value=60)

            input_data['HEIGHT'] = st.number_input("Height (cm) :red[*]", min_value=100, max_value=220) 
            input_data['GRAVIDA'] = st.selectbox("Gravida :red[*]", ['', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6'])
            input_data['LIVE'] = st.selectbox("Live Births :red[*]", ['', 'L0', 'L1', 'L2', 'L3', 'L4'])
            input_data['TWIN_PREGNANCY'] = st.selectbox("Twin Pregnancy :red[*]", ['', 'No', 'Yes'])
            input_data['BLOOD_SUGAR'] = st.number_input("Blood Sugar (e.g., 66, 80, 112)")
            input_data['ANC1FLG'] = st.selectbox("ANC1FLG", ['No', 'Yes'])
            input_data['ANC4FLG'] = st.selectbox("ANC4FLG", ['No', 'Yes'])  
            input_data['MISSANC3FLG'] = st.selectbox("MISSANC3FLG", ['No', 'Yes'])
            input_data['PLACE_OF_DELIVERY'] = st.selectbox("Place of Delivery",['Govt', 'Private', "Other State", "Other Govt", "Transit", "Home", "C-Section", "Live"])
            input_data['HIV'] = st.selectbox("HIV", ['No', 'Yes'])
            input_data['SCREENED_FOR_MENTAL_HEALTH'] = st.selectbox("Screened for Mental Health", ['No', 'Yes'])
            input_data['PHQ_ACTION'] = st.selectbox("PHQ Action", ['No', 'Give counselling', 'Psychiatrist for treatment'])
            input_data['IS_PREV_PREG'] = st.selectbox("Is Prev Pregante", ['No', 'Yes'])
            input_data['GESTANTIONAL_DIA'] = st.selectbox("Gestational Diabetes (0 or 1) :red[*]",['','No', 'Yes'])
            input_data['ODEMA_TYPE'] = st.selectbox("Odema Type", ['No', 'Pedal Oedema', 'Non-dependent oedema (Facial puffiness, abdominal oedema, vulval oedema)'])
            input_data['WARNING_SIGNS_SYMPTOMS_HTN'] = st.selectbox("Warning Signs Symptoms HTN", ['No', "Headache", "Vomitting", "Decreased urine output", "Blurring of vision", "Epigastric pain"])
            input_data['CONSANGUINITY'] = st.selectbox("Consanguinity", ['No', 'Yes'])
            input_data['HIV_RESULT'] = 'Positive' if input_data['HIV'] == 'Yes' else 'Negative'
            input_data['HEART_RATE'] = st.number_input("Heart Rate (e.g., 110) :red[*]", min_value=40, max_value=180)
            input_data['THYROID'] = st.selectbox("Thyroid", ['No', 'Yes'])
            input_data['IRON_SUCROSE_INJ'] = st.selectbox("Iron Sucrose Injection", ['No', 'Yes'])
            input_data['FEVER'] = st.number_input("Fever (e.g., 98) :red[*]", min_value=90, max_value=110)

        with col2:
            age_months = st.number_input("Age (Months)", min_value=0, max_value=11)
            input_data['AGE'] = age_years + age_months / 12
            input_data['PARITY'] = st.selectbox("Parity :red[*]", ['', 'P0', 'P1', 'P2', 'P3', 'P4'])
            input_data['DEATH'] = st.number_input("Infant Deaths :red[*]", min_value=0, step=1)
            input_data['BP1'] = st.number_input("BP1 (e.g., 60,70, 80) :red[*]", min_value=60, max_value=200)
            input_data['HEMOGLOBIN'] = st.number_input("Hemoglobin Level (g/dL) :red[*]", step=0.1, max_value=22.0, min_value=3.0)
            input_data['ANC2FLG'] = st.selectbox("ANC2FLG", ['No', 'Yes'])
            input_data['MISSANC1FLG'] = st.selectbox("MISSANC1FLG", ['No', 'Yes'])
            input_data['MISSANC4FLG'] = st.selectbox("MISSANC4FLG", ['No', 'Yes'])
            input_data['PULSE_RATE'] = st.number_input("Pulse Rate (BPM) :red[*]", min_value=40, max_value=180)
            input_data['IFA_QUANTITY'] = st.number_input("IFA Quantity", step=0.1)
            input_data['PHQ_SCORE'] = st.slider("PHQ Score", 0, 27)
            input_data['GAD_ACTION'] = st.selectbox("GAD Action", ['No', 'Give counselling', 'Psychiatrist for treatment'])
            input_data['PREVIOUS_ABORTION'] = st.selectbox("Previous Abortion :red[*]", ['', 'No', 'Yes'])
            input_data['CONVULSION_SEIZURES'] = st.selectbox("Convulsion Seizures :red[*]", ['', 'No', 'Yes'])
            input_data['HEP_RESULT'] = st.selectbox("HEP Result", ['No', 'Yes'])
            input_data['ANY_COMPLAINTS_BLEEDING_OR_ABNORMAL_DISCHARGE'] = st.selectbox("Complaints: Bleeding/Discharge", ['No', 'Yes'])
            input_data['UTERUS_SIZE'] = st.number_input("Uterus Size (e.g., 70)")
            input_data['URINE_SUGAR'] = st.selectbox('Urine Sugar', ['No', 'Yes'])
            input_data['RH_NEGATIVE'] = st.selectbox("RH Negative :red[*]", ['', 'No', 'Yes'])
            input_data['EXP_DOD'] = st.text_input("Expected Delivery Date (DD/MM/YYYY)")
            if input_data['EXP_DOD'] and not re.match(r"^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/\d{4}$", input_data['EXP_DOD']):
                st.error("Please use date format DD/MM/YYYY")

        with col3:
            input_data['WEIGHT'] = st.number_input("Weight (kg) :red[*]", min_value=30, max_value=150)
            input_data['ABORTIONS'] = st.selectbox("Abortions :red[*]", ['', 'A0', 'A1', 'A2', 'A3'])
            
            blood_names = ["A Positive", "A Negative", "B Positive", "B Negative", "AB Positive", "AB Negative", "O Positive", "O Negative"]
            blood_name = st.selectbox("Blood Group Name :red[*]", blood_names)
            husband_blood_name = st.selectbox("Husband Blood Group Name :red[*]", blood_names)
            blood_code = blood_names.index(blood_name) + 1
            input_data['BLOOD_GRP'] = blood_code
            input_data['HUSBAND_BLOOD_GROUP'] =  blood_names.index(husband_blood_name) + 1
            input_data['BLOOD_GRP_NAME'] = blood_name

            input_data['BP'] = st.number_input("BP (e.g., 110, 112) :red[*]", min_value=60, max_value=200)
            input_data['IFA_TABLET'] = st.selectbox("IFA Tablet", ['No', 'Yes'])
            input_data['ANC3FLG'] = st.selectbox("ANC3FLG", ['No', 'Yes'])
            input_data['MISSANC2FLG'] = st.selectbox("MISSANC2FLG", ['No', 'Yes'])
            input_data['DELIVERY_MODE'] = st.selectbox("Delivery Mode", ['Normal', 'C-Section'])
            input_data['RESPIRATORY_RATE'] = st.number_input("Respiratory Rate :red[*]", min_value=10, max_value=40)
            input_data['CALCIUM'] = st.selectbox("Calcium", ['No', 'Yes'])  
            input_data['GAD_SCORE'] = st.slider("GAD Score", 0, 21)
            input_data['NO_OF_WEEKS'] = st.number_input("No of Weeks (e.g., 12, 35)")  
            input_data['KNOWN_EPILEPTIC'] = st.selectbox("Known Epileptic :red[*]", ['', 'No', 'Yes'])
            input_data['ODEMA'] = st.selectbox("Odema", ['No', 'Yes'])
            input_data['OGTT_2_HOURS'] = st.number_input("OGTT2 Hours (e.g., 98, 100, 110)")  
            input_data['FOLIC_ACID'] = st.number_input("Folic Acid (e.g., 0)")      
            input_data['URINE_ALBUMIN'] = st.selectbox('Urine Albumin', ['No', 'Yes'])
            input_data['SYPHYLIS'] = st.selectbox("Syphilis", ['No', 'Yes'])  
            input_data['ANC_DATE'] = st.text_input("ANC Date (e.g., 5/2/2017)")

        # Optional: add logic check
        if input_data['GRAVIDA'] and input_data['PARITY'] and input_data['ABORTIONS'] and input_data['LIVE']:
            if not validate_pregnancy_logic(input_data['GRAVIDA'].replace("G", ""),
                                            input_data['PARITY'].replace("P", ""),
                                            input_data['ABORTIONS'].replace("A", ""),
                                            input_data['LIVE'].replace("L", "")):
                st.error("Sum of Parity, Abortions, and Live Births must not exceed Gravida.")
        
        # === Predict Button ===
        if st.button("Predict Risk Level"):
            empty_fields = []
            for field in mandatory_fields:
                if input_data[field] == '' or input_data[field] is None:
                    empty_fields.append(field)
            if empty_fields:
                st.session_state.error = f"Please fill in the mandatory fields: {empty_fields}"
                st.rerun()
            prediction_start = True
            classify_result = True
    
    if prediction_start:

        if 'model_data' in st.session_state:
            del st.session_state.model_data
        if 'error' in st.session_state:
            del st.session_state.error
        if 'chat_messages' in st.session_state:
            del st.session_state.chat_messages
    
        # Convert input to DataFrame
        if classify_result:
            classify_result = classify_risk(input_data)
            classify_result_clean = classify_result.strip().replace("_", " ").title().lower()
            print(f"Classified Risk Level: {classify_result_clean}")
            
        query_string = f"{input_data}"
        # print(query_string)
        with st.spinner("Fetching Results from RAG system... Please wait."):
            response = rag_query(query_string)
        if "</think>" in response:
            response = response.strip().split("</think>")[1]
        response = response.strip()

        risk_level_clean = response.lower()
        print(f"RAG Risk Level: {response}")
        final_output = risk_level_clean
        if classify_result:
            final_output = avrage_mapping.get(round(float(output_mapping.get(risk_level_clean, 1)+output_mapping.get(classify_result_clean,1))/2), "medium")
            print(f"Final Output Risk Level: {final_output}")
            
        print(f"==============================================================")
        with st.spinner("Fetching The explaination For the risk classification.... Please wait."):
            explanation = get_explaination(query_string, final_output)
            # print(f"Explanation: {explanation}")
        
        nutrition_que = f'''
            I've provided you with the context of a patient, and the risk level of that patient's pregnancy is {final_output}.
            Based on the context, please provide a concise and meaningful nutrition plan for that patient in approx 6 points and in well formated way with headings, subpoints and appropriate icons.
            '''
        with st.spinner("Fetching The Nutrition details for the patient.... Please wait."):
            nutrition_tips = ask_gpt(nutrition_que, context=query_string, risk=final_output)
        
        st.session_state.model_data = {
            "query_string": query_string,"classification": final_output,
            "nutrition_tips": nutrition_tips, "explations": explanation,}
        st.session_state.ai_chat = True
        st.rerun()
        
    if 'model_data' in st.session_state:
        final_output = st.session_state.model_data.get("classification", "medium")
        explanation = st.session_state.model_data.get("explations", "No explanation available.")
        nutrition_tips = st.session_state.model_data.get("nutrition_tips", "No nutrition tips available.")
        display_results(final_output, explanation, nutrition_tips)
    
    # Chat Interface Always Visible
    if st.session_state.ai_chat:
        chat_interface()
        
def display_results(final_output, explanation, nutrition_tips):
    display_label = final_output.capitalize() + " Risk"
    st.success(f"Predicted Pregnancy Risk: **{display_label}**")
    risk_position = {
        "low": "10%",
        "medium": "50%",
        "high": "90%"
    }

    arrow_position = risk_position.get(final_output.lower(), "50%")
    risk_meter_html = f"""
    <div style="margin-top: 30px;">
        <h4 style="text-align: center;">Pregnancy Risk Meter</h4>
        <div style="position: relative; background: linear-gradient(to right, #4caf50, #ffeb3b, #f44336);
                    height: 30px; border-radius: 15px; margin: 10px 40px;">
            <div style="position: absolute; left: {arrow_position}; top: -10px; transform: translateX(-50%);">
                <span style="font-size: 30px;">⬇️</span>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin: 0 40px;">
            <span style="color: #4caf50;"><b>Low</b></span>
            <span style="color: #ff9800;"><b>Medium</b></span>
            <span style="color: #f44336;"><b>High</b></span>
        </div>
    </div>
    """
    st.markdown(risk_meter_html, unsafe_allow_html=True)

    st.markdown("### Explanation")
    st.info(explanation)
    # === Chat Button ===
    st.title("Nutrition tips from AI Assistant")
    st.write(f"{nutrition_tips}")

# Function to display the chat interface
def chat_interface():
    st.title("🤖 Chat with AI Pregnancy Assistant")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "system", "content": "You are a pregnancy assistant."}]

    for msg in st.session_state.chat_messages:
        if msg["role"] != "system":  # Normalize role to lowercase
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask your question...")

    if user_input:
        # Append user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)


        if "model_data" in st.session_state:
            context = st.session_state.model_data.get("query_string", "")
            risk = st.session_state.model_data.get("classification", "Medium")
            # Get GPT response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = ask_gpt(user_input, context, risk=risk)
                    st.markdown(response)

            # Append assistant message
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
        else:
            st.markdown("No context available. Please fill the form first and run the prediction.")



st.sidebar.markdown("###")
st.sidebar.markdown("###")
st.sidebar.markdown("###")

image_base64 = get_image_base64("logo.png")
image_html = f"""
<div style="display: flex; justify-content: center;">
    <img src="data:image/png;base64,{image_base64}" width="100">
</div>
"""
st.sidebar.markdown("###")
st.sidebar.markdown("###")


st.sidebar.markdown(image_html, unsafe_allow_html=True)

st.sidebar.header("""This app predicts pregnancy risk level based on maternal and pregnancy-related data.""")

if "use_chat" not in st.session_state:
    st.session_state.use_chat = False



st.title("Pregnancy Risk Classification")

# Authentication Flow
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if 'ai_chat' not in st.session_state:
    st.session_state.ai_chat = False

if not st.session_state.logged_in:
    login_page()
else:
    if not st.session_state.use_chat and st.sidebar.button("💬 Use Chat Input Instead"):
        st.session_state.use_chat = True
        st.session_state.chat_input_mode = True
        if 'model_data' in st.session_state:
            del st.session_state.model_data
        st.rerun()
    if st.session_state.use_chat and st.sidebar.button("📝 Back to Form Input"):
        st.session_state.use_chat = False
        if 'model_data' in st.session_state:
            del st.session_state.model_data
        st.rerun()
    main_page()