import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        st.success("✅ Registration successful! You can now login.")
    except sqlite3.IntegrityError:
        st.error("❌ Username already exists!")
    conn.close()

def check_login(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

# Streamlit UI
st.title("Student Risk Prediction System")

if not st.session_state["logged_in"]:
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("🔑 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_login(username, password):
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials!")
    
    with tab2:
        st.subheader("📝 Register")
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            register_user(new_user, new_pass)
else:
    tab1, tab2 = st.tabs(["📂 Bulk Prediction", "🧑‍🎓 Individual Prediction"])
    
    with tab1:
        st.subheader("📊 Bulk Risk Prediction System")
        uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            required_columns = {'roll_number', 'study_hours', 'engagement_score', 'assignment_completion', 'mental_health_score', 'risk_category'}
            if not required_columns.issubset(df.columns):
                st.error("❌ Dataset must contain required columns!")
            else:
                st.success("✅ Dataset loaded successfully!")
                st.dataframe(df.head())
                
                scaler = StandardScaler()
                features = ['study_hours', 'engagement_score', 'assignment_completion', 'mental_health_score']
                X = df[features]
                X_scaled = scaler.fit_transform(X)
                joblib.dump(scaler, 'scaler.pkl')
                
                model = joblib.load('student_risk_model.pkl')
                y_pred = model.predict(X_scaled)
                
                risk_mapping = {0: "Low", 1: "Medium", 2: "High"}
                df['predicted_risk_category'] = [risk_mapping[p] for p in y_pred]
                
                # Confusion Matrix
                y_true = df['risk_category'].map({"Low": 0, "Medium": 1, "High": 2})
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"]).plot(ax=ax)
                st.pyplot(fig)
                
                roll_numbers = ["Predict All"] + sorted(df['roll_number'].astype(str).unique().tolist())
                selected_roll = st.selectbox("Select Roll Number", roll_numbers)
                
                if st.button("Predict"):
                    if selected_roll == "Predict All":
                        category_counts = df['predicted_risk_category'].value_counts()
                        st.write("Prediction Distribution:", category_counts.to_dict())
                        
                        fig, ax = plt.subplots()
                        category_counts.plot(kind='bar', color=['blue', 'red', 'green'], ax=ax)
                        ax.set_title("Risk Category Distribution")
                        ax.set_xlabel("Risk Category")
                        ax.set_ylabel("Count")
                        st.pyplot(fig)
                    else:
                        student_data = df[df['roll_number'].astype(str) == selected_roll]
                        if student_data.empty:
                            st.error("❌ Roll number not found!")
                        else:
                            risk_category = student_data['predicted_risk_category'].values[0]
                            st.write(f"Prediction for {selected_roll}: {risk_category}")
                            
                            feedback = {
                                "Low": [
                                    "Great job! Maintain your study habits.",
                                    "Keep engaging actively in studies.",
                                    "Your assignment completion is on track!",
                                    "Stay consistent in your study hours.",
                                    "Maintain a balanced mental health routine."
                                ],
                                "Medium": [
                                    "You're doing well, but some improvements needed.",
                                    "Increase study hours to reduce risk.",
                                    "Try to participate more in discussions.",
                                    "Ensure all assignments are completed on time.",
                                    "Seek help if you're feeling stressed."
                                ],
                                "High": [
                                    "You may be at risk, take action soon.",
                                    "Increase your study hours for better performance.",
                                    "Engage more with peers and instructors.",
                                    "Prioritize assignment completion.",
                                    "Consider reaching out for academic or mental health support."
                                ]
                            }
                            st.write("### Feedback:")
                            for point in feedback.get(risk_category, []):
                                st.write(f"- {point}")
    
    with tab2:
        st.subheader("🧑‍🎓 Individual Risk Prediction")
        study_hours = st.number_input("Study Hours", min_value=0.0, step=0.5)
        engagement_score = st.number_input("Engagement Score", min_value=0.0, step=0.5)
        assignment_completion = st.number_input("Assignment Completion", min_value=0.0, step=0.5)
        mental_health_score = st.number_input("Mental Health Score", min_value=0.0, step=0.5)

        if st.button("Predict Individual Risk"):
            model = joblib.load('student_risk_model.pkl')
            scaler = joblib.load('scaler.pkl')

            input_data = np.array([[study_hours, engagement_score, assignment_completion, mental_health_score]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            risk_mapping = {0: "Low", 1: "Medium", 2: "High"}
            predicted_risk = risk_mapping.get(prediction, "Unknown")

            st.write(f"### Predicted Risk Category: {predicted_risk}")

            # Feedback messages
            feedback = {
                "Low": [
                    "✅ Keep up the great work! Maintain your study routine.",
                    "📚 Stay engaged with your coursework.",
                    "📝 Ensure your assignments are completed on time.",
                    "💡 Continue practicing self-care for mental well-being."
                ],
                "Medium": [
                    "🔄 Consider increasing your study hours for improvement.",
                    "📖 Engage more actively in discussions and study groups.",
                    "⚠ Pay close attention to assignment deadlines.",
                    "💬 If you're feeling stressed, talk to a mentor or counselor."
                ],
                "High": [
                    "🚨 Immediate action needed! Prioritize your studies.",
                    "🆘 Seek support from teachers or academic advisors.",
                    "📅 Create a structured study plan for better time management.",
                    "🧘‍♂ Consider mental health resources to manage stress effectively."
                ]
            }

            st.write("### Feedback:")
            for point in feedback.get(predicted_risk, []):
                st.write(f"- {point}")
