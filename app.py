# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import base64
# import shap

# scaler = pickle.load(open('scaler.pkl', 'rb'))

# st.title("Heart Disease Predictor")
# tab1,tab2,tab3 = st.tabs(['Predict','Bulk Predict', 'Model Information'])

# with tab1:
#     age = st.text_input("Age (years)", placeholder="Type age...")
#     sex = st.selectbox("Sex", ["Select Sex","Male", "Female"])
#     chest_pain = st.selectbox("Chest Pain Type", ["Select chest pain type","Atypical Angina(ATA)", "Non-Anginal Pain(NAP)", "Asymptotic(ASY)", "Typical Angina(TA)"])
#     resting_bp = st.text_input("Resting Blood Pressure (mm Hg)",placeholder="Type resting bp...")
#     cholesterol = st.text_input("Cholesterol (mm/dl)", placeholder="Type cholesteriol...")
#     fasting_bs = st.text_input("Fasting Blood Sugar", placeholder="Type fasting bs...")
#     resting_ecg = st.selectbox("Resting ECG Result", ["Select resting ECG","Normal", "ST-T wave Abnormality(ST)", "Left Ventricular Hypertrophy(LVH)"])
#     max_hr = st.text_input("Maximum Heart Rate Achieved", placeholder="Type max hr...")
#     exercise_angina =  st.selectbox("Exercise-Induced Angina", ["Select exercise angina","Yes", "No"])
#     oldpeak = st.text_input("Oldpeak Value", placeholder="Type oldpeak value...")
#     st_slope = st.selectbox("Slope of peak Exercise ST Segment ", ["Select st slope","Up", "Flat", "Down"])
#     father_history = st.selectbox("Father Heart Disease History", ["Select", "Yes", "No"])
#     grandfather_history = st.selectbox("Grandfather Heart Disease History", ["Select", "Yes", "No"])

#     #convert categorical inputs to numeric

#     sex = 0 if sex == "Male" else 1  
#     age = int(age) if age else 0
#     resting_bp = int(resting_bp) if resting_bp else 0
#     cholesterol = int(cholesterol) if cholesterol else 0
#     max_hr = int(max_hr) if max_hr else 0
#     oldpeak = float(oldpeak) if oldpeak else 0.0
#     chest_pain = ["Atypical Angina(ATA)", "Non-Anginal Pain(NAP)", "Asymptotic(ASY)", "Typical Angina(TA)"].index(chest_pain)
#     # convert text to number
    
#     fasting_bs_input = int(fasting_bs) if fasting_bs else 0

#     # convert to model format
    
#     fasting_bs = 0 if fasting_bs_input <= 120 else 1
#     resting_ecg = ["Normal", "ST-T wave Abnormality(ST)", "Left Ventricular Hypertrophy(LVH)"].index(resting_ecg)
#     exercise_angina = 1 if exercise_angina == "Yes" else 0
#     st_slope = ["Up", "Flat", "Down"].index(st_slope)
#     if father_history == "Select" or grandfather_history == "Select": family_history = None
#     else: father_history = 1 if father_history == "Yes" else 0
#     grandfather_history = 1 if grandfather_history == "Yes" else 0
    
#     # create combined feature
#     family_history = 1 if (father_history == 1 or grandfather_history == 1) else 0


#     #create a Dataframe with user Inputs

#     input_data = pd.DataFrame({
#         'Age' : [age],
#         'Sex' : [sex],
#         'ChestPainType' : [chest_pain],
#         'RestingBP' : [resting_bp],
#         'Cholesterol' : [cholesterol],
#         'FastingBS' : [fasting_bs],
#         'RestingECG' : [resting_ecg], 
#         'MaxHR' : [max_hr],
#         'ExerciseAngina' : [exercise_angina],
#         'Oldpeak' : [oldpeak],
#         'ST_Slope': [st_slope],
#         'FatherHistory' : [father_history],
#         'GrandfatherHistory' : [grandfather_history],
#         'FamilyHistory' : [family_history]
#     }) 

#     algonames = ['Logistic Regression', 'Support Vector Machine', 'Decision Trees', 'Random Forest']
#     modelnames = ['logisticR.pkl', 'svm.pkl', 'decisionT.pkl', 'randomF.pkl'] 

#     input_scaled = scaler.transform(input_data)

#     def predict_heart_disease(data):
#         results = []

#         for modelname in modelnames:
#             model = pickle.load(open(modelname, 'rb'))

#             prediction = model.predict(data)[0]

#             # check if model supports probability
            
#             if hasattr(model, "predict_proba"):
#                 prob = model.predict_proba(data)[0][1]
#             else:
#                 prob = None

#             results.append((prediction,prob))
#         return results
    
#     #create submit button


#     _, _, _, col, _, _, _ = st.columns([1,1,1,2,1,1,1])
#     with col:
#         submit = st.button("Submit")
    
#     if submit:
        
#         st.subheader('Results......')
#         st.markdown('------------------')

#         input_scaled = scaler.transform(input_data)

#         result = predict_heart_disease(input_scaled)        

#         for i in range(len(result)): 
#             st.subheader(algonames[i]) 

#             if result[i][0] == 0: 
#                 st.write("No Heart Disease Detected")
#             else:
#                 st.write("Heart Disease Detected") 
                
#             st.markdown('--------------------------')
    

    

#     st.subheader("Detailed Results (With Probability)")

# input_scaled = scaler.transform(input_data)

# for i in range(len(modelnames)):
#     model = pickle.load(open(modelnames[i], 'rb'))
    
#     pred = model.predict(input_scaled)[0]

#     # # probability (if available)
#     # if hasattr(model, "predict_proba"):
#     #     prob = model.predict_proba(input_scaled)[0][1]
#     # else:
#     #     prob = None

#     # st.write(f"Model: {algonames[i]}")

#     # Colored result
#     if pred == 0:
#         st.success(" No Heart Disease Detected")
#     else:
#         st.error("Heart Disease Detected")

#     # # Probability display
#     # if prob is not None:
#     #     st.info(f"Probability of Heart Disease: {prob:.2f}")

#     st.markdown('--------------------------') 



#         # ================= SHAP EXPLANATION =================
#     st.subheader("Why this Prediction?")

# # Load model (Random Forest)
#     model = pickle.load(open('randomF.pkl', 'rb'))

# # Prediction for this model
#     rf_pred = model.predict(input_scaled)[0]

# # Create SHAP explainer
#     explainer = shap.Explainer(model)
#     shap_values = explainer(input_scaled)

# # Handle SHAP output correctly
#     if len(shap_values.values.shape) == 3:
#         impacts = shap_values.values[0][:, 1]   # class 1 (Heart Disease)
#     else:
#         impacts = shap_values.values[0]

#     impacts = impacts.tolist()

# # Feature names and values
#     feature_names = input_data.columns
#     values = input_data.iloc[0].values

# # Combine and sort
#     explanations = list(zip(feature_names, values, impacts))
#     explanations = sorted(explanations, key=lambda x: abs(x[2]), reverse=True)

# # SHOW FINAL RESULT
#     if rf_pred == 1:
#         st.error("Prediction: Heart Disease = YES")
#     else:
#         st.success("Prediction: Heart Disease = NO")

#     st.markdown("Key Factors:")

# # Show top 5 features nicely
#     for feature, value, impact in explanations[:5]:
        
#         if impact > 0:
#             st.error(f"{feature} (Value: {value}) → Increasing Risk")
#         else:
#             st.success(f"{feature} (Value: {value}) → Reducing Risk")






import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap

# Load scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Heart Disease Predictor")

tab1, tab2 = st.tabs(['Predict', 'Model Information'])

with tab1:

    # ================= INPUTS =================
    age = st.text_input("Age (years)",placeholder="Type age...")
    sex = st.selectbox("Sex", ["Select Sex","Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Select chest pain type","Atypical Angina(ATA)", "Non-Anginal Pain(NAP)", "Asymptomatic(ASY)", "Typical Angina(TA)"])
    resting_bp = st.text_input("Resting Blood Pressure (mm Hg)",placeholder="Type resting bp...")
    cholesterol = st.text_input("Cholesterol (mm/dl)", placeholder="Type cholesterol...")
    fasting_bs = st.text_input("Fasting Blood Sugar", placeholder="Type fasting bs...")
    resting_ecg = st.selectbox("Resting ECG Result", ["Select resting ECG","Normal", "ST-T wave Abnormality(ST)", "Left Ventricular Hypertrophy(LVH)"])
    max_hr = st.text_input("Maximum Heart Rate Achieved", placeholder="Type max hr...")
    exercise_angina =  st.selectbox("Exercise-Induced Angina", ["Select exercise angina","Yes", "No"])
    oldpeak = st.text_input("Oldpeak Value", placeholder="Type oldpeak value...")
    st_slope = st.selectbox("Slope of peak Exercise ST Segment ", ["Select st slope","Up", "Flat", "Down"])
    father_history = st.selectbox("Father Heart Disease History", ["Select", "Yes", "No"])
    grandfather_history = st.selectbox("Grandfather Heart Disease History", ["Select", "Yes", "No"])

    # ================= CONVERSION =================
    sex = 0 if sex == "Male" else 1  
    age = int(age) if age else 0
    resting_bp = int(resting_bp) if resting_bp else 0
    cholesterol = int(cholesterol) if cholesterol else 0
    max_hr = int(max_hr) if max_hr else 0
    oldpeak = float(oldpeak) if oldpeak else 0.0

    chest_pain = ["Atypical Angina(ATA)", "Non-Anginal Pain(NAP)", "Asymptomatic(ASY)", "Typical Angina(TA)"].index(chest_pain)

    # Fasting BS (important fix)
    fasting_bs_input = int(fasting_bs) if fasting_bs else 0
    fasting_bs = 0 if fasting_bs_input <= 120 else 1

    resting_ecg = ["Normal", "ST-T wave Abnormality(ST)", "Left Ventricular Hypertrophy(LVH)"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Up", "Flat", "Down"].index(st_slope)

    father_history = 1 if father_history == "Yes" else 0
    grandfather_history = 1 if grandfather_history == "Yes" else 0
    family_history = 1 if (father_history == 1 and grandfather_history == 1) else 0

    # ================= DATAFRAME =================
    input_data = pd.DataFrame({
        'Age' : [age],
        'Sex' : [sex],
        'ChestPainType' : [chest_pain],
        'RestingBP' : [resting_bp],
        'Cholesterol' : [cholesterol],
        'FastingBS' : [fasting_bs],
        'RestingECG' : [resting_ecg], 
        'MaxHR' : [max_hr],
        'ExerciseAngina' : [exercise_angina],
        'Oldpeak' : [oldpeak],
        'ST_Slope': [st_slope],
        'FatherHistory' : [father_history],
        'GrandfatherHistory' : [grandfather_history],
        'FamilyHistory' : [family_history]
    })

    # ================= MODELS =================
    algonames = ['Logistic Regression', 'Support Vector Machine', 'Decision Trees', 'Random Forest', 'Feature Enhanced Explainable Hybrid Stacking Model']
    modelnames = ['logisticR.pkl', 'svm.pkl', 'decisionT.pkl', 'randomF.pkl', 'Feature Enhanced Explainable Hybrid Stacking Model.pkl'] 

    def predict_heart_disease(data):
        results = []
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)[0]

            # probability (if supported) 
            
            if hasattr(model, "predict_proba"): 
                prob = model.predict_proba(data)[0][1]
            else:
                prob = None
    
            results.append((prediction,prob))
        return results

    # ================= CENTER BUTTON =================
    _, _, _, col, _, _, _ = st.columns([1,1,1,2,1,1,1])
    with col:
        submit = st.button("Submit")

    # ================= OUTPUT =================
    if submit:

        input_scaled = scaler.transform(input_data)
        
        # ===== RESULTS =====
        st.subheader('Results......')
        st.markdown('------------------') 

        result = predict_heart_disease(input_scaled)

        for i in range(len(result)): 
            st.subheader(algonames[i])

            pred,prob = result[i]

            if pred == 0:
                st.success("No Heart Disease Detected")
            else:
                st.error("Heart Disease Detected")

            # SHOW PROBABILITY
            if prob is not None:
                st.info(f"Probability: {prob:.2f}")

            st.markdown('--------------------------')

        # ===== SHAP =====
        st.subheader("Why this Prediction? (SHAP implementation based on Random forest)")

        model = pickle.load(open('randomF.pkl', 'rb'))
        rf_pred = model.predict(input_scaled)[0]

        explainer = shap.Explainer(model)
        shap_values = explainer(input_scaled)

        if len(shap_values.values.shape) == 3:
            impacts = shap_values.values[0][:, 1]
        else:
            impacts = shap_values.values[0]

        impacts = impacts.tolist()

        feature_names = input_data.columns

        display_values = { 
            "Age": age,
            "Sex": "Male" if sex == 0 else "Female",
            "ChestPainType": ["ATA","NAP","ASY","TA"][chest_pain],
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs_input,
            "RestingECG": ["Normal","ST","LVH"][resting_ecg],
            "MaxHR": max_hr,
            "ExerciseAngina": "Yes" if exercise_angina == 1 else "No",
            "Oldpeak": oldpeak,
            "ST_Slope": ["Up","Flat","Down"][st_slope],
            "FatherHistory": "Yes" if father_history == 1 else "No",
            "GrandfatherHistory": "Yes" if grandfather_history == 1 else "No",
            "FamilyHistory": "Yes" if family_history == 1 else "No"
            }

        explanations = list(zip(feature_names,impacts))
        explanations = sorted(explanations, key=lambda x: abs(x[1]), reverse=True)

        if rf_pred == 1:
            st.error("Prediction: Heart Disease = YES")
        else:
            st.success("Prediction: Heart Disease = NO")

        st.markdown("Key Factors:")

        for feature, impact in explanations[:5]:
            value = display_values.get(feature, "N/A")
            if impact > 0:
                st.error(f"{feature} (Value: {value}) → Increasing Risk")
            else:
                st.success(f"{feature} (Value: {value}) → Reducing Risk")



# with tab2:

#     import plotly.express as px
#     import pandas as pd
#     from sklearn.metrics import accuracy_score

#     st.subheader("Model Accuracy Comparison")

#     # Load test data (from Colab)
#     X_test_scaled = pickle.load(open('X_test_scaled.pkl', 'rb'))
#     y_test = pickle.load(open('y_test.pkl', 'rb'))

#     # Model names
#     algonames = ['SVM', 'Logistic Regression', 'Random Forest', 'Decision Tree']
#     modelnames = ['svm.pkl', 'logisticR.pkl', 'randomF.pkl', 'decisionT.pkl']

#     accuracies = []  

#     # Calculate accuracy for each model
#     for i in range(len(modelnames)):
#         model = pickle.load(open(modelnames[i], 'rb'))
#         pred = model.predict(X_test_scaled)
#         acc = accuracy_score(y_test, pred)
#         accuracies.append(acc)

#     # Create DataFrame (same as Colab)
#     accuracy_df = pd.DataFrame({
#         'Model': algonames,
#         'Score': accuracies
#     }).sort_values(by='Score', ascending=False)

#     # Plotly Bar Chart
#     fig = px.bar(
#         accuracy_df,
#         x='Model',
#         y='Score',
#         title='Model Accuracy Comparison',
#         labels={'Score': 'Accuracy', 'Model': 'Machine Learning Model'},
#         height=500,
#         template='plotly_white',
#         color='Model'
#     )

#     fig.update_layout(
#         xaxis_title='Machine Learning Model',
#         yaxis_title='Accuracy Score'
#     )

#     # Show chart in Streamlit
#     st.plotly_chart(fig, use_container_width=True)

#     # Best Model Highlight

#     best_row = accuracy_df.loc[accuracy_df['Score'].idxmax()]
#     best_model = best_row['Model']
#     best_score = best_row['Score']

#     st.success(f"Best Model: {best_model} (Accuracy: {best_score:.4f})")

#     st.markdown("---")

#     # Detailed Model Info
#     descriptions = {
#         "Logistic Regression": "Good for linear relationships and baseline model",
#         "SVM": "Effective for high-dimensional data",
#         "Decision Tree": "Easy to interpret but can overfit",
#         "Random Forest": "Robust and handles feature interactions well"
#     }

#     for i in range(len(accuracy_df)):
#         name = accuracy_df.iloc[i]['Model']
#         score = accuracy_df.iloc[i]['Score']

#         st.info(f"🔹 {name}")
#         st.write(f"Accuracy: {score:.4f}")
#         st.write(f"Description: {descriptions.get(name, 'No description available')}")
#         st.markdown("---") 




# with tab2:
#     import plotly.express as px         
#     data = {'Logistic Regression' : 0.858696, 'SVM' : 0.864130, 'Random Forest' : 0.858696	, 'Decision Tree' : 0.831522 }
#     Models = list(data.keys())
#     Accuracies = list(data.values())
#     df = pd.DataFrame(list(zip(Models,Accuracies)), columns=['Models', 'Accuracies'])
#     colors = ['blue', 'green', 'yellow', 'red']
#     best_model = df.loc[df['Accuracies'].idxmax(), 'Models']
#     df['Color'] = df['Models'].apply(lambda x: 'Best' if x == best_model else 'Others')
#     fig = px.bar(df,y='Accuracies', x = 'Models')
#     st.plotly_chart(fig)
#     fig.update_traces(texttemplate='%{y:.4f}', textposition='outside')

with tab2:

    import plotly.express as px
    from sklearn.metrics import accuracy_score

    st.subheader("Model Accuracy Comparison")

    # ================= LOAD TEST DATA =================

    X_test_scaled = pickle.load(open('X_test_scaled.pkl', 'rb'))
    y_test = pickle.load(open('y_test.pkl', 'rb'))

    # ================= MODEL LIST =================

    algonames = [
        'Logistic Regression', 
        'SVM',
        'Decision Tree',
        'Random Forest', 
        'Feature Enhanced Explainable Hybrid Stacking Model'
    ]

    modelnames = [
        'logisticR.pkl', 
        'svm.pkl',
        'decisionT.pkl',
        'randomF.pkl',
        'Feature Enhanced Explainable Hybrid Stacking Model.pkl'
    ]

    accuracies = []

    # ================= CALCULATE ACCURACIES =================

    for modelname in modelnames:

        model = pickle.load(open(modelname, 'rb'))

        pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, pred)
        
        accuracies.append(acc) 

    # ================= DATAFRAME =================

    df = pd.DataFrame({ 
        'Models': algonames,
        'Accuracies': accuracies
    })

    # ================= BEST MODEL =================

    best_model = df.loc[
        df['Accuracies'].idxmax(),
        'Models'
    ]

    # ================= GRAPH =================

    fig = px.bar(
        df,
        x='Models', 
        y='Accuracies',
        color='Models',
        text='Accuracies',
        title='Accuracy Comparison of ML Models',
        template='plotly_white'
    )

    fig.update_traces(
        texttemplate='%{text:.6f}',
        textposition='outside' 
    )

    fig.update_layout(
        xaxis_title='Machine Learning Models',
        yaxis_title='Accuracy Score',
        height=500
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)  

    # ================= BEST MODEL DISPLAY =================

    best_score = df['Accuracies'].max()

    st.markdown("---")

    st.subheader("About the Models")

    st.markdown("""
    • **Logistic Regression:** Used as a baseline machine learning classification model.

    • **Support Vector Machine (SVM):** Effective for classification and high-dimensional datasets.

    • **Decision Tree:** Simple and interpretable tree-based prediction model.

    • **Random Forest:** Ensemble learning model using multiple decision trees for improved prediction performance.

    • **Feature Enhanced Explainable Hybrid Stacking Model (FEEHSM):** Proposed stacking ensemble model that combines Logistic Regression, Support Vector Machine, Decision Tree, and Random Forest to improve prediction accuracy, robustness, and interpretability.""")

    st.markdown("---")

    st.subheader("Features Used")

    st.markdown("""
    • **Age** <br>
    • **Sex** <br>
    • **Chest Pain Type**<br>
    • **Resting Blood Pressure**<br>
    • **Cholesterol**<br>
    • **Fasting Blood Sugar**<br>
    • **Resting ECG**<br>
    • **Maximum Heart Rate**<br>
    • **Exercise-Induced Angina**<br>
    • **Oldpeak**<br>
    • **ST Slope**<br>
    • **Father History**<br>
    • **Grandfather History**<br>
    • **Family History**
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Explainable AI")

    st.write("""
    • SHAP Explainability is used to identify how input
    features influence the prediction generated by
    the Random Forest model.
    """)