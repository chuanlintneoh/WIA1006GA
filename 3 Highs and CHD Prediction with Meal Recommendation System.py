
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package
import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns

# Data Preprocessing
data = pd.DataFrame(pd.read_csv('Dataset.csv'))
data.dropna(inplace=True)
# Separate the features and targets
from sklearn.model_selection import train_test_split
X = data.iloc[:, :-4].values
y_HighBP = data['HighBP'].values
y_Hypercholesterolemia = data['Hypercholesterolemia'].values
y_diabetes = data['diabetes'].values
y_TenYearCHD = data['TenYearCHD'].values
# Split the dataset into training and testing sets
X_train, X_test, y_HighBP_train, y_HighBP_test = train_test_split(X, y_HighBP, test_size=0.2, random_state=42)
X_train, X_test, y_Hypercholesterolemia_train, y_Hypercholesterolemia_test = train_test_split(X, y_Hypercholesterolemia, test_size=0.2, random_state=42)
X_train, X_test, y_diabetes_train, y_diabetes_test = train_test_split(X, y_diabetes, test_size=0.2, random_state=42)
X_train, X_test, y_TenYearCHD_train, y_TenYearCHD_test = train_test_split(X, y_TenYearCHD, test_size=0.2, random_state=42)
# Scale the features with min-max scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Classification Model Development(Logistic Regression)
from sklearn.linear_model import LogisticRegression
classifier_HighBP = LogisticRegression(random_state=42)
classifier_HighBP.fit(X_train, y_HighBP_train)
classifier_Hypercholesterolemia = LogisticRegression(random_state=42)
classifier_Hypercholesterolemia.fit(X_train, y_Hypercholesterolemia_train)
classifier_diabetes = LogisticRegression(random_state=42)
classifier_diabetes.fit(X_train, y_diabetes_train)
classifier_TenYearCHD = LogisticRegression(random_state=42)
classifier_TenYearCHD.fit(X_train, y_TenYearCHD_train)
# Perform prediction with Logistic Regression Classifier
y_HighBP_pred = classifier_HighBP.predict(X_test)
y_Hypercholesterolemia_pred = classifier_Hypercholesterolemia.predict(X_test)
y_diabetes_pred = classifier_diabetes.predict(X_test)
y_TenYearCHD_pred = classifier_TenYearCHD.predict(X_test)
# Evaluate the model using Cross Validation
from sklearn.model_selection import cross_val_score
classifier_HighBP = LogisticRegression(random_state=42)
scores_HighBP = cross_val_score(classifier_HighBP, X, y_HighBP, cv=5)
print("HighBP Accuracy:", np.mean(scores_HighBP))
classifier_Hypercholesterolemia = LogisticRegression(random_state=42)
scores_Hypercholesterolemia = cross_val_score(classifier_Hypercholesterolemia, X, y_Hypercholesterolemia, cv=5)
print("Hypercholesterolemia Accuracy:", np.mean(scores_Hypercholesterolemia))
classifier_diabetes = LogisticRegression(random_state=42)
scores_diabetes = cross_val_score(classifier_diabetes, X, y_diabetes, cv=5)
print("diabetes Accuracy:", np.mean(scores_diabetes))
classifier_TenYearCHD = LogisticRegression(random_state=42)
scores_TenYearCHD = cross_val_score(classifier_TenYearCHD, X, y_TenYearCHD, cv=5)
print("TenYearCHD Accuracy:", np.mean(scores_TenYearCHD))

# Classification Model Development (Random Forest)
from sklearn.svm import SVC
classifierSVC_HighBP = SVC(kernel='linear', random_state=42)
classifierSVC_HighBP.fit(X_train, y_HighBP_train)
classifierSVC_Hypercholesterolemia = SVC(kernel='linear', random_state=42)
classifierSVC_Hypercholesterolemia.fit(X_train, y_Hypercholesterolemia_train)
classifierSVC_diabetes = SVC(kernel='linear', random_state=42)
classifierSVC_diabetes.fit(X_train, y_diabetes_train)
classifierSVC_TenYearCHD = SVC(kernel='linear', random_state=42)
classifierSVC_TenYearCHD.fit(X_train, y_TenYearCHD_train)
# Perform prediction
y_HighBP_pred = classifierSVC_HighBP.predict(X_test)
y_Hypercholesterolemia_pred = classifierSVC_Hypercholesterolemia.predict(X_test)
y_diabetes_pred = classifierSVC_diabetes.predict(X_test)
y_TenYearCHD_pred = classifierSVC_TenYearCHD.predict(X_test)
# Evaluate the model
from sklearn.metrics import classification_report
print("HighBP Classification Report:")
print(classification_report(y_HighBP_test, y_HighBP_pred))
print("Hypercholesterolemia Classification Report:")
print(classification_report(y_Hypercholesterolemia_test, y_Hypercholesterolemia_pred))
print("diabetes Classification Report:")
print(classification_report(y_diabetes_test, y_diabetes_pred))
print("TenYearCHD Classification Report:")
print(classification_report(y_TenYearCHD_test, y_TenYearCHD_pred))

# # Classification Model Development (Neural Network)
# Y = data[['HighBP', 'Hypercholesterolemia', 'diabetes', 'TenYearCHD']].values
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# from keras.models import Sequential
# from keras.layers import Dense
# # Build the model
# model = Sequential()
# model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(4, activation='softmax'))
# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
# # Evaluate the model
# _, accuracy = model.evaluate(X_test, y_test)
# print('Accuracy: %.2f' % (accuracy*100))
# y_pred = model.predict(X_test)
# # Convert the predictions to binary values
# y_pred_binary = np.round(y_pred)
# # Print the classification report
# print(classification_report(y_test, y_pred_binary))




import tkinter
import tkinter.messagebox
import customtkinter as customtkinter
from tkinter import ttk

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("3 Highs and CHD Prediction with Meal Recommendation System")
        self.geometry(f"{1100}x580")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)
        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="High Blood Pressure,\nHigh Cholesterol,\nDiabetes,\nCoronary Heart Disease\nPrediction Model",
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)
        self.dataset_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Data Set",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   anchor="w", command=self.dataset_button_event)
        self.dataset_button.grid(row=1, column=0, sticky="ew")
        self.predict_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Predict",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.predict_button_event)
        self.predict_button.grid(row=2, column=0, sticky="ew")
        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # Create Dataset frame
        self.dataset_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.dataset_frame.grid_columnconfigure(0, weight=1)

        # Create Predict frame
        '''
        User inputs:
        dropdown    Gender: Male, Female
        slider      Age: 
        slider      Education: 0-4
        switch      Smoking: Yes/No
        slider      Cigarettes per day: 
        slider      Exercise mins per week:
        switch      Vegetarian: Yes/No
        switch      BPMeds: Yes/No
        switch      Prevalent stroke: Yes/No
        switch      Prevalent hyp: Yes/No
        switch      highBPFH: Yes/No
        switch      hyperchoFH: Yes/No
        switch      diabetesFH: Yes/No
        checkbox + slider   totChol: 
        checkbox + slider   sysBP: 
        checkbox + slider   diaBP: 
        checkbox + slider   BMI: 
        checkbox + slider   heartRate: 
        checkbox + slider   glucose (mg/dL): 
        '''
        self.predict_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.predict_frame.grid(row=10,column=3)
        self.predict_frame.grid_columnconfigure((0,1), weight=1)
        self.gender_label = customtkinter.CTkLabel(self.predict_frame,text="Gender:")
        self.gender_switch = customtkinter.CTkSegmentedButton(self.predict_frame,values=["Male","Female"])
        self.age_slider = customtkinter.CTkSlider(self.predict_frame,from_=1,to=75,number_of_steps=74)
        self.age_label = customtkinter.CTkLabel(self.predict_frame,text=("Age: "+ str(self.age_slider.get())))
        self.education_slider = customtkinter.CTkSlider(self.predict_frame,from_=0,to=4,number_of_steps=4)
        self.education_label = customtkinter.CTkLabel(self.predict_frame,text=("Education Level: " + str(self.education_slider.get())))
        self.smoking_label = customtkinter.CTkLabel(self.predict_frame,text="Smoking")
        self.smoking_switch = customtkinter.CTkSegmentedButton(self.predict_frame,values=["No","Yes"])
        self.cigsPerDay_slider = customtkinter.CTkSlider(self.predict_frame,from_=0,to=20,number_of_steps=20)
        self.cigsPerDay_label = customtkinter.CTkLabel(self.predict_frame,text=("Cigarettes per day: " + str(self.cigsPerDay_slider.get())))
        self.exerciseMinPerWeek_slider = customtkinter.CTkSlider(self.predict_frame,from_=0,to=120,number_of_steps=12)
        self.exerciseMinPerWeek_label = customtkinter.CTkLabel(self.predict_frame,text=("Exercise per week: " + str(self.exerciseMinPerWeek_slider.get()) + " mins"))
        self.vegetarian_label = customtkinter.CTkLabel(self.predict_frame,text="Vegetarian")
        self.vegetarian_switch = customtkinter.CTkSegmentedButton(self.predict_frame,values=["No","Yes"])
        self.BPMeds_label = customtkinter.CTkLabel(self.predict_frame,text="Blood pressure medication")
        self.BPMeds_switch = customtkinter.CTkSegmentedButton(self.predict_frame,values=["No","Yes"])
        self.prevalentStroke_label = customtkinter.CTkLabel(self.predict_frame,text="Prevalent stroke")
        self.prevalentStroke_switch = customtkinter.CTkSegmentedButton(self.predict_frame,values=["No","Yes"])
        self.prevalentHyp_label = customtkinter.CTkLabel(self.predict_frame,text="Prevalent hypertension")
        self.prevalentHyp_switch = customtkinter.CTkSegmentedButton(self.predict_frame,values=["No","Yes"])
        self.highBPFH_label = customtkinter.CTkLabel(self.predict_frame,text="High blood pressure family history")
        self.highBPFH_switch = customtkinter.CTkSegmentedButton(self.predict_frame,values=["No","Yes"])
        self.hyperchoFH_label = customtkinter.CTkLabel(self.predict_frame,text="High cholesterol family history")
        self.hyperchoFH_switch = customtkinter.CTkSegmentedButton(self.predict_frame,values=["No","Yes"])
        self.diabetesFH_label = customtkinter.CTkLabel(self.predict_frame,text="Diabetes family history")
        self.diabetesFH_switch = customtkinter.CTkSegmentedButton(self.predict_frame,values=["No","Yes"])
        self.totChol_slider = customtkinter.CTkSlider(self.predict_frame,from_=0,to=200,number_of_steps=200)
        self.totChol_label = customtkinter.CTkLabel(self.predict_frame,text=("Total cholesterol: "+str(self.totChol_slider.get()) + " mg/dL"))
        self.sysBP_slider = customtkinter.CTkSlider(self.predict_frame,from_=0,to=180,number_of_steps=180)
        self.sysBP_label = customtkinter.CTkLabel(self.predict_frame,text=("Systolic blood pressure: "+str(self.sysBP_slider.get())+ " mmHg"))
        self.diaBP_slider = customtkinter.CTkSlider(self.predict_frame,from_=0,to=150,number_of_steps=150)
        self.diaBP_label = customtkinter.CTkLabel(self.predict_frame,text=("Diastolic blood pressure: "+str(self.diaBP_slider.get())+ " mmHg"))
        self.BMI_slider = customtkinter.CTkSlider(self.predict_frame,from_=0,to=30,number_of_steps=60)
        self.BMI_label = customtkinter.CTkLabel(self.predict_frame,text=("BMI: "+str(self.BMI_slider.get())))
        self.heartRate_slider = customtkinter.CTkSlider(self.predict_frame,from_=0,to=180,number_of_steps=180)
        self.heartRate_label = customtkinter.CTkLabel(self.predict_frame,text=("Heart rate: "+str(self.heartRate_slider.get())+" bpm"))
        self.glucose_slider = customtkinter.CTkSlider(self.predict_frame,from_=0,to=180,number_of_steps=180)
        self.glucose_label = customtkinter.CTkLabel(self.predict_frame,text=("Glucose: "+str(self.glucose_slider.get()) + " mg/dL"))
        self.submit_button = customtkinter.CTkButton(self.predict_frame,text="Submit",command=self.submit_button_event,hover=True,anchor="s",width=0)

        # add command to update label values when sliders are moved
        self.age_slider.configure(command=self.update_age_label)
        self.education_slider.configure(command=self.update_education_label)
        self.cigsPerDay_slider.configure(command=self.update_cigsPerDay_label)
        self.exerciseMinPerWeek_slider.configure(command=self.update_exerciseMinPerWeek_label)
        self.totChol_slider.configure(command=self.update_totChol_label)
        self.sysBP_slider.configure(command=self.update_sysBP_label)
        self.diaBP_slider.configure(command=self.update_diaBP_label)
        self.BMI_slider.configure(command=self.update_BMI_label)
        self.heartRate_slider.configure(command=self.update_heartRate_label)
        self.glucose_slider.configure(command=self.update_glucose_label)
        
        self.age_label.grid(row=0,column=0)
        self.education_label.grid(row=2,column=0)
        self.cigsPerDay_label.grid(row=4,column=0)
        self.exerciseMinPerWeek_label.grid(row=6,column=0)
        self.totChol_label.grid(row=8,column=0)
        self.sysBP_label.grid(row=10,column=0)
        self.diaBP_label.grid(row=12,column=0)
        self.BMI_label.grid(row=14,column=0)
        self.heartRate_label.grid(row=16,column=0)
        self.glucose_label.grid(row=18,column=0)
        self.gender_label.grid(row=0,column=1)
        self.smoking_label.grid(row=2,column=1)
        self.vegetarian_label.grid(row=4,column=1)
        self.BPMeds_label.grid(row=6,column=1)
        self.prevalentStroke_label.grid(row=8,column=1)
        self.prevalentHyp_label.grid(row=10,column=1)
        self.highBPFH_label.grid(row=12,column=1)
        self.hyperchoFH_label.grid(row=14,column=1)
        self.diabetesFH_label.grid(row=16,column=1)

        self.age_slider.grid(row=1,column=0)
        self.education_slider.grid(row=3,column=0)
        self.cigsPerDay_slider.grid(row=5,column=0)
        self.exerciseMinPerWeek_slider.grid(row=7,column=0)
        self.totChol_slider.grid(row=9,column=0)
        self.sysBP_slider.grid(row=11,column=0)
        self.diaBP_slider.grid(row=13,column=0)
        self.BMI_slider.grid(row=15,column=0)
        self.heartRate_slider.grid(row=17,column=0)
        self.glucose_slider.grid(row=19,column=0)
        self.gender_switch.grid(row=1,column=1)
        self.smoking_switch.grid(row=3,column=1)
        self.vegetarian_switch.grid(row=5,column=1)
        self.BPMeds_switch.grid(row=7,column=1)
        self.prevalentStroke_switch.grid(row=9,column=1)
        self.prevalentHyp_switch.grid(row=11,column=1)
        self.highBPFH_switch.grid(row=13,column=1)
        self.hyperchoFH_switch.grid(row=15,column=1)
        self.diabetesFH_switch.grid(row=17,column=1)

        self.submit_button.grid(column=2)

        # Select default frame
        self.select_frame_by_name("Data Set")

    def update_age_label(self,value):
        self.age_label.configure(text=("Age: " + str(self.age_slider.get())))

    def update_education_label(self,value):
        self.education_label.configure(text=("Education Level: " + str(self.education_slider.get())))
    
    def update_cigsPerDay_label(self,value):
        self.cigsPerDay_label.configure(text=("Cigarettes per day: " + str(self.cigsPerDay_slider.get())))
    
    def update_exerciseMinPerWeek_label(self,value):
        self.exerciseMinPerWeek_label.configure(text="Exercise per week: " + str(self.exerciseMinPerWeek_slider.get())+ " mins")
        
    def update_totChol_label(self,value):
        self.totChol_label.configure(text=("Total cholesterol: "+str(self.totChol_slider.get())+ " mg/dL"))

    def update_sysBP_label(self,value):
        self.sysBP_label.configure(text=("Systolic blood pressure: "+str(self.sysBP_slider.get())+ " mmHg"))

    def update_diaBP_label(self,value):
        self.diaBP_label.configure(text=("Diastolic blood pressure: "+str(self.diaBP_slider.get())+ " mmHg"))

    def update_BMI_label(self,value):
        self.BMI_label.configure(text=("BMI: "+str(self.BMI_slider.get())))

    def update_heartRate_label(self,value):
        self.heartRate_label.configure(text=("Heart rate: "+str(self.heartRate_slider.get())+" bpm"))

    def update_glucose_label(self,value):
        self.glucose_label.configure(text=("Glucose: "+str(self.glucose_slider.get())+ " mg/dL"))

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.dataset_button.configure(fg_color=("gray75", "gray25") if name == "Data Set" else "transparent")
        self.predict_button.configure(fg_color=("gray75", "gray25") if name == "Predict" else "transparent")

        # show selected frame
        if name == "Data Set":
            self.dataset_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.dataset_frame.grid_forget()
        if name == "Predict":
            self.predict_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.predict_frame.grid_forget()

    def dataset_button_event(self):
        self.select_frame_by_name("Data Set")

    def predict_button_event(self):
        self.select_frame_by_name("Predict")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)
        
    def submit_button_event(self):
        if self.gender_switch.get() == "Female":
            gender = 0
        elif self.gender_switch.get() == "Male":
            gender = 1
        else :
            gender = "N/A"
        age = self.age_slider.get()
        education = self.education_slider.get()
        if self.smoking_switch.get() == "No":
            currentSmoker = 0
        elif self.smoking_switch.get() == "Yes":
            currentSmoker = 1
        else :
            currentSmoker = "N/A"
        cigsPerDay = self.cigsPerDay_slider.get()
        exerciseMinPerWeek = self.exerciseMinPerWeek_slider.get()
        if self.vegetarian_switch.get() == "No":
            vegetarian = 0
        elif self.vegetarian_switch.get() == "Yes":
            vegetarian = 1
        else :
            vegetarian = "N/A"
        if self.BPMeds_switch.get() == "No":
            BPMeds = 0
        elif self.BPMeds_switch.get() == "Yes":
            BPMeds = 1
        else :
            BPMeds = "N/A"
        if self.prevalentStroke_switch.get() == "No":
            prevalentStroke = 0
        elif self.prevalentStroke_switch.get() == "Yes":
            prevalentStroke = 1
        else :
            prevalentStroke = "N/A"
        if self.prevalentHyp_switch.get() == "No":
            prevalentHyp = 0
        elif self.prevalentHyp_switch.get() == "Yes":
            prevalentHyp = 1
        else :
            prevalentHyp = "N/A"
        if self.highBPFH_switch.get() == "No":
            highBPFH = 0
        elif self.highBPFH_switch.get() == "Yes":
            highBPFH = 1
        else :
            highBPFH = "N/A"
        if self.hyperchoFH_switch.get() == "No":
            hyperchoFH = 0
        elif self.hyperchoFH_switch.get() == "Yes":
            hyperchoFH = 1
        else :
            hyperchoFH = "N/A"
        if self.diabetesFH_switch.get() == "No":
            diabetesFH = 0
        elif self.diabetesFH_switch.get() == "Yes":
            diabetesFH = 1
        else :
            diabetesFH = "N/A"
        totChol = self.totChol_slider.get()
        sysBP = self.sysBP_slider.get()
        diaBP = self.diaBP_slider.get()
        BMI = self.BMI_slider.get()
        heartRate = self.heartRate_slider.get()
        glucose = self.glucose_slider.get()
        print("User input:")
        print("Gender: " ,gender)
        print("Age: " ,age)
        print("Education: " ,education)
        print("CurrentSmoker: " ,currentSmoker)
        print("CigsPerDay: " ,cigsPerDay)
        print("ExerciseMinsPerWeek" ,exerciseMinPerWeek)
        print("Vegetarian: " ,vegetarian)
        print("BloodPressureMedication: " ,BPMeds)
        print("PrevalentStroke: " ,prevalentStroke)
        print("PrevalentHypertension: " ,prevalentHyp)
        print("HighBloodPressureFamilyHistory: " ,highBPFH)
        print("HypercholesterolFamilyHistory: " ,hyperchoFH)
        print("DiabetesFamilyHistory: " ,diabetesFH)
        print("TotalCholesterol: " ,totChol)
        print("SystolicBloodPressure: " ,sysBP)
        print("DiastolicBloodPressure: " ,diaBP)
        print("BMI: " ,BMI)
        print("HeartRate: " ,heartRate)
        print("Glucose: " ,glucose)

if __name__ == "__main__":
    app = App()
    app.mainloop()
