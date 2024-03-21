import tkinter
import tkinter.messagebox
import customtkinter as customtkinter
from tkinter import ttk
from PIL import Image, ImageTk
from customtkinter import CTkImage

#Import the machine learning model
import pickle
with open("random_forest_model.pkl","rb") as file:
    classifier = pickle.load(file)

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("3 Highs Prediction")
        self.geometry(f"{1100}x580")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)
        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="High Blood Pressure,\nHigh Cholesterol,\nDiabetes\nPrediction Model",
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)
        self.introduction_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Introduction",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   anchor="w", command=self.introduction_button_event)
        self.introduction_button.grid(row=1, column=0, sticky="ew")
        self.predict_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Predict",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      anchor="w", command=self.predict_button_event)
        self.predict_button.grid(row=2, column=0, sticky="ew")
        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # Create Introduction frame
        self.introduction_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.introduction_frame.grid_columnconfigure(0, weight=1)

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
        self.totChol_slider = customtkinter.CTkSlider(self.predict_frame,from_=0,to=200,number_of_steps=700)
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

        # Select default frame (Introduction Frame)
        self.select_frame_by_name("Introduction")
        self.explanation_label = customtkinter.CTkLabel(self.introduction_frame, text="3 Highs Prediction: Assessing the Risk of Common Health Conditions\n\n",
                                                        justify="center", font=customtkinter.CTkFont(size=22, weight="bold"), wraplength=800)
        self.explanation_label.grid(row=0, column=0, padx=20, pady=20)

        self.explanation_label2 = customtkinter.CTkLabel(self.introduction_frame, text="The 3 Highs Prediction system is designed to analyze various health factors and provide an assessment of the risk associated with three common health conditions: High Blood Pressure, High Cholesterol, and Diabetes. By inputting relevant information and utilizing predictive algorithms, the system can generate personalized risk predictions and recommendations for individuals. This can be helpful in promoting early detection, preventive measures, and informed decision-making for better health outcomes.\n\nThe system takes into account various factors such as age, gender, lifestyle habits, medical history, and biometric measurements to provide accurate risk assessments. It utilizes advanced algorithms and data analysis techniques to generate predictions based on existing medical research and patterns. By leveraging the power of predictive analytics, the system aims to empower individuals to take proactive steps towards managing their health and reducing the risk of developing these common health conditions.\n",
                                                        justify="left", font=customtkinter.CTkFont(size=16), wraplength=800)
        self.explanation_label2.grid(row=2, column=0, padx=20, pady=10)

        #Image paths
        image_paths = [
            "images\\ImageIntroduction1.jpeg",
            "images\\ImageIntroduction2.jpg",
            "images\\ImageIntroduction3.jpg"
        ]

        # Create a CTkFrame to hold the images
        images_frame = customtkinter.CTkFrame(self.introduction_frame)
        images_frame.grid(row=3, column=0, padx=20, pady=10)

        # Iterate over the image paths and create CTkLabel for each image
        for i, path in enumerate(image_paths):
            # Load and resize the image
            image = Image.open(path)
            resized_image = image.resize((200, 200))

            # Convert the resized image to a Tkinter-compatible format
            tk_image = ImageTk.PhotoImage(resized_image)

            # Create a CTkCanvas and place the image on it
            canvas = customtkinter.CTkCanvas(images_frame, width=200, height=200, bg="white")
            canvas.grid(row=1, column=i, padx=10, pady=10)
            canvas.create_image(0, 0, anchor="nw", image=tk_image)

            # Keep a reference to the image to prevent it from being garbage collected
            canvas.image = tk_image

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
        self.introduction_button.configure(fg_color=("gray75", "gray25") if name == "Introduction" else "transparent")
        self.predict_button.configure(fg_color=("gray75", "gray25") if name == "Predict" else "transparent")

        # show selected frame
        if name == "Introduction":
            self.introduction_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.introduction_frame.grid_forget()
        if name == "Predict":
            self.predict_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.predict_frame.grid_forget()

    def introduction_button_event(self):
        self.select_frame_by_name("Introduction")

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

        new_data = [[gender,age,education,currentSmoker,cigsPerDay,exerciseMinPerWeek,vegetarian,BPMeds,prevalentStroke,prevalentHyp,highBPFH
                    ,hyperchoFH,diabetesFH,totChol,sysBP,diaBP,BMI,heartRate,glucose]]
        
        #Perform prediction using the machine learning model
        prediction_values = classifier.predict(new_data)

        #Map prediction values to label
        prediction_labels = self.map_prediction_to_labels(prediction_values)

        #Remove the predict frame from the layout
        self.predict_frame.grid_forget()

        #Create and grid the result frame in its place
        self.result_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.result_frame.grid(row=0, column=1, sticky="nsew")

        # Configure grid layout for the result frame
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_rowconfigure(0, weight=1)

        #Display the prediction result in the result frame
        self.display_prediction_result(prediction_labels)

    def map_prediction_to_labels(self, prediction_values):
        labels = []

        if prediction_values[0][0] == 0:
            labels.append("Low Risk for High Blood Pressure")
        else:
            labels.append("High Risk for High Blood Pressure")

        if prediction_values[0][1] == 0:
            labels.append("Low Risk for High Cholesterol")
        else:
            labels.append("High Risk for High Cholesterol")

        if prediction_values[0][2] == 0:
            labels.append("Low Risk for Diabetes")
        else:
            labels.append("High Risk for Diabetes")

        return labels

    #Display the prediction results
    def display_prediction_result(self, prediction_labels):

        # Create the back button
        self.back_button = customtkinter.CTkButton(self.result_frame, text="Back", command=self.back_button_event)
        self.back_button.grid(row=2, column=0, padx=20, pady=20, sticky="e")

        #Generate category-specific advice based on prediction labels
        advice = "Advice:\n\n"

        if "High Risk for High Blood Pressure" in prediction_labels:
            advice += "For High Blood Pressure:\n"
            advice += "\u2022Avoid or limit the processed and packaged foods to reduce the sodium intake.\n"
            advice += "\u2022Include potassium-rich foods from sources such as beans, lentils, and chickenpeas.\n"
            advice += "\u2022Limit the intake of alcohol because drinking excessive alcohol can raise blood pressure.\n\n\n"

        if "High Risk for High Cholesterol" in prediction_labels:
            advice += "For High Cholesterol:\n"
            advice += "\u2022Avoid or limit foods high in saturated and trans fats, such as fatty meats, full-fat dairy products, fried foods, and commercially baked goods.\n"
            advice += "\u2022Include heart-healthy fats from sources like avocados, nuts, seeds, and olive oil.\n"
            advice += "\u2022Opt for foods rich in soluble fiber, such as oats, barley, fruits, vegetables, and legumes, which can help lower cholesterol levels.\n\n\n"

        if "High Risk for Diabetes" in prediction_labels:
            advice += "For Diabetes:\n"
            advice += "\u2022Limit the intake of sugary beverages and opt for water, unsweetened tea, or infused water.\n"
            advice += "\u2022Control portion sizes and avoid oversized meals to help manage blood sugar levels.\n"
            advice += "\u2022Choose carbohydrates that have a low glycemic index, such as whole grains, legumes, and non-starchy vegetables.\n\n\n"

        if all(label.startswith("Low Risk") for label in prediction_labels):
            advice += "\u2022Continue with a balanced diet, including plenty of fruits, vegetables, lean proteins, and healthy fats.\n"
            advice += "\u2022Stay physically active by engaging in regular exercise that you enjoy.\n"
            advice += "\u2022Maintain a healthy weight through a combination of healthy eating and regular physical activity.\n"
            advice += "\u2022Find healthy ways to manage stress, such as practicing relaxation techniques or engaging in hobbies.\n"
            advice += "\u2022Schedule regular check-ups with your healthcare professional to monitor your overall health.\n"

        # Open and resize the images
        result_image = Image.open("images\\ResultsIcon.png")
        result_image = result_image.resize((50, 50))  # Adjust the size as needed
        result_icon = ImageTk.PhotoImage(result_image)

        advice_image = Image.open("images\\AdviceIcon.png")
        advice_image = advice_image.resize((50, 50))  # Adjust the size as needed
        advice_icon = ImageTk.PhotoImage(advice_image)

        # Create a frame for the result icon and label
        result_frame = customtkinter.CTkFrame(self.result_frame)
        result_frame.grid(row=0, column=0, padx=20, pady=(0,10), sticky="nw")
        result_frame.grid_rowconfigure(0, weight=1)

        # Create the result icon label
        result_icon_label = tkinter.Label(result_frame, image=result_icon)
        result_icon_label.image = result_icon  # Keep a reference to the image to avoid garbage collection
        result_icon_label.grid(row=0, column=0, padx=(0, 10))

        # Create the result text label
        result_text_label = customtkinter.CTkLabel(result_frame, text="Results:\n" + "\n".join(prediction_labels), justify="left",
                                                font=customtkinter.CTkFont(size=18, weight="bold"), wraplength=1000)
        result_text_label.grid(row=0, column=1, sticky="w")

        # Create a frame for the advice icon and label
        advice_frame = customtkinter.CTkFrame(self.result_frame)
        advice_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        # Create the advice icon label
        advice_icon_label = tkinter.Label(advice_frame, image=advice_icon)
        advice_icon_label.image = advice_icon  # Keep a reference to the image to avoid garbage collection
        advice_icon_label.grid(row=0, column=0, padx=(0, 10))

        # Create the advice text label
        advice_text_label = customtkinter.CTkLabel(advice_frame, text=advice, justify="left",
                                                font=customtkinter.CTkFont(size=18), wraplength=1000)
        advice_text_label.grid(row=0, column=1, sticky="w")
        

    #Back Button function
    def back_button_event(self):
        #Remove the result frame from the layout
        self.result_frame.grid_remove()

        #Back to the original frame
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.predict_frame.grid(row=0, column=1, sticky ='nsew')

        self.grid_columnconfigure(0,weight=0)
        self.grid_columnconfigure(1,weight=3)
        
if __name__ == "__main__":
    app = App()
    app.mainloop()
