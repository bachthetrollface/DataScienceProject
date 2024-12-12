import pickle as pkl
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import random

property_type_map = {'Căn hộ': "Apartment", 
                     'Căn hộ Studio': "Studio Apartment", 
                     'Nhà phố': "Townhouse", 
                     'Nhà riêng': "Private House", 
                     'biệt thự': "Mansion"}
def convert_to_eng(name:str) -> str:
    return property_type_map[name] if name in property_type_map.keys() else "undefined"

# Load the model and encoder
encoder = pkl.load(open("models/CategoricalDataEncoder.h5", 'rb'))
model = pkl.load(open("models/RandomForest.h5", 'rb'))
data = pd.read_csv("data/test_data_2nd.csv")

# Preparing the dataset
df = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# For displaying categorical columns (decode the encoded columns)
df_display = df.copy()
df_display[["Property Type", "Address", "Law Document", "City"]] = encoder.inverse_transform(
    df[["Property Type", "Address", "Law Document", "City"]]
)
df_display["Property Type"] = df_display["Property Type"].apply(convert_to_eng)
df_display = df_display.astype({"Bedrooms":np.int32, 
                                "Bathrooms":np.int32, 
                                "Year":np.int32, 
                                "Quarter":np.int32,
                                "Postal Code":np.int32})

# Function to update the record displayed on the GUI and perform prediction
def update_record_and_predict():
    try:
        # Get a random record
        idx = random.randrange(0, data.shape[0])
        record = df.iloc[idx]
        display_record = df_display.iloc[idx]
        
        # Update the display with the new record's details in the Treeview
        for i, col in enumerate(df_display.columns):
            treeview.item(i+1, values=(col, display_record[col]))  # Update Treeview rows

        # Perform prediction
        prediction = model.predict(pd.DataFrame([record]))  # Assuming regression model
        
        # Show the prediction result
        prediction_label.config(
            text=f"Actual Price: {y[idx]:.3f} billion VND\nPredicted Price: {prediction[0]:.3f} billion VND"
        )
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# GUI Setup
def create_gui():
    # Create the main window
    window = tk.Tk()
    window.title("Real Estate Price Prediction")
    window.geometry("800x600")
    window.config(bg="#f5f5f5")  # Light grey background for the window
    
    # Title label
    title_label = tk.Label(window, text="Randomly Selected Record", font=("Arial", 20, "bold"), bg="#f5f5f5")
    title_label.pack(pady=20)
    
    # Frame for record and prediction
    frame = tk.Frame(window, bg="#f5f5f5")
    frame.pack(padx=20, pady=10, fill=tk.X)
    
    # Treeview for displaying the record information
    global treeview
    treeview = ttk.Treeview(frame, columns=("Attribute", "Value"), show="headings", height=14)
    treeview.pack(pady=10, fill=tk.X)

    # Define columns and headings
    treeview.heading("Attribute", text="Attribute", anchor="w")
    treeview.heading("Value", text="Value", anchor="w")
    
    # Add empty rows for the record values
    for i in range(len(df_display.columns)):
        treeview.insert("", "end", iid=i+1, values=("", ""))

    # Label for displaying the prediction result
    global prediction_label
    prediction_label = tk.Label(window, text="Predicted Value: ", font=("Arial", 16, "bold"), bg="#f5f5f5", fg="#333")
    prediction_label.pack(pady=20)
    
    # Button to select a new random record and update the GUI
    new_record_button = tk.Button(window, text="Try New Record", command=update_record_and_predict, font=("Arial", 14), bg="#4CAF50", fg="black", relief="raised", padx=10, pady=5)
    new_record_button.pack(pady=15)
    
    # Initial update (load the first random record)
    update_record_and_predict()

    # Run the GUI
    window.mainloop()

# Start the GUI
create_gui()
