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


encoder = pkl.load(open("models/CategoricalDataEncoder.h5", 'rb'))
model = pkl.load(open("models/RandomForest.h5", 'rb'))
data = pd.read_csv("data/test_data_2nd.csv")

df = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

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


def update_record_and_predict():
    try:
        idx = random.randrange(0, data.shape[0])
        record = df.iloc[idx]
        display_record = df_display.iloc[idx]

        for i, col in enumerate(df_display.columns):
            treeview.item(i+1, values=(col, display_record[col]))

        prediction = model.predict(pd.DataFrame([record]))
        prediction_label.config(
            text=f"Actual Price: {y[idx]:.3f} billion VND\nPredicted Price: {prediction[0]:.3f} billion VND"
        )
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def create_gui():
    window = tk.Tk()
    window.title("Real Estate Price Prediction")
    window.geometry("800x600")
    window.config(bg="#f5f5f5")

    title_label = tk.Label(window, text="Randomly Selected Record", font=("Arial", 20, "bold"), bg="#f5f5f5")
    title_label.pack(pady=20)

    frame = tk.Frame(window, bg="#f5f5f5")
    frame.pack(padx=20, pady=10, fill=tk.X)
    
    global treeview
    treeview = ttk.Treeview(frame, columns=("Attribute", "Value"), show="headings", height=14)
    treeview.pack(pady=10, fill=tk.X)
    treeview.heading("Attribute", text="Attribute", anchor="w")
    treeview.heading("Value", text="Value", anchor="w")
    for i in range(len(df_display.columns)):
        treeview.insert("", "end", iid=i+1, values=("", ""))

    global prediction_label
    prediction_label = tk.Label(window, text="Predicted Value: ", font=("Arial", 16, "bold"), bg="#f5f5f5", fg="#333")
    prediction_label.pack(pady=20)

    new_record_button = tk.Button(window, text="Try New Record", command=update_record_and_predict, font=("Arial", 14), bg="#4CAF50", fg="black", relief="raised", padx=10, pady=5)
    new_record_button.pack(pady=15)

    update_record_and_predict()
    window.mainloop()

create_gui()
