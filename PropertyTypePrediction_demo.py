import tkinter as tk
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
import numpy as np
import pandas as pd
# from ExtractingFeatures import ExtractingFeatures

city_map = {
    "HCM":0,
    "Ho Chi Minh City":0,
    "HN":1,
    "Hanoi":1,
    "Ha Noi":1
}
type_map = {
    0:"Apartment",
    1:"Studio Apartment",
    2:"Townhouse",
    3:"Private House",
    4:"Mansion"
}
def show_output():
    rf = pkl.load(open("RandomForest_Classifier_demo.h5", 'rb'))
    # print(rf.feature_names_in_)
    
    area = float(area_entry.get())
    bedroom = float(bedroom_entry.get())
    bathroom = float(bathroom_entry.get())
    city = city_entry.get()
    address = address_entry.get()
    year = year_entry.get()
    price = float(price_entry.get())
    city = city_map.get(city, 1)
    
    # latitude = ExtractingFeatures().locationExtract([address], component='latitude')
    # longitude = ExtractingFeatures().locationExtract([address], component='longitude')
    latitude=20
    longitude=100
    # print(latitude, longitude)
    
    input_data = pd.DataFrame.from_dict({
        "Area (m2)":[area],
        "Bedrooms": [bedroom],
        "Bathrooms": [bathroom],
        "Year": [year],
        "Latitude": [latitude],
        "Longitude": [longitude],
        "Postal Code": [100000], # get postal code
        "City": [city],
        "Price (billion VND)": [price]
    },
    orient='columns'
    )
    type = rf.predict(input_data)[0]
    output_text = f"The estate is a/an {type_map[type]}"
    output_label.config(text=output_text)

root = tk.Tk()
root.title("Real Estates Property Type Classification")

area_label = tk.Label(root, text="Area:")
area_label.grid(row=0, column=0, padx=10, pady=10)
area_entry = tk.Entry(root)
area_entry.grid(row=0, column=1, padx=10, pady=10)

bedroom_label = tk.Label(root, text="Number of bedrooms:")
bedroom_label.grid(row=1, column=0, padx=10, pady=10)
bedroom_entry = tk.Entry(root)
bedroom_entry.grid(row=1, column=1, padx=10, pady=10)

bathroom_label = tk.Label(root, text="Number of bathrooms:")
bathroom_label.grid(row=2, column=0, padx=10, pady=10)
bathroom_entry = tk.Entry(root)
bathroom_entry.grid(row=2, column=1, padx=10, pady=10)

city_label = tk.Label(root, text="City:")
city_label.grid(row=3, column=0, padx=10, pady=10)
city_entry = tk.Entry(root)
city_entry.grid(row=3, column=1, padx=10, pady=10)

address_label = tk.Label(root, text="Address:")
address_label.grid(row=4, column=0, padx=10, pady=10)
address_entry = tk.Entry(root)
address_entry.grid(row=4, column=1, padx=10, pady=10)

year_label = tk.Label(root, text="Year:")
year_label.grid(row=5, column=0, padx=10, pady=10)
year_entry = tk.Entry(root)
year_entry.grid(row=5, column=1, padx=10, pady=10)

price_label = tk.Label(root, text="Price:")
price_label.grid(row=6, column=0, padx=10, pady=10)
price_entry = tk.Entry(root)
price_entry.grid(row=6, column=1, padx=10, pady=10)

submit_button = tk.Button(root, text="Predict", command=show_output)
submit_button.grid(row=7, column=0, columnspan=2, pady=20)

output_label = tk.Label(root, text="", font=("Arial", 14))
output_label.grid(row=8, column=0, columnspan=2, pady=10)

root.mainloop()
