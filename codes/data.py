
import numpy as np
import pandas as pd


# Load your CSV file
csv_file_path = '/Users/habeeb/Downloads/AQI/aqData.csv'

try:
    data = pd.read_csv(csv_file_path, sep=',')  # You can change the delimiter as needed
except pd.errors.ParserError:
    print("Error while reading CSV file. Check if the delimiter is correct or if there are any formatting issues.")
    exit()

data.head()