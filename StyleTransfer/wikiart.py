import pandas as pd
import requests
import os

# First download the file from https://www.kaggle.com/datasets/antoinegruson/-wikiart-all-images-120k-link
# Read the CSV file
df = pd.read_csv("wikiart_scraped.csv")

# Extract the Link column
links = df["Link"]

# Define the path to the directory to save the images
output_dir = "input/style"

# Create the directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through the links and download the images
for i, link in enumerate(links):       
    filename = f"{i}.jpg"
    filepath = os.path.join(output_dir, filename)
    response = requests.get(link)
    with open(filepath, "wb") as f:
        f.write(response.content)
    print(f"Downloaded image {i+1}/{len(df)}")
    
print("All images downloaded successfully!")