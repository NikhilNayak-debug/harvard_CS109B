import pandas as pd
import requests
import os

# Read the CSV file
df = pd.read_csv("wikiart_scraped.csv")

# Extract the Link column
links = df["Link"]

# Create a directory to save the images if it does not exist
if not os.path.exists("images"):
    os.makedirs("images")

# Loop through the links and download the images
for i, link in enumerate(links):
    # Create the file path and name to save the image
    file_path = f"images/{i}.jpg"
    
    # Download the image
    response = requests.get(link)

    # Save the image to the file path
    with open(file_path, "wb") as f:
        f.write(response.content)
    
    print(f"Image {i} downloaded successfully!")
