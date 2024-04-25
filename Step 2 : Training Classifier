import numpy as np
from PIL import Image
import os, cv2

def train_classifier(data_dir):
    # Get the list of image files in the data directory
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []  # List to store face images
    ids = []    # List to store corresponding user IDs

    # Iterate through each image file
    for image in path:
        img = Image.open(image).convert('L')  # Open image and convert to grayscale
        imageNp = np.array(img, 'uint8')       # Convert image to numpy array
        
        # Parse the file name to extract the user ID
        file_name = os.path.basename(image)
        id_str = file_name.split(".")[1]       # Get the part between '.' and '.jpg'
        id = int(id_str.split("_")[0])         # Extract the ID from the part before '_flipped'

        faces.append(imageNp)  # Add face image to the faces list
        ids.append(id)         # Add user ID to the ids list

    # Convert faces and ids lists to numpy arrays
    faces = np.array(faces)
    ids = np.array(ids).astype(np.int32)

    # Create LBPH Face Recognizer
    clf = cv2.face.LBPHFaceRecognizer_create()
    
    # Train the recognizer with faces and corresponding IDs
    clf.train(faces, ids)
    
    # Save the trained recognizer to a file
    clf.write("classifier1.yml")

# Call the train_classifier function with the data directory
train_classifier("data")
