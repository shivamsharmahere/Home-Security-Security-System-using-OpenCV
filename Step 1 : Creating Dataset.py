import cv2
from tqdm import tqdm
import os

def generate_dataset(user_name, user_id, video_path):
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            return None

        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        return cropped_face

    cap = cv2.VideoCapture(video_path)
    img_id = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc='Generating Dataset', unit=' frames')

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if there's an issue reading frames from the video capture
        
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame), (200, 200))
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_path = f"data/{user_name}.{user_id}.{img_id}.jpg"  # Save images with user's name and ID
            cv2.imwrite(file_path, face)
            
            # Augmenting the dataset by applying transformations
            flipped_img = cv2.flip(face, 1)
            flipped_file_path = f"data/{user_name}.{user_id}.{img_id}_flipped.jpg"  # Save flipped images with user's name and ID
            cv2.imwrite(flipped_file_path, flipped_img)
            
            img_id += 1
            progress_bar.update(1)
                
    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()
    print("Image augmentation completed.")

def register_user(users_dict):
    user_name = input("Enter your name: ")
    video_path = input("Enter the path of the video file: ")
    user_id = max(users_dict.keys(), default=0) + 1  # Generate a unique user ID
    users_dict[user_id] = user_name
    return user_id, user_name, video_path

def get_users_dict(data_dir):
    users_dict = {}
    for filename in os.listdir(data_dir):
        user_id = int(filename.split(".")[1])
        user_name = filename.split(".")[0]
        users_dict[user_id] = user_name
    return users_dict

# Main program
data_dir = "data"
users_dict = get_users_dict(data_dir)

while True:
    choice = input("Do you want to register your face? (yes/no): ").lower()
    if choice == 'yes':
        user_id, user_name, video_path = register_user(users_dict)
        generate_dataset(user_name, user_id, video_path)
    elif choice == 'no':
        break
    else:
        print("Invalid choice. Please enter 'yes' or 'no'.")

print("Registered users:")
for user_id, user_name in users_dict.items():
    print(f"User ID: {user_id}, Name: {user_name}")
