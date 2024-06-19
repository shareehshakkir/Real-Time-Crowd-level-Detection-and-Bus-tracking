from cv2 import VideoCapture, imencode
import cv2
from time import sleep
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db, firestore,storage
import sys
def capture_image():
    camera = VideoCapture(0)
    ret, frame = camera.read()
    camera.release()
    image_path = "captured_image.jpg"
    cv2.imwrite(image_path, frame)
    return image_path
model = YOLO(r"best.torchscript")
cred = credentials.Certificate("trackntravel-8abd7-firebase-adminsdk-bkbwz-ccd8d71b1d.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://trackntravel-8abd7.firebaseio.com/',
    'storageBucket': 'trackntravel-8abd7.appspot.com'
})
db = firestore.client()
bucket = storage.bucket()
ref=db.collection('users')
def toggle_device(busid):
    try:
        query = ref.where('busid', '==', busid).get()
        
        if query:
            doc_ref = ref.document(query[0].id)
            doc_ref_dict = doc_ref.get().to_dict()
            
            # Check if the 'inuse' field exists and is a boolean
            if 'inuse' in doc_ref_dict and isinstance(doc_ref_dict['inuse'], bool):
                # Toggle the boolean value of 'inuse'
                inuse_value = not doc_ref_dict['inuse']
                doc_ref.update({'inuse': inuse_value})
                print("Set to", inuse_value)
            else:
                print("'inuse' field is missing or not a boolean in document with busid:", busid)
        else:
            print("Document not found for busid:", busid)
    except Exception as e:
        print("Error toggling device:", e)




# Modify update_data function to release resources properly
from datetime import datetime

def update_data(busid, number_of_heads, crowd, image):
    try:
        destination_folder = f"photos/{busid}/"
        print(destination_folder)
        '''# Delete existing folder with images using busid
        existing_folder = bucket.blob(destination_folder)
        print(existing_folder.exists())
        if existing_folder.exists():
            blobs = bucket.list_blobs(prefix=destination_folder)  # List all blobs in the folder
            for blob in blobs:
                blob.delete()  # Delete each blob (image) in the folder
            print("Existing folder deleted successfully.")'''
        blobs = list(bucket.list_blobs(prefix=destination_folder))
        
        if blobs:
            # "Folder" exists if there are blobs with the specified prefix
            print("Existing folder detected.")
            
            # Delete all blobs (images) in the folder
            for blob in blobs:
                blob.delete()
            print("Existing folder deleted successfully.")
        else:
            print("Folder does not exist.")

        # Generate a unique filename using busid and timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_filename = f"{timestamp}"  # Assuming the image format is JPG

        # Upload new image with the unique filename to the busid folder
        new_blob = bucket.blob(destination_folder + image_filename)
        new_blob.upload_from_filename(image, predefined_acl='publicRead')
        image_url = new_blob.public_url

        # Update document in Firestore with the new image URL
        ref = db.collection('users')
        query = ref.where('busid', '==', busid).get()

        if query:
            doc_ref = ref.document(query[0].id)
            doc_ref.update({'no_of_people': number_of_heads, 'crowd': str(crowd), 'photo': image_url})
            print("Data updated successfully")
        else:
            print("Document not found for busid:", busid)
    except Exception as e:
        print("Error updating data:", e)


import signal

def signal_handler(signal, frame):
    print("\nExiting program.")
    toggle_device(busid)
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


busid = input("Enter bus id: ")
toggle_device(busid)


while True:
        image = capture_image()
        results = model(image)
        for result in results:
            boxes = result.boxes
            masks = result.masks
            number_of_heads = len(result.boxes)
            print('Total number of heads:', number_of_heads)
            keypoints = result.keypoints
            probs = result.probs
            result.show()
            left, right, middle = [0, 0, 0]
            blurred_image = cv2.imread(image)
            image = cv2.imread(image)
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 =[int(x1),int(y1),int(x2),int(y2)]

                blurred_head = cv2.GaussianBlur(blurred_image[y1:y2, x1:x2], (51, 51), 0)
                # Replace head region with blurred version
                blurred_image[y1:y2, x1:x2] = blurred_head

            # Show the blurred image
        
            blurred_image_path = "blurred_image.jpg"
            cv2.imwrite(blurred_image_path, blurred_image)
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                image_width = image.shape[1]
                if center_x < image_width / 3:
                    left += 1
                elif center_x > 2 * image_width / 3:
                    right += 1
                else:
                    middle += 1
            max = 50
            crowdlevel = 0
            if number_of_heads <= 50 and number_of_heads>=40:
                if middle <= 10 and left > 20 and right > 20:
                    crowdlevel = 7
                elif middle==0:
                    crowdlevel = 6
            elif number_of_heads>50:
                if middle >= 10:
                    crowdlevel = 10
                elif middle < 5:
                    crowdlevel = 9
                else:
                    crowdlevel = 8
            else:
                if left<=20 and right <=20:
                    if left>=15 and right>=15:
                        crowdlevel=5
                    elif left>=10 and right>=10:
                        crowdlevel=4
                    elif left>=5 and right>=5:
                        crowdlevel=3
                    elif left>0 and right>0:
                        crowdlevel=2
                    elif middle<10:
                        crowdlevel=1
                    else:
                        crowdlevel=0
                
        update_data(busid,number_of_heads,crowdlevel,blurred_image_path)
        sleep(60)

