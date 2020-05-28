import face_recognition
from sklearn import svm
import os

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
train_dir = os.listdir('models/')

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("models/" + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("models/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        # If training image contains none or more than faces, print an error message and exit
        if len(face_bounding_boxes) != 1:
            print(person + "/" + person_img + " contains none or more than one faces and can't be used for training.")
            exit()
        else:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings, names)

# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('train/test_image.jpg')
test_image2 = face_recognition.load_image_file('train/test_image2.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
face_locations2 = face_recognition.face_locations(test_image2)


no = len(face_locations)
no2 = len(face_locations2)


print("Number of faces detected in test 1: ", no)
print("Number of faces detected in test 2: ", no2)


# Predict all the faces in the test image using the trained classifier
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    print("Model testowy 1 jest podobny do:")
    print(*name)

for i in range(no2):
    test_image_enc2 = face_recognition.face_encodings(test_image2)[i]
    name2 = clf.predict([test_image_enc2])
    print("Model testowy 2 jest podobny do:")
    print(*name2)
