import face_recognition

laska1 = "models/cruz/image-2.jpeg"
laska2 = "models/cruz/image-5.jpeg"
laska3 = "models/clarke/image-5.jpeg"


def load_face_to_comparison(face_to_compare):
    try:
        known_image = face_recognition.load_image_file(face_to_compare)
        # print(known_image)
        face_encoding = face_recognition.face_encodings(known_image)[0]
        # print(face_encoding)
        return face_encoding
    except:
        print("Can't recognize face on this image:" + face_to_compare)
        return False


def load_pattern_face(pattern_face):
    try:
        known_image = face_recognition.load_image_file(pattern_face)
        pattern_face_encoding = face_recognition.face_encodings(known_image)[0]
        # print(pattern_face_encoding)
        return pattern_face_encoding
    except:
        print("Can't recognize face on this image:" + pattern_face)
        return False


def face_comparison(face_to_compare, pattern_face):

    if (load_face_to_comparison(face_to_compare) is not False and
            load_pattern_face(pattern_face) is not False):

        encoded_face_to_compare = load_face_to_comparison(face_to_compare)
        encoded_pattern_face = load_pattern_face(pattern_face)
        known_encoding = [
            encoded_face_to_compare
        ]

        face_distances = face_recognition.face_distance(
            known_encoding, encoded_pattern_face)

        for i, face_distance in enumerate(face_distances):
            print(
                "Face to compare has a distance of {:.2} from pattern\
                face #{}".format(
                    face_distance, i))
            print(
                "- With a normal cutoff of 0.6, would the face to compare\
                match the pattern face? {}".format(face_distance < 0.6))
            print(
                "- With a very strict cutoff of 0.5, would the face to compare\
                match the pattern face? {}".format(face_distance < 0.5))
            print()
    else:
        pass


face_comparison(laska1, laska2)
