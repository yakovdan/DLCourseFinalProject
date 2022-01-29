import cv2
import os
import dlib
from align_faces_parallel import align_face


def run_alignment(image_path):
    print(os.getcwd())
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

path = os.getcwd()
path = path + "/ImagesFromNet/converted"
files = os.listdir(path)
for filename in files[0:1]:
    name = filename.split(".")[0]
    result = run_alignment(path+"/"+filename)
    #result.save(name+"_aligned.jpeg", "JPEG")