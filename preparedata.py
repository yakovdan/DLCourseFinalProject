import numpy as np
import cv2
import os
import dlib
from align_faces_parallel import align_face

from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor

face_detector = RetinaFacePredictor(threshold=0.8, device="cuda",
                                    model=(RetinaFacePredictor.get_model('mobilenet0.25')))
face_parser = RTNetPredictor(
    device="cuda", ckpt=None, encoder="rtnet50", decoder="fcn", num_classes=11)


def face_area(face):
    return (face[2] - face[0]) * (face[3] - face[1])


def predicate(faces):
    l = faces.shape[0]
    areas = np.zeros(l, dtype=np.float32)
    for i in range(faces.shape[0]):
        a = face_area(faces[i])
        areas[i] = a
    return areas


def segment_face(frame):
    faces = face_detector(frame, rgb=False)
    if len(faces) > 1:
        items = predicate(faces)
        order = np.argsort(items)
        order = np.flip(order, axis=0)
        faces = faces[order]
        faces = faces[0].reshape((1, 15))

    # Parse faces
    masks = face_parser.predict_img(frame, faces, rgb=False)
    frame_copy = np.copy(frame)
    ind = masks[0, :, :] == 0
    ind2 = masks[0, :, :] != 0
    frame_copy[:, :, 0][ind] = 0
    frame_copy[:, :, 1][ind] = 0
    frame_copy[:, :, 2][ind] = 0
    frame_copy[:, :, 0][ind2] = 255
    frame_copy[:, :, 1][ind2] = 255
    frame_copy[:, :, 2][ind2] = 255

    frame_thresh = 255 * np.ones((256, 256), dtype=np.uint8)
    frame_thresh[ind] = 0

    thresh = cv2.threshold(frame_thresh, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    cc_areas = [stats[x, cv2.CC_STAT_AREA] for x in range(1, numLabels)]
    largest_label_by_area = np.argmax(np.array(cc_areas)) + 1
    ind3 = (labels == largest_label_by_area)
    ind4 = (labels != largest_label_by_area)
    frame_copy = np.copy(frame)
    frame_copy[:, :, 0][ind] = 0
    frame_copy[:, :, 1][ind] = 0
    frame_copy[:, :, 2][ind] = 0
    frame_copy[:, :, 0][ind4] = 0
    frame_copy[:, :, 1][ind4] = 0
    frame_copy[:, :, 2][ind4] = 0
    return frame_copy



def run_alignment(image_path):
    print(os.getcwd())
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    #print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


fullnames = ['/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/10020.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/1004.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/10043.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/1017.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/10222.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/10251.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/10301.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/1031.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/10645.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/11595.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/116.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/11705.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/11728.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/11736.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/11738.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/11781.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/11951.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/11984.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/12.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/42914.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/44278.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/44390.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/44443.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/4448.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/45245.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/45356.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/45404.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/45437.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/45974.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/46057.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/46064.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/46144.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/46178.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/46202.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/46291.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/46313.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/46338.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/47352.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/12537.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/12649.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/12734.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/12900.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/12901.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/13443.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/13459.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/13462.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/13505.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/1352.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/13542.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/13574.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/13598.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/13607.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/14518.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/14737.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/14782.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/14856.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/15420.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/15493.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/1550.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/15523.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/15559.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/15567.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/15657.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/15750.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/16092.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/16647.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/16829.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/17726.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/17734.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/1776.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/17783.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/17823.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/17827.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/18745.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/18797.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/18868.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/2005.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/20161.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/20271.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/20395.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/20400.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/20765.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/20815.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/2154.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/21574.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/21583.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/21606.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/21612.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/21648.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/21651.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/21662.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/21692.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/21715.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/21744.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/21745.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/12019.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/15439.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/20039.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/21793.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/26384.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/29578.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/33857.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/37049.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/4280.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/47409.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/49981.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/51393.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/56917.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/61528.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/6483.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/68235.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/8492.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/23026.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/23034.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/23040.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/23125.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/2326.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/23286.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/24419.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/24476.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/24563.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/24620.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/24681.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/25693.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/25757.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/2576.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/2578.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/2631.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/26342.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/26350.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/26377.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/26503.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/26542.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/27441.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/27455.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/27477.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/27503.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/27536.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/2754.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/27596.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/27620.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/27623.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/27658.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/27661.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/29421.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/29432.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/29444.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/29455.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/29478.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/2957.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/29608.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/29637.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/30578.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/30648.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/30662.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/30719.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/30755.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/30936.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/32039.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/32125.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/32192.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/32267.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/32295.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/32430.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/3346.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/33556.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/33586.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/33740.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/33748.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/33895.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/33908.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/34913.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/34932.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/35883.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/35886.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/35900.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/35913.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/35931.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/36002.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/36004.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/36048.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/36054.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/36136.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/36149.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/36151.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/36845.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/36857.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/36865.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/37078.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/37472.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/37522.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/37565.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/37606.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/37672.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/37689.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/37776.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/37841.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/38333.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/38558.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/39080.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/39199.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/39220.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/39276.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/4027.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/40296.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/41184.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/41198.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/41231.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/41282.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/41355.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/41631.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/41926.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/4206.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/42663.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/42676.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/42742.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/64906.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/65844.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/65856.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/65862.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/65940.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/65954.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/65967.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/65990.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/66032.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/66108.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/67034.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/67054.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/67139.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/67340.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/68080.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/68086.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/6811.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/68215.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/68225.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/47434.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/4749.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/47510.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/47529.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/47551.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/47563.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/47564.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/47567.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/4757.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/47607.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/48641.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/48706.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/48753.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/48762.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/48827.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/48888.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/48931.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/4996.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/49963.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/49989.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/50002.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/50014.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/50040.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/50070.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/50086.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/50719.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/508.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/50841.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/51166.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/51174.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/51195.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/51202.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/51230.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/51234.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/51259.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/51269.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/51295.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/51357.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/5154.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/5283.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/52986.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/53020.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/53061.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/53235.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/53335.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/53898.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/53904.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/53946.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/53960.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/53967.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/54075.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/54153.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/54156.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/54234.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/55307.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/56828.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/56868.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/56929.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/56993.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/57034.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/57541.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/57629.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/57645.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/57715.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/57773.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/5898.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/58992.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/59061.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/5909.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/60506.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/60619.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/60628.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/60632.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/60636.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/60811.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/61486.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/61581.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/61586.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/61602.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/61605.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/61617.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/61642.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/61647.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/61650.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/61656.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/61663.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/6167.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/62671.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/62680.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/62684.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/62688.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/62707.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/62798.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/62830.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/62937.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/63491.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/63571.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/63576.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/6364.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/63662.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/63679.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/63710.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/6374.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/68253.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/69353.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/69381.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/69403.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/69472.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/7098.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/7188.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/7233.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/7329.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/7335.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/735.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/7383.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/7384.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/7388.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/7421.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/8403.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/8430.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/8617.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/8639.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/8655.jpg',
             '/home/yakovdan/win/FinalProject/DataSet_all_aligned/Close/8671.jpg']

p1 = '/home/yakovdan/win/FinalProject/DataSet_all_aligned_and_segmented/Close'
p2 = '/home/yakovdan/win/FinalProject/DataSet_all_aligned_and_segmented/Close_val'
p3 = '/home/yakovdan/win/FinalProject/DataSet_all_aligned_and_segmented/Open'
p4 = '/home/yakovdan/win/FinalProject/DataSet_all_aligned_and_segmented/Open_val'
for pname in [p1, p2, p3, p4]:
    os.chdir(pname)
    filenames = os.listdir()
    for num, name in enumerate(filenames):
        try:
            result = run_alignment(name)
            open_cv_image = np.array(result)
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            segmented_opencv_image = segment_face(open_cv_image)
            fname = f"image_{num}.jpeg"
            cv2.imwrite(f"./ready/{fname}", segmented_opencv_image)
            print(f"Finished: {num}")
        except Exception:
            print(f"Failed aligning {num}, {name}")
            print(str(Exception))