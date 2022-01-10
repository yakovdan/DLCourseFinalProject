import cv2
import os
from imutils import face_utils
import matplotlib.pyplot as plt
from argparse import Namespace
import time
import sys
import pprint
import numpy as np
import torch
import torchvision.transforms as transforms
from e4e_editings import latent_editor # note: e4e editings, not restyle editings
from utils.inference_utils import run_on_batch

ganspace_pca = torch.load('ganspace_pca/ffhq_pca.pt')

ROOT_DIR = os.getcwd()
CODE_DIR = ROOT_DIR+'/restyle_encoder'
sys.path.append(CODE_DIR)
os.chdir(f'{CODE_DIR}')

from restyle_encoder.utils.common import tensor2im
from restyle_encoder.models.psp import pSp
from restyle_encoder.models.e4e import e4e
import dlib
from scripts.align_faces_parallel import align_face


def plot_ellipse(x_coordinates, y_coordinates, z_coordinates):
    plt.contour(x_coordinates, y_coordinates, z_coordinates, levels=[1], colors='r', linewidths=2)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')


def extract_left_eye_landmark(img_gray_list):
    landmarks_list = []
    for gray_image in img_gray_list:
        rects = detector(gray_image, 1)
        left_eye_landmarks_list = []
        for (i, rect) in enumerate(rects):
            shape = predictor(gray_image, rect)
            shape = face_utils.shape_to_np(shape)
            left_eye = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
            shape = shape[left_eye[0]:left_eye[1]]
            for j, (x, y) in enumerate(shape):
                left_eye_landmarks_list.append((x, y))
            landmarks_list.append(left_eye_landmarks_list)
    return landmarks_list


def transform_coordinate_system(points_list, vertical_dimension, horizontal_dimension):
    rotated_list = [(vertical_dimension - x, y) for x, y in points_list]
    shifted_origin_list = [(x - horizontal_dimension // 2, y - vertical_dimension // 2) for x, y in rotated_list]
    return shifted_origin_list

def map_to_center(points_list):
    x_values = [x for x, _ in points_list]
    y_values = [y for _, y in points_list]
    x_center = np.mean(x_values)
    y_center = np.mean(y_values)
    centered_list = []
    for x, y in points_list:
        centered_list.append((x - x_center, y - y_center))
    return centered_list, x_center, y_center


def coeffs_to_mat(coeffs):
    A, B, C, D, E, F = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], -1
    #mat_form = [[A, B/2, D/2], [B/2, C, E/2], [D/2, E/2, F]]
    mat_form = [[A, B/2], [B/2, C]]
    return np.array(mat_form)


def fit_ellipse_to_eye(points_list):
    # Extract x coords and y coords of the ellipse as column vectors
    X = np.array([x for x, _ in points_list]).reshape((6, 1))
    Y = np.array([y for _, y in points_list]).reshape((6, 1))

    # Formulate and solve the least squares problem ||Ax - b ||^2
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b)[0].squeeze()

    # Print the equation of the ellipse in standard form
    #print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1],x[2],x[3],x[4]))

    # Plot the noisy data
    #plt.scatter(X, Y, label='Data Points')


    # Plot the least squares ellipse
    x_coord = np.linspace(-100, 100, 300)
    y_coord = np.linspace(-25, 20, 300)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord

    mat = coeffs_to_mat(x)

    return X_coord, Y_coord, Z_coord, mat


def derivative(points):
    if len(points) < 3:
        return []
    derivate_list = []
    for idx in range(1, len(points)-1):
        value = (points[idx+1] - points[idx-1]) / 2
        derivate_list.append(value)
    return derivate_list

def estimate_convexity(ecc_list):
    der1 = derivative(ecc_list)
    der2 = derivative(der1)
    der2_np_abs = np.abs(np.array(der2))
    q1, q9 = np.quantile(der2_np_abs, [0.1, 0.9])
    a = (der2_np_abs > q1)
    b = (der2_np_abs < q9)
    c = a & b
    der2_filt = der2_np_abs[c]
    normalized_sum = 100 * np.sum(der2_filt) / len(der2_filt)
    return normalized_sum < 0.5



# this function takes an image and an encoder as parameters
# and returns a latent vector

def image_to_latent(net, opts, input_img):
    with torch.no_grad():
        avg_image = get_avg_image(net)
        tic = time.time()
        result_batch, result_latents = run_on_batch(input_img.unsqueeze(0).cuda(), net, opts, avg_image)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))
    return result_batch, result_latents


def close_eyes_by_amount(editor, ganspace_pca, latents, step):
    directions = {
        'eye_openness':            (54,  7,  8,  20+step),
        'smile':                   (46,  4,  5, -25),
        'trimmed_beard':           (58,  7,  9,  20),
        'white_hair':              (57,  7, 10, -24),
        'lipstick':                (34, 10, 11,  20)
    }

    image_eyesclosed = editor.apply_ganspace([torch.from_numpy(latents[0][-1]).cuda()], ganspace_pca, [directions["eye_openness"]])
    image_eyesclosed_np = cv2.cvtColor(np.array(image_eyesclosed), cv2.COLOR_RGB2BGR)
    image_eyesclosed_np_gray = cv2.cvtColor(image_eyesclosed_np, cv2.COLOR_BGR2GRAY)
    return image_eyesclosed_np, image_eyesclosed_np_gray


def get_coupled_results(result_batch, transformed_image, resize_amount):
    """
    Visualize output images from left to right (the input image is on the right)
    """
    result_tensors = result_batch[0]  # there's one image in our batch
    result_images = [tensor2im(result_tensors[iter_idx]) for iter_idx in range(opts.n_iters_per_batch)]
    input_im = tensor2im(transformed_image)
    res = np.array(result_images[0].resize(resize_amount))
    for idx, result in enumerate(result_images[1:]):
        res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
    res = np.concatenate([res, input_im.resize(resize_amount)], axis=1)
    return res


def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image



def run_alignment(image_path):
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

# id in [37000, 37999]
def process_image_by_id(editor, image_id, debug=False):
    filename = f'../FFHQ/37000/{str(image_id)}.jpg'
    aligned_PIL_image = run_alignment(filename)
    transformed_PIL_image = img_transforms(aligned_PIL_image)
    # perform inversion
    with torch.no_grad():
        avg_image = get_avg_image(net)
        tic = time.time()
        result_batch, result_latents = run_on_batch(transformed_PIL_image.unsqueeze(0).cuda(), net, opts, avg_image)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))
    # compute images with closed eyes
    image_list = []
    image_list_color = []
    tic = time.time()
    for step in range(0, 80, 3):
        closed_eyes_image_color_np, close_eyes_image_gray_np = close_eyes_by_amount(editor, ganspace_pca, result_latents, step)
        if debug:
            cv2.imwrite(f'gray_amount_{step}.jpg', close_eyes_image_gray_np)
        image_list.append(close_eyes_image_gray_np)
        image_list_color.append(closed_eyes_image_color_np)

    toc = time.time()
    print('Closing steps took {:.4f} seconds.'.format(toc - tic))
    tic = time.time()
    all_landmarks = extract_left_eye_landmark(image_list)
    toc = time.time()
    print('Extracting left eye landmarks took {:.4f} seconds.'.format(toc - tic))
    ecc_list = []
    tic = time.time()
    for landmark_list in all_landmarks:
        if len(landmark_list) != 6:
            continue
        centered, x_center, y_center = map_to_center(landmark_list)
        x, y, z, mat = fit_ellipse_to_eye(centered)
        w, _ = np.linalg.eig(mat)
        ecc_list.append(w[0]/w[1])
    toc = time.time()
    print('Processing landmarks took {:.4f} seconds.'.format(toc - tic))
    tic = time.time()
    error_code = 0
    if len(ecc_list) < 20:
        output_image = tensor2im(transformed_PIL_image)
        output_image.save(f"../DataSet/Errors/{image_id}.jpg")
        error_code = 1
    elif not estimate_convexity(ecc_list):
        output_image = tensor2im(transformed_PIL_image)
        output_image.save(f"../DataSet/Errors/{image_id}.jpg")
        error_code = 2
    else:
        min_idx = np.argmin(ecc_list) - 2
        output_image = tensor2im(result_batch[0][-1])
        output_image.save(f"../DataSet/Open/{image_id}.jpg")
        cv2.imwrite(f"../DataSet/Close/{image_id}.jpg", image_list_color[min_idx])
        error_code = 0
    toc = time.time()
    print('File I/O took {:.4f} seconds.'.format(toc - tic))
    return error_code

experiment_type = 'ffhq_encode'


EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "pretrained_models/restyle_psp_ffhq_encode.pt",
        "image_path": "input_ffhq1.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }

}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]


model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')

opts = ckpt['opts']
pprint.pprint(opts)


# update the training options
opts['checkpoint_path'] = model_path


opts = Namespace(**opts)
if experiment_type == 'horse_encode':
    net = e4e(opts)
else:
    net = pSp(opts)

net.eval()
net.cuda()
print('Model successfully loaded!')
editor = latent_editor.LatentEditor(net.decoder, False)

image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
img_transforms = EXPERIMENT_ARGS['transform']
# original_image = Image.open(image_path).convert("RGB")
# original_image = original_image.resize((256, 256))
# input_image = run_alignment(image_path)
# transformed_image = img_transforms(input_image)

opts.n_iters_per_batch = 5
opts.resize_outputs = False  # generate outputs at full resolution


# with torch.no_grad():
#     avg_image = get_avg_image(net)
#     tic = time.time()
#     result_batch, result_latents = run_on_batch(transformed_image.unsqueeze(0).cuda(), net, opts, avg_image)
#     toc = time.time()
#     print('Inference took {:.4f} seconds.'.format(toc - tic))
#
# resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
# res = get_coupled_results(result_batch, transformed_image, resize_amount)


FFHQ_PATH = ROOT_DIR + "/FFHQ/"
image_directory = "37000/"
full_image_directory = FFHQ_PATH+image_directory
#
# Coordinate transformation. In OpenCV, the coordinate system is (y, x)  - first axis is "down" amd the second is "right", the origin is at top left corner.
#
# We want to transform the coordinate system to a cartesian system with the origin at the center - (x, y) such that x goes right and y goes up.
#
# So, first we flip vertically and then we change order of coordinates
#

#

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def plot_ellipse(x_coordinates, y_coordinates, z_coordinates):
    plt.contour(x_coordinates, y_coordinates, z_coordinates, levels=[1], colors='r', linewidths=2)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')


# def extract_left_eye_landmark(img_gray_list):
#     landmarks_list = []
#     for gray_image in img_gray_list:
#         rects = detector(gray_image, 1)
#         left_eye_landmarks_list = []
#         for (i, rect) in enumerate(rects):
#             shape = predictor(gray_image, rect)
#             shape = face_utils.shape_to_np(shape)
#             left_eye = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
#             for j, (x, y) in enumerate(shape):
#                 if j in range(left_eye[0], left_eye[1]):
#                     #cv2.circle(image_eyesopened_np, (x, y), 1, (0, 255, 0), -1)
#                     left_eye_landmarks_list.append((x, y))
#             landmarks_list.append(left_eye_landmarks_list)
#     return landmarks_list

# KNONW BAD: 122


for idx in range(0, 1000):
    tic = time.time()
    status = process_image_by_id(editor, 37000+idx)
    with open('summary.txt', 'a') as summary_file:
        summary_file.write(f"{idx}: {str(status)}\n")
    toc = time.time()
    print(idx, status, (toc-tic))


