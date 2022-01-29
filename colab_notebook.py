from PIL import Image
from typing import List, Optional
import dnnlib
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
import torch.nn as nn
from torchvision import datasets, models, transforms
from e4e_editings import latent_editor # note: e4e editings, not restyle editings
from utils.inference_utils import run_on_batch
import legacy
# used seeds: [0, 42]
torch.manual_seed(99)
np.random.seed(99)

ganspace_pca = torch.load('ganspace_pca/ffhq_pca.pt')

ROOT_DIR = os.getcwd()
CODE_DIR = ROOT_DIR+'/restyle_encoder'
sys.path.append(CODE_DIR)
os.chdir(f'{CODE_DIR}')

from restyle_encoder.utils.common import tensor2im
from restyle_encoder.models.psp import pSp
from restyle_encoder.models.e4e import e4e
import dlib
import re
from scripts.align_faces_parallel import align_face

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
    frame = cv2.resize(frame, dsize=(256, 256), interpolation=cv2.INTER_LANCZOS4)
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

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def generate_images(classification_model, num_images,
                    network_pkl: str = '/home/yakovdan/win/FinalProject/pretrained/ffhq.pkl',
                    truncation_psi: float = 1,
                    noise_mode: str = "const",  # ['const', 'random', 'none']
                    outdir: str = "/home/yakovdan/win/stylegan2_output_2901_0100/"):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    softmax = nn.Softmax(dim = 1)
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.

    label = torch.zeros([1, G.c_dim], device=device)
    latents_highest_conf = []
    pred_vals_highest_conf = []
    all_latents = []
    all_w = []
    all_pred_vals = []

    # Generate images.
    #np.random.RandomState(seeds[image_idx % len(seeds)])
    #np.random.RandomState(97)
    image_idx = 0
    total_tries = 0
    actual_tries = 0
    while image_idx < num_images: # and total_tries < 10000:
        print(f"Generating image number {image_idx} out of {total_tries}")
        z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
        w_samples = G.mapping(z, None)  # [N, L, C]
        img_test = G.synthesis(w_samples, noise_mode='const')

        # img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        # img = img[0].cpu().numpy()
        # temp_image = Image.fromarray(img, 'RGB')
        # temp_image.save("tempimage.jpg", "JPEG")

        img_test = (img_test.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_test = img_test[0].cpu().numpy()
        #temp_image_test = Image.fromarray(img_test, 'RGB')
        #temp_image_test.save("tempimage.jpg", "JPEG")


        try:
            aligned_image = align_face(img_test, predictor=predictor)
            open_cv_image_aligned = np.array(aligned_image)
            # Convert RGB to BGR
            open_cv_image_aligned = open_cv_image_aligned[:, :, ::-1].copy()
            seg_image = segment_face(open_cv_image_aligned)
        except BaseException as err:
            print(f"Unexpected {err}, {type(err)}")
            total_tries += 1
            continue
        seg_image = seg_image[:, :, ::-1].copy()
        seg_image = Image.fromarray(seg_image, 'RGB')
        open_cv_image = np.array(seg_image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        cls_input = classifier_transforms(seg_image).reshape((1, 3, 224, 224)).to(device)
        logits_result = classification_model(cls_input)
        result = softmax(logits_result)
        pred_val, pred = torch.max(result, 1)
        pred = pred.cpu().item()
        pred_val = pred_val.cpu().detach().item()
        total_tries += 1
        actual_tries += 1
        all_latents.append(z.cpu().detach().numpy())
        all_w.append(w_samples.cpu().detach().numpy())
        if pred == 0:
            all_pred_vals.append(1-pred_val)
            cv2.imwrite(f'{outdir}/closed/image_{total_tries:05d}_{actual_tries:05d}.jpg', open_cv_image)
        else:
            cv2.imwrite(f'{outdir}/open/image_{total_tries:05d}_{actual_tries:05d}.jpg', open_cv_image)
            all_pred_vals.append(pred_val)

        if total_tries > 0 and total_tries % 500 == 0:
            w_np = np.concatenate(all_w)
            latents_np = np.concatenate(all_latents)
            predvals_np = np.array(all_pred_vals)
            np.save(f'{outdir}/latents_{total_tries}', latents_np)
            np.save(f'{outdir}/predvals_{total_tries}', predvals_np)
            np.save(f'{outdir}/w_{total_tries}', w_np)
        if pred == 0 and pred_val > 0.97:
            latents_highest_conf.append(z.cpu().detach().numpy())
            pred_vals_highest_conf.append(pred_val)
            cv2.imwrite(f'{outdir}/highconf_closed/image_{image_idx:05d}_{total_tries:05d}.jpg', open_cv_image)
            image_idx += 1

            latents_np = np.concatenate(latents_highest_conf)
            predvals_np = np.array(pred_vals_highest_conf)
            np.save(f'{outdir}/highconf_closed/latents_test{image_idx}', latents_np)
            np.save(f'{outdir}/highconf_closed/predvals_{image_idx}', predvals_np)


def displayNP(img_np):
    cv2.imshow('test', img_np)
    cv2.waitKey(0)


def displayPIL(p):
    img_np = np.array(p)
    displayNP(img_np)


def fit_parabola(ecc_list):
    l = len(ecc_list)
    ecc_list_np = np.array(ecc_list).reshape(l,1)
    x = np.array(list(range(len(ecc_list)))).reshape(l, 1)
    x2 =np.array([t ** 2 for t in x]).reshape(l, 1)
    ones = np.ones_like(x)
    mat = np.concatenate([x2, x, ones], axis=1)
    sol = np.linalg.lstsq(mat, ecc_list_np, rcond='warn')
    return sol




def compare_sizes(item1, item2):
    x1, y1 = item1[0]
    x2, y2 = item1[1]
    diag1 = (x2-x1)**2 + (y2-y1)**2
    x3, y3 = item2[0]
    x4, y4 = item2[1]
    diag2 = (x4 - x3) ** 2 + (y4 - y3) ** 2
    return diag1 > diag2


def plot_ellipse(x_coordinates, y_coordinates, z_coordinates):
    plt.contour(x_coordinates, y_coordinates, z_coordinates, levels=[1], colors='r', linewidths=2)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')


def extract_left_eye_landmark(img_gray_list):
    landmarks_list = []
    rects = detector(img_gray_list[0], 1)
    # sometimes more than one rectangle is returned. select the largest
    if len(rects) == 0:
        return landmarks_list

    if len(rects) > 1:
        sorted(rects, key=lambda r: r.area(), reverse=True)
        rects = rects[0:1]
    rect = rects[0]
    for gray_image in img_gray_list:
        left_eye_landmarks_list = []

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
    sol = fit_parabola(ecc_list)
    a, b, c = sol[0]
    vertex = -b / (2*a)
    min_idx = np.argmin(np.array(ecc_list))
    print(f"vertex: {vertex}, min_idx: {min_idx}")
    return vertex > 0, vertex, min_idx



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


def close_eyes_by_amount(editor, ganspace_pca, latents, steps):
    directions = [(54,  7,  8,  20+s) for s in steps]
    image_eyesclosed = editor.apply_ganspace([torch.from_numpy(latents[0][-1]).cuda()], ganspace_pca, directions)
    image_eyesclosed_np = [cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR) for x in image_eyesclosed]
    image_eyesclosed_np_gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in image_eyesclosed_np]

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
    filename = f'../SourceImages/{image_id:05d}.jpg'
    aligned_PIL_image = Image.open(filename)
    transformed_PIL_image = img_transforms(aligned_PIL_image)
    # perform inversion
    with torch.no_grad():

        tic = time.time()
        result_batch, result_latents = run_on_batch(transformed_PIL_image.unsqueeze(0).cuda(), net, opts, avg_image)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))
    # compute images with closed eyes
    image_list = []
    image_list_color = []
    tic = time.time()
    steps = list(range(0, 80, 3))
    for step_idx in range(0, len(steps), 4):
        steps1 = steps[step_idx:step_idx+4]
        closed_eyes_image_color_np, close_eyes_image_gray_np = close_eyes_by_amount(editor, ganspace_pca, result_latents, steps1)
        image_list.extend(close_eyes_image_gray_np)
        image_list_color.extend(closed_eyes_image_color_np)
    # if debug:
    #     cv2.imwrite(f'gray_amount_{step}.jpg', close_eyes_image_gray_np)


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
        plt.contour(x, y, z, levels=[1], colors=('r'), linewidths=2)
        ecc_list.append(w[0]/w[1])
    toc = time.time()
    print('Processing landmarks took {:.4f} seconds.'.format(toc - tic))
    tic = time.time()
    error_code = 0
    if len(ecc_list) < 20:
        output_image = tensor2im(transformed_PIL_image)
        output_image.save(f"../DataSet_run2/Errors/1/{image_id:05d}.jpg")
        error_code, v, i = 1, -1, -1
    else:
        convex_status, v, i = estimate_convexity(ecc_list)
        if not convex_status:
            output_image = tensor2im(transformed_PIL_image)
            output_image.save(f"../DataSet_run2/Errors/2/{image_id:05d}.jpg")
            error_code = 2
        else:
            min_idx = np.argmin(ecc_list) - 2
            output_image = tensor2im(result_batch[0][-1])
            output_image.save(f"../DataSet_run2/Open/{image_id}.jpg")
            cv2.imwrite(f"../DataSet_run2/Close/{image_id}.jpg", image_list_color[min_idx])
            error_code = 0
    toc = time.time()
    print('File I/O took {:.4f} seconds.'.format(toc - tic))
    return error_code, v, i

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
avg_image = get_avg_image(net)
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


#################################
# GENERATE CLOSED EYES PICTURE  #
#################################

# count_good = 0
# count_bad = 0
#
# start_idx  = 0
# end_idx = 1000
# if len(sys.argv) == 3:
#     start_idx = int(sys.argv[1])
#     end_idx = int(sys.argv[2])
# a = os.getcwd()
# a = a +"/../SourceImages/idlist.txt"
#
# with open(a,'r') as idfile:
#     file_ids = idfile.readlines()
# file_ids = [int(x) for x in file_ids][6750:]
# for image_counter, idx in enumerate(file_ids):#range(start_idx, end_idx):
#     tic = time.time()
#     status, vertex, min_idx = process_image_by_id(editor, idx, debug=True)
#     if status == 0:
#         count_good += 1
#     else:
#         count_bad += 1
#     with open('summary.txt', 'a') as summary_file:
#         summary_file.write(f"{idx}: {str(status)}, vertex : {vertex}, min_idx: {min_idx}\n")
#     toc = time.time()
#     print(idx, status, (toc-tic), 100*count_bad/(image_counter+1), image_counter+1)

############################
# SAMPLE FROM LATENT SPACE #
############################
#Initialize the model for this run
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft, input_size = initialize_model("resnet", 2, True, use_pretrained=True)
data_dict = torch.load("/home/yakovdan/win/FinalProject/DataSet_all_aligned_and_segmented/best_classifier.pt")
model_ft.load_state_dict(data_dict['state_dict'])
model_ft = model_ft.to(device)
model_ft.eval()
classifier_transforms = transforms.Compose([transforms.Resize(input_size),
                                            transforms.CenterCrop(input_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


#test_img = cv2.imread("/home/yakovdan/win/FinalProject/"+'image_00103.jpg')
generate_images(model_ft, 500)
# with dnnlib.util.open_url('/home/yakovdan/win/FinalProject/pretrained/ffhq.pkl') as f:
#     G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
#
# label = torch.zeros([1, G.c_dim], device=device)
# all_w = np.load('/home/yakovdan/win/FinalProject/w_40000.npy')
# boundary = torch.tensor(np.load('/home/yakovdan/win/FinalProject/boundary.npy')).cuda()
# w_idx = 33780
# for factor in range(0, 20, 1):
#     w = torch.tensor(all_w[w_idx, :]).float().cuda().reshape([1, 18, 512])
#     w = w + factor*boundary
#     img_test = G.synthesis(w, noise_mode='const')
#     img_test = (img_test.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#     img_test = img_test[0].cpu().numpy()
#     temp_image_test = Image.fromarray(img_test, 'RGB')
#     temp_image_test.save(f"/home/yakovdan/win/test_output/test_{w_idx}_{factor}.jpeg", "JPEG")

# done_flag = False
# for idx in range(1000):
#     z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
#     img = G(z, label, truncation_psi=1, noise_mode="const")
#     img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#     img = img[0].cpu().numpy()
#     temp_image = Image.fromarray(img, 'RGB')
#     temp_image.save(f"/home/yakovdan/win/FinalProject/images_for_test/image_{idx}.jpg")
#     z1 = z.cpu().numpy()
#     np.save(f'/home/yakovdan/win/FinalProject/images_for_test/latent_{idx}', z1)
# z1 = z.cpu().numpy()
# np.save('latent_test2', z1)

# z2 = torch.tensor(np.load('/home/yakovdan/win/FinalProject/latent_164.npy')).cuda()
# z3 = z2 + 5 * boundary
# torch.manual_seed(100)
# np.random.seed(100)
#
# img = G(z2, label, truncation_psi=1, noise_mode="const")
# img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
# img = img[0].cpu().numpy()
# temp_image = Image.fromarray(img, 'RGB')
# plt.imshow(temp_image)
# plt.show()
#
# img = G(z3, label, truncation_psi=1, noise_mode="const")
# img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
# img = img[0].cpu().numpy()
# temp_image = Image.fromarray(img, 'RGB')
# plt.imshow(temp_image)
# plt.show()
#
# print("done")
###########################################
# Open Eyes Using PCA
############################################

# image_yakov = run_alignment('/home/yakovdan/win/FinalProject/yakov_closed_eyes.jpg')
# image_bibi = run_alignment('/home/yakovdan/win/FinalProject/bibi.jpg')
# #
# #
# with torch.no_grad():
#     tic = time.time()
#     result_batch_yakov, result_latents_yakov = run_on_batch(img_transforms(image_yakov).unsqueeze(0).cuda(), net, opts, avg_image)
#     toc = time.time()
#     print('Inference took {:.4f} seconds.'.format(toc - tic))
#
# with torch.no_grad():
#     tic = time.time()
#     result_batch_bibi, result_latents_bibi = run_on_batch(img_transforms(image_bibi).unsqueeze(0).cuda(), net, opts, avg_image)
#     toc = time.time()
#     print('Inference took {:.4f} seconds.'.format(toc - tic))

# np.save('latent_bibi', result_latents_bibi[0])
# np.save('latent_yakov', result_latents_yakov[0])
#
# for v in np.arange(0, -80, -10):
#     directions = [(54,  7,  8,  20+s) for s in [v]]
#     image_eyesclosed = editor.apply_ganspace([torch.from_numpy(result_latents_yakov[0][-1]).cuda()], ganspace_pca, directions)
#     plt.imshow(image_eyesclosed[0])
#     plt.show()



# plt.imshow(tensor2im(result_batch_trump[0][-1]))
# plt.show()
# plt.imshow(tensor2im(result_batch_yakov[0][-1]))
# plt.show()

################################
# Generate Images from latents #
##################################
# latents = np.load('/home/yakovdan/win/FinalProject/latents_b.npy')
# #vec = np.load('/home/yakovdan/win/FinalProject/latents_b.npy')[-1]
# for i, vec in enumerate(latents):
#     print(i)
#     vec1 = vec.reshape((1, 512))
#     image = editor._latents_to_image(torch.tensor(vec1).float().to("cuda"))
#     plt.imshow(image[0])
#     plt.show()