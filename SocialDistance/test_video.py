from __future__ import absolute_import, division, print_function

import numpy as np
import PIL.Image as pil
import cv2
import torch
from torchvision import transforms
from gluoncv import model_zoo, data
import time
import mxnet as mx
import networks
from scipy.spatial.distance import euclidean
import argparse


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i",   "--input", required=True,  help="Path to input",      default='video.mp4', type=str)
ap.add_argument("-st",  "--score", required=False, help="Scores Threshold",   default=float(0.5),  type=float)
ap.add_argument("-dt",  "--dist",  required=False, help="Distance Threshold", default=int(50),     type=int)
ap.add_argument("-dpt", "--depth", required=False, help="Depth Threshold",    default=float(0.1),  type=float)
args = vars(ap.parse_args())


###############################################################################

# Draw bounding box for NO SOCIAL DISTANCE
def draw_bbox(img, disp_resized_np, bounding_boxs, scores):
    
    
    bboxs = []
    persons = []
    
    
    # Check if numpy array
    if isinstance(bounding_boxs[0], mx.nd.NDArray):
        bboxs = bounding_boxs[0].asnumpy()
    if isinstance(scores[0], mx.nd.NDArray):
        scores = scores[0].asnumpy()
        
    
    # Get box coordinates for scores above threshold
    bounding_boxs = []
    for i, bbox in enumerate(bboxs):
        if scores[i] > args["score"]:
            xmin, ymin, xmax, ymax = [int(x) for x in bbox]
            if all(x >= 0 for x in [xmin, ymin, xmax, ymax]):
                bounding_boxs.append([xmin, ymin, xmax, ymax])
    
    
    # Draw bounding box
    for i in range(len(bounding_boxs)):
        if i not in persons:
            for j in range(len(bounding_boxs)):
                if i!= j:
                        
                    xmin,ymin,xmax,ymax = bounding_boxs[i]
                    center1 = ((xmin+xmax)/2, (ymin+ymax)/2)
                    depth1= np.average(disp_resized_np[ymin:ymax,xmin:xmax])
                    
                    xmin,ymin,xmax,ymax = bounding_boxs[j]
                    center2 = ((xmin+xmax)/2, (ymin+ymax)/2)
                    depth2= np.average(disp_resized_np[ymin:ymax,xmin:xmax])
                    
                    depth = abs(depth1-depth2)
                    dist = euclidean(center1, center2)

                    if depth < args["depth"]:
                        if dist < args["dist"]:
                        
                            xmin,ymin,xmax,ymax = bounding_boxs[i]
                            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                            
                            xmin,ymin,xmax,ymax = bounding_boxs[j]
                            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                            
                            persons.append(i)
                            persons.append(j)
                            
                            break
    

###############################################################################

# Load Pretrained model

encoder_path = "encoder.pth"
depth_decoder_path = "depth.pth"

encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval()


net = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
net.reset_class(["person"], reuse_weights=['person'])


###############################################################################


# Read video
video = cv2.VideoCapture(args["input"])

out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 192))

# Exit if video not opened
if not video.isOpened():
    print("Could not open video")

# Read first frame
ok, input_image = video.read()
if not ok:
    print('Cannot read video file')
frame_count = 0


while True:
    
    frame_count += 1
    print('Frame: {}'.format(frame_count))
    
    # Read a new frame
    start_time = time.time()
    ok, input_image = video.read()
    if not ok:
        break
    
    
    # Pass frame to network
    x, img = data.transforms.presets.ssd.transform_test(mx.nd.array(input_image),short=512)
    class_IDs, scores, bounding_boxs = net(x)
    
    img_copy = input_image.copy()
    input_image = pil.fromarray(input_image)
    original_width, original_height = input_image.size
    
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    
    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
    
    
    
    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = depth_decoder(features)
    
    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(disp,
        (original_height, original_width), mode="bilinear", align_corners=False)
    
    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    
    
    # Draw bounding box for NO SOCIAL DISTANCE
    draw_bbox(img, disp_resized_np, bounding_boxs, scores)
    
    
    # Show and write frames
    cv2.namedWindow('Depth_information', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Depth_information', 500, 300)
    disp_resized_np = np.array(disp_resized_np * 255, dtype = np.uint8)
    disp_resized_np = cv2.applyColorMap(disp_resized_np, cv2.COLORMAP_BONE)
    cv2.imshow('Depth_information',disp_resized_np)
    cv2.namedWindow('Detect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detect', 500, 300)
    cv2.imshow('Detect',img)
    out.write(img)
    
    # Break if ESC pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break



out.release()
video.release()
cv2.destroyAllWindows()  