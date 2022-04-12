import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
from torchvision.transforms import transforms as transforms
import os
import pandas as pd
from utils import coco_names
import imutils
import time

class maskR_CNN:

    def __init__(self, file_path, output_folder, coco_names, threshold, colors):
        self.file_path = file_path
        self.coco_names = coco_names
        self.threshold = threshold
        self.output_folder = output_folder
        self.colors = colors

    def get_file(self):
        if self.file_path.split('.')[1] in ['jpg', 'jpeg', 'png']:
            image = Image.open(file_path).convert('RGB')
            return image

        else:
            self.file = cv2.VideoCapture(file_path)
            return self

    def mask_r_cnn_model(self):
        self.model = maskrcnn_resnet50_fpn(pretrained=True, progress=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()
        return self

    def get_outputs(self, image):
        with torch.no_grad():
            outputs = self.model(image)
        
        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        scores = [round(score, 3) for score in scores]

        thresholded_preds_inidices = [scores.index(i) for i in scores if i > self.threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)

        masks = (outputs[0]['masks'] > self.threshold).squeeze().detach().cpu().numpy()

        masks = masks[:thresholded_preds_count]

        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]

        boxes = boxes[:thresholded_preds_count]

        labels = [self.coco_names[i] for i in outputs[0]['labels']]
        self.class_ids = [self.coco_names.index(label) for label in labels]
        return masks, boxes, labels, scores

    def draw_segmentation_map(self, image, masks, boxes, labels, scores):
        alpha = 1 
        beta = 0.4
        gamma = 0

        for i in range(len(masks)):
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)

            color = [int(c) for c in self.colors[self.class_ids[i]]]
            confidence = str(scores[i])

            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)

            image = np.array(image)

            cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, thickness=2)

            cv2.rectangle(image, (boxes[i][0][0], boxes[i][0][1]), 
                    (boxes[i][0][0]+200, boxes[i][0][1]-50), 
                    color, -1)

            cv2.putText(image, "{} {}".format(labels[i], confidence), 
                        (boxes[i][0][0], boxes[i][0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 
                        thickness=2, lineType=cv2.LINE_AA)
        return image

    def apply_segmentation(self, image):
        transform = transforms.Compose([
        transforms.ToTensor()
        ])

        orig_image = image.copy()
        image = transform(image)
        image = image.unsqueeze(0).to(self.device)
        masks, boxes, labels, scores = self.get_outputs(image)
        self.result = self.draw_segmentation_map(orig_image, masks, boxes, labels, scores)
    
    def write_image_on_directory(self):
        self.result = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        save_path = f"{self.output_folder}/{self.file_path.split('/')[-1].split('.')[0]}_mask_rcnn_detection.jpg"
        cv2.imwrite(save_path, self.result)

    def run_on_video(self):
        self.get_file()
        self.mask_r_cnn_model()

        writer = None

        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
            total = int(self.file.get(prop))
            print("[INFO] {} total frames in video".format(total))

        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total = -1

        # While capturing video
        while True:
            (grabbed, frame) = self.file.read()

            if not grabbed:
                break

            frame = Image.fromarray(frame)

            self.mask_r_cnn_model()
            self.apply_segmentation(frame)
                    
            # Write on video frame
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                
                writer = cv2.VideoWriter(output_folder + '/' + 
                self.file_path.split('/')[-1].split('.')[0] +'mask_rcnn_detection.avi', fourcc, 30,
                (self.result.shape[1], self.result.shape[0]), True)

            writer.write(self.result)

        writer.release()
        self.file.release()

if __name__ == '__main__':
    file_folder = 'utils/'
    output_folder = 'output'
    file_name = os.listdir(file_folder)[0]
    file_path = file_folder + file_name
    colors = np.random.randint(0, 255, size=(len(coco_names), 3), dtype="uint8")

    threshold = 0.85

    print('Starting Detection...')
    start = time.time()

    # Images
    mask_rcnn = maskR_CNN(file_path, output_folder, coco_names, threshold, colors)
    image = mask_rcnn.get_file()
    mask_rcnn.mask_r_cnn_model()
    mask_rcnn.apply_segmentation(image)
    mask_rcnn.write_image_on_directory()

    # Video
    # mask_rcnn = maskR_CNN(file_path, output_folder, coco_names, threshold, colors)
    # mask_rcnn.run_on_video()

    end = time.time()
    print('Success!')
    print(f"Operation took {end-start} seconds")