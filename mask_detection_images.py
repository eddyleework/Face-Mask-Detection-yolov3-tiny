import cv2
import numpy as np
import os

confThreshold = 0.5
nmsThreshold = 0.5
font = cv2.FONT_HERSHEY_PLAIN
classes = ['no-mask', 'mask']

net = cv2.dnn.readNet('weight_cfg_dir/yolov3_custom.weights', 'weight_cfg_dir/yolov3_custom.cfg')
# net = cv2.dnn.readNet('weight_cfg_dir/yolov3_tiny_custom.weights', 'weight_cfg_dir/yolov3_tiny_custom.cfg')

path = 'images_dir'


def main():
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        img = cv2.resize(img, (900, 600))

        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)

        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                id = np.argmax(scores)
                confidence = scores[id]
                if confidence > confThreshold:
                    w, h = (int(detection[2] * width), int(detection[3] * height))
                    x, y = (int((detection[0] * width) - w / 2), int((detection[1] * height) - h / 2))

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str((int(confidences[i] * 100)))

                if label == 'no-mask':
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.putText(img, label + " " + confidence + '%', (x, y - 8), font, 2, (0, 0, 255), 1)

                if label == 'mask':
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
                    cv2.putText(img, label + " " + confidence + '%', (x, y - 18), font, 2, (255, 255, 255), 1)

        cv2.imshow('Image', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
