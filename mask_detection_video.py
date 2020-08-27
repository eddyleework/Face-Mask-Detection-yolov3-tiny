import cv2
import numpy as np
import queue
import threading
import time
import pyttsx3

confThreshold = 0.4
nmsThreshold = 0.4
cap = cv2.VideoCapture('video_dir/mask_test_1.mp4')  # video file
# cap = cv2.VideoCapture(0)                          # webcam

font = cv2.FONT_HERSHEY_PLAIN
queue_warn = queue.Queue(1)
classes = ['no-mask', 'mask']
maskCheck = None

# net = cv2.dnn.readNet('weight_cfg_dir/yolov3_custom.weights', 'weight_cfg_dir/yolov3_custom.cfg')   # yolov3
net = cv2.dnn.readNet('weight_cfg_dir/yolov3_tiny_custom.weights', 'weight_cfg_dir/yolov3_tiny_custom.cfg')  #yol0v3 tiny


def maskThread():
    while True:
        if (queue_warn.get()):
            engine = pyttsx3.init()
            engine.say('마스크 플리즈')
            engine.runAndWait()
        queue_warn.queue.clear()


def main():
    start = 0
    while True:

        _, img = cap.read()
        img = cv2.flip(img, 1)
        print(img.shape)

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
                    cv2.putText(img, label + " " + confidence + '%', (x, y - 8), font, 2, (0, 0, 255), 2)
                    maskCheck = True
                if label == 'mask':
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
                    cv2.putText(img, label + " " + confidence + '%', (x, y - 18), font, 2, (255, 255, 255), 2)

        if maskCheck:
            if queue_warn.full():
                queue_warn.queue.clear()
            queue_warn.put(maskCheck)

        end = (time.time() - start)
        start = time.time()
        frame_rate = 1/end
        print(frame_rate)
        cv2.putText(img, "yolov3-tiny, FPS : %0.1f" % frame_rate, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        maskCheck = False

        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    t_1 = threading.Thread(target=maskThread)
    t_1.start()
    main()
