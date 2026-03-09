import warnings
import cv2
import numpy as np
import torch
from counter import Counter

from nets import nn
from utils import util

warnings.filterwarnings("ignore")


def draw_line(image, x1, y1, x2, y2, index):
    w = 10
    h = 10
    color = (200, 0, 0)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Top left
    cv2.line(image, (x1, y1), (x1 + w, y1), color, 3)
    cv2.line(image, (x1, y1), (x1, y1 + h), color, 3)

    # Top right
    cv2.line(image, (x2, y1), (x2 - w, y1), color, 3)
    cv2.line(image, (x2, y1), (x2, y1 + h), color, 3)

    # Bottom right
    cv2.line(image, (x2, y2), (x2 - w, y2), color, 3)
    cv2.line(image, (x2, y2), (x2, y2 - h), color, 3)

    # Bottom left
    cv2.line(image, (x1, y2), (x1 + w, y2), color, 3)
    cv2.line(image, (x1, y2), (x1, y2 - h), color, 3)

    text = f'ID:{str(index)}'
    cv2.putText(image, text,
                (x1, y1 - 2),
                0, 0.5, (0, 255, 0),
                thickness=1, lineType=cv2.FILLED)


def main():
    size = 640

    #  Load model on CPU
    checkpoint = torch.load('./weights/v8_n.pt', map_location='cpu', weights_only=False)
    model = checkpoint['model'].float()
    model.eval()

    # Take an input from a camera
    # Change camera index to use a different camera
    reader = cv2.VideoCapture(0)

    if not reader.isOpened():
        print("Error opening video stream or file")

    fps = int(reader.get(cv2.CAP_PROP_FPS))
    bytetrack = nn.BYTETracker(fps)

    # read initial frame to get dimensions
    success, frame = reader.read()
    if not success:
        print("Could not read initial frame from camera")
        return

    height, width = frame.shape[:2]

    A = (width//2, 0)
    B = (width//2, height - 1)

    counter = Counter(A, B)
    
    while reader.isOpened():
        success, frame = reader.read()
        if not success:
            break

        boxes = []
        confidences = []
        object_classes = []

        image = frame.copy()
        shape = image.shape[:2]

        r = size / max(shape[0], shape[1])
        if r != 1:
            h, w = shape
            image = cv2.resize(image,
                               dsize=(int(w * r), int(h * r)),
                               interpolation=cv2.INTER_LINEAR)

        h, w = image.shape[:2]
        image, ratio, pad = util.resize(image, size)
        shapes = shape, ((h / shape[0], w / shape[1]), pad)

        # HWC → CHW, BGR → RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = np.ascontiguousarray(sample)

        sample = torch.from_numpy(sample).unsqueeze(0)
        sample = sample.float()      #  float32
        sample = sample / 255.0

        # Inference on CPU
        with torch.no_grad():
            outputs = model(sample)

        # NMS
        outputs = util.non_max_suppression(outputs, 0.001, 0.7)

        for i, output in enumerate(outputs):
            detections = output.clone()
            util.scale(detections[:, :4],
                       sample[i].shape[1:],
                       shapes[0],
                       shapes[1])

            detections = detections.cpu().numpy()

            for detection in detections:
                x1, y1, x2, y2 = list(map(int, detection[:4]))
                boxes.append([x1, y1, x2, y2])
                confidences.append(detection[4])
                object_classes.append(detection[5])

        outputs = bytetrack.update(np.array(boxes),
                                   np.array(confidences),
                                   np.array(object_classes))
        
        counter.update(outputs)

        if len(outputs) > 0:
            boxes = outputs[:, :4]
            identities = outputs[:, 4]
            object_classes = outputs[:, 6]

            for i, box in enumerate(boxes):
                if object_classes[i] != 0:
                    continue

                x1, y1, x2, y2 = list(map(int, box))
                index = int(identities[i]) if identities is not None else 0
                draw_line(frame, x1, y1, x2, y2, index)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        # ADDED

        x_left = 10
        x_right = width - 10
        y0 = 30
        dy = 35
        
        cv2.putText(frame,
            f"Ingress: {counter.in_count}",
            (x_left, y0 + 0*dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2)

        cv2.putText(frame,
            f"---------->",
            (x_left, y0 + 1*dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2)        
        
        (text_width, text_height), _ = cv2.getTextSize(
            f"Egress: {counter.out_count}",
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            2
        )

        x_right = width - text_width - 10

        cv2.putText(frame, f"Egress: {counter.out_count}", (x_right, y0),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        (text_width_arrow, _), _ = cv2.getTextSize(
            "<----------",
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            2
        )

        x_right_arrow = width - text_width_arrow - 10

        cv2.putText(frame, "<----------", (x_right_arrow, y0 + dy),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.line(frame, A, B, (0, 0, 255), 2)

        cv2.imshow('Frame', frame.astype('uint8'))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
