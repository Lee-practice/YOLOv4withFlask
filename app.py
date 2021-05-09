#yolov4를 tf로 변환한 것을 사용하기 위한 것
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np

#flask를 사용하기 위한 것
from flask import Flask, render_template, Response


framework = 'tf' #'tflite' # tf, tflite, trt
weights = './checkpoints/yolov4-416' #변환한 모델이 저장된 경로 적기
size = 416 # resize images to
tiny = False  # yolo-tiny인 경우 True 아니라면 False
model = 'yolov4' # yolov3 or yolov4
iou = 0.45 # iou threshold
score = 0.25 # score threshold

input_size = 416
webcam = cv2.VideoCapture(0) #webcam 사용
app = Flask(__name__)

#tf model load
saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

def gen_frames():
    frame_id = 0
    while True:
        return_value, frame = webcam.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            if frame_id == webcam.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")
        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)


        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        result = np.asarray(image)

        #각 물체가 몇%의 확률로 해당 물체라고 판별했는지 해당 물체를 판별한 시각을 출력
        object_num = -1
        flag = 0
        for i in scores.numpy()[0]:
            object_num += 1
            now_time = time.strftime('%Y'+'-'+'%m'+'-'+'%d'+'T'+'%H' + '-' + '%M' + '-' + '%S')
            if (i != 0):
                print(object_num, '번째 물체의 확률:', scores.numpy()[0][object_num], '시각:', now_time)
            else:
                if (object_num == 0):
                    flag = 1
                break

        
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #이미지 저장
        if (flag == 0):
            cv2.imwrite("C:/Users/USER/Desktop/capture/" + now_time + ".png", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        frame_id += 1

        #webcam에서 찍고 있는 화면을 web상에서 보여줌.
        ret, buffer = cv2.imencode('.jpg', result)
        frame1 = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
