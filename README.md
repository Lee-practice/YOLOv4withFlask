# YOLOv4_flask

## 프로젝트 설명
> YOLOv4로 custom data를 학습하여 생성한 weights를 tf 모델로 변환하고, 이를 활용하여 web상에서 webcam을 통해 object detection 결과를 보여주도록 하였다. <br>
> 학습을 여러 버전을 두고 진행했는데, 이 프로젝트에서는 3번째 버전을 사용하여 진행하였다.

## 진행 과정
1. YOLOv4를 통해 학습 진행 (custom data 사용: take-out cup) - 구글 colab 사용
2. 학습 결과로 얻은 darknet 가중치 파일을 tensorflow를 위한 것으로 변환 - 구글 colab 사용
3. 변환된 model을 flask를 통해 web 상에서 webcam을 활용하여 사용할 수 있도록 함


### 참고 사이트
> https://hanryang1125.tistory.com/16 <br>
> https://webnautes.tistory.com/1417 <br>
> https://blog.daum.net/ejleep1/1005 <br>
 
### 실행에 활용한 프로젝트
> https://github.com/hunglc007/tensorflow-yolov4-tflite
