import os.path

import cv2
from flask import Flask, render_template, request
from torch.distributed.elastic.multiprocessing.redirects import redirect
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("best.pt")

def process_image(image_path):
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Ошибка: Не удалось загрузить изображение по пути {image_path}")
        exit()

    results = model.predict(source=frame, conf=0.6)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Вместо возврата полного пути, возвращаем только имя файла
    output_filename = 'output_image.jpg'
    output_path = os.path.join('static', output_filename)
    cv2.imwrite(output_path, frame)
    print(f"Результат сохранён в {output_path}")
    return output_filename


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        upload_path = os.path.join('static', file.filename)
        file.save(upload_path)

        result_image_path = process_image(upload_path)

        return render_template('result.html', result_image=result_image_path)

if __name__ == '__main__':
    app.run(debug=True)
