import time

from flask import Flask, request, redirect, render_template, send_from_directory
from PIL import Image
import os
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']





@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            with Image.open(file_path) as image:
                yolo = YOLO('runs\\detect\\train4\\weights\\best.pt')

                result = yolo(image, save=True, classes=[0,1,2])

                flipped_filename = 'predict_'+filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], flipped_filename)

                result_path = os.path.join(result[0].save_dir,filename)
                with Image.open(result_path) as image:
                    image.save(file_path)
            return render_template('web.html', uploaded_image=filename, flipped_image=flipped_filename)

    return render_template('web.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run()
