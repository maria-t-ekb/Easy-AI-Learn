from flask import Flask, request, redirect, url_for, send_from_directory, render_template, send_file
# import easyailearn
import yolo_train
import sqlite3
import os

app = Flask(__name__)

# @app.route('/')
# def index():
#     return send_from_directory('.', 'index_image_all.html')

# @app.before_request
# def before_first_request():
#     init_db()

# @app.get('/favicon.ico')
# def get_favicon():
#     return Flask.send_static_file('static/favicon.ico')

@app.route('/')
def index():
    # conn = get_db_connection()
    # posts = conn.execute('SELECT * FROM posts').fetchall()
    # conn.close()
    return render_template('index.html')   # , posts=posts

@app.route('/image_all_dataset')
def image_all_dataset():
    return render_template('index_image_all.html')

@app.route('/image_part_dataset')
def image_part_dataset():
    return render_template('index_image_part.html')

@app.route('/audio_all_dataset')
def audio_all_dataset():
    return render_template('index_audio_all.html')

@app.route('/audio_part_dataset')
def audio_part_dataset():
    return render_template('index_audio_part.html')

@app.route('/pose_all_dataset')
def pose_all_dataset():
    return render_template('index_pose_all.html')

@app.route('/pose_part_dataset')
def pose_part_datasetv():
    return render_template('index_pose_part.html')

@app.route('/text_all_dataset')
def text_all_dataset():
    return render_template('index_text_all.html')

@app.route('/text_part_dataset')
def text_part_dataset():
    return render_template('index_text_part.html')

@app.route('/go-about')
def go_about():
    return redirect(url_for('index1'))

def upload_all_dir(files):
    for file in files:
        relative_path = file.filename
        save_path = os.path.join('uploads', relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
    dir_name = [x for x in relative_path.split('/')][0]
    return dir_name

@app.route('/image_all_dataset_upload', methods=['POST'])
def image_all_dataset_upload():
    if 'files' not in request.files:
        return 'Файл не найден в запросе', 400

    files = request.files.getlist('files')
    dir_name = upload_all_dir(files)

    y = int(request.form['epochs'])
    d = 1

    return yolo_train.y_train(f'uploads/{dir_name}', y, d)
    # return f'{dir_name}, {y}, {d}'


@app.route('/audio_all_dataset_upload', methods=['POST'])
def audio_all_dataset_upload():
    if 'files' not in request.files:
        return 'Файл не найден в запросе', 400

    files = request.files.getlist('files')
    dir_name = upload_all_dir(files)

    y = int(request.form['epochs'])
    d = 1

    return yolo_train.train_and_test('audio', 1, f'uploads/{dir_name}', epoch=y)
    # return f'{dir_name}, {y}, {d}'


@app.route('/pose_all_dataset_upload', methods=['POST'])
def pose_all_dataset_upload():
    if 'files' not in request.files:
        return 'Файл не найден в запросе', 400

    files = request.files.getlist('files')
    dir_name = upload_all_dir(files)

    y = int(request.form['epochs'])
    d = 1

    return yolo_train.train_and_test('pose', 1, f'uploads/{dir_name}', epoch=y)
    # return f'{dir_name}, {y}, {d}'


@app.route('/text_all_dataset_upload', methods=['POST'])
def text_all_dataset_upload():
    if 'files' not in request.files:
        return 'Файл не найден в запросе', 400

    files = request.files.getlist('files')
    dir_name = upload_all_dir(files)

    y = int(request.form['epochs'])
    d = 1

    return yolo_train.train_and_test('text', 1, f'uploads/{dir_name}', epoch=y)
    # return f'{dir_name}, {y}, {d}'


@app.route('/part_dataset_upload', methods=['POST'])
def part_dataset_upload():
    project_name = request.form.get('ProjectName')
    dir_name = request.form.get('DirName')
    if not (dir_name or project_name):
        return "Ошибка: не передано имя директории", 400

    files = request.files.getlist('files[]')
    saved_files = []

    for file in files:
        if file:
            filename = file.filename
            filepath = os.path.join(f'uploads/{project_name}/{dir_name}/{filename}')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            saved_files.append(filename)

    return 'Файлы успешно загружены'


@app.route('/image_part_dataset_train', methods=['POST'])
def image_part_dataset_train():
    project_name = request.form.get('ProjectName')
    y = int(request.form['epochs'])
    d = 1
    # return f'{project_name}, {y}, {d}'
    return yolo_train.y_train(f'uploads/{project_name}', y, d)

@app.route('/audio_part_dataset_train', methods=['POST'])
def audio_part_dataset_train():
    project_name = request.form.get('ProjectName')
    y = int(request.form['epochs'])
    d = 1
    # return f'{project_name}, {y}, {d}'
    return yolo_train.train_and_test('audio', 1, f'uploads/{project_name}', y, d)

@app.route('/pose_part_dataset_train', methods=['POST'])
def pose_part_dataset_train():
    project_name = request.form.get('ProjectName')
    y = int(request.form['epochs'])
    d = 1
    # return f'{project_name}, {y}, {d}'
    return yolo_train.train_and_test('pose', 1, f'uploads/{project_name}', epoch=y)

@app.route('/text_part_dataset_train', methods=['POST'])
def text_part_dataset_train():
    project_name = request.form.get('ProjectName')
    y = int(request.form['epochs'])
    d = 1
    # return f'{project_name}, {y}, {d}'
    return yolo_train.train_and_test('text', 1, f'uploads/{project_name}', epoch=y)


@app.route('/image_test', methods=['POST'])
def image_test():
    try:
        file = request.files['file']
        filename = file.filename
        file.save(f'uploads/{filename}')
        return yolo_train.y_test(f'uploads/{filename}')
        # return 'file'
    except:
        return ' '

@app.route('/audio_test', methods=['POST'])
def audio_test():
    try:
        file = request.files['file']
        filename = file.filename
        file.save(f'uploads/{filename}')
        return yolo_train.train_and_test('audio', 2, f'uploads/{filename}')
        # return 'file'
    except:
        return ' '

@app.route('/pose_test', methods=['POST'])
def pose_test():
    try:
        file = request.files['file']
        filename = file.filename
        file.save(f'uploads/{filename}')
        return yolo_train.train_and_test('pose', 2, f'uploads/{filename}')
        # return 'file'
    except:
        return ' '

@app.route('/text_test', methods=['POST'])
def text_test():
    try:
        text = request.form['text']
        # filename = file.filename
        # file.save(f'uploads/{filename}')
        return yolo_train.train_and_test('text', 2, text)
        # return 'file'
        # return text
    except:
        return ' '


@app.route('/upload_neuro')
def upload_neuro():
    try:
        path = "uploads/Без названия.jpg"
        return send_file(path, as_attachment=True, download_name='neuro.pdf')
    except:
        return ' '

# def get_db_connection():
#     conn = sqlite3.connect('database.db')
#     conn.row_factory = sqlite3.Row
#     return conn

# def close_db_connection(conn):
#     conn.close()
#
# def init_db():
#     conn = get_db_connection()
#     conn.execute('CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, content TEXT NOT NULL)')
#     conn.close()

if __name__ == '__main__':
    # app.run()
    app.run(host='0.0.0.0')
