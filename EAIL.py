from flask import Flask, request, redirect, url_for, send_from_directory, render_template
# import easyailearn
import yolo_train
import sqlite3
import os

app = Flask(__name__)

# @app.route('/')
# def index():
#     return send_from_directory('.', 'index1.html')

@app.before_request
def before_first_request():
    init_db()

@app.route('/')
def index():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM posts').fetchall()
    conn.close()
    return render_template('index.html', posts=posts)

@app.route('/page1')
def about():
    return render_template('index1.html')

@app.route('/page2')
def about1():
    return render_template('12.html')

@app.route('/go-about')
def go_about():
    return redirect(url_for('index1'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return 'Файл не найден в запросе', 400

    # file = request.files['file']

    files = request.files.getlist('files')
    for file in files:
        relative_path = file.filename
        save_path = os.path.join('uploads', relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)

    dir_name = [x for x in relative_path.split('/')][0]

    y = request.form['epochs']
    d = request.form['device']

    # Можно сохранить файл:
    # filename = file.filename
    # os.mkdir()
    # filename = 'dataset'
    # file.save(f'uploads/{filename}')

    # file.save(f'uploads/dataset')  # Убедись, что папка uploads существует
    return yolo_train.y_train(f'uploads/{dir_name}', y, d)
    # return f'{filename}, {y}, {d}'
    # return f'Файл успешно загружен'


@app.route('/upload1', methods=['POST'])
def ai_test():
    file = request.files['file']
    filename = file.filename
    file.save(f'uploads/{filename}')
    return yolo_train.y_test(f'uploads/{filename}')
    # return 'file'


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def close_db_connection(conn):
    conn.close()

def init_db():
    conn = get_db_connection()
    conn.execute('CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, content TEXT NOT NULL)')
    conn.close()

if __name__ == '__main__':
    app.run()
