from flask import Flask, render_template, request, redirect, url_for
import sys, subprocess, os
import time
app = Flask(__name__)

# 초기 화면
@app.route("/")
def first():
    return render_template("index.html")

# login
@app.route("/login", methods=('GET', 'POST'))
def login():
    return render_template("login.html")

# login
@app.route("/modal_login", methods=('GET', 'POST'))
def modal_login():
    return render_template("modal_login.html")

# register- 회원가입
@app.route("/register")
def register():
    return render_template('register.html')

# 파일 업로드하기 위한 메인 화면
@app.route("/main")
def main():
    return render_template("index2.html")

@app.route("/upload_done", methods=["POST"])
def upload_done():
    input_file = 'static/img/1.jpeg'
    output_file = 'static/photo/result_cartoon.png'

    if os.path.isfile(input_file):
        os.remove(input_file)
    if os.path.isfile(output_file):
        os.remove(output_file)

    uploaded_files = request.files["file"]
    uploaded_files.save("static/img/{}.jpeg".format(1))
    value = None
    value = request.form['character']
    print(request.form['character'])
    print("value:" + value)
    if value=="0":
        subprocess.call(["D:/project/web/static/disney/disneyenv/Scripts/python.exe", "D:/project/web/static/disney/full.py"])
        time.sleep(5)
        return redirect(url_for("print_photo_py"))
    elif value=="1":
        subprocess.call(["D:/project/web/static/photo2cartoon/photo2cartoonenv/Scripts/python.exe", "D:/project/web/static/photo2cartoon/test.py"])
        time.sleep(2)
        return redirect(url_for("ani_print_photo_py"))

@app.route("/print_photo_py")
def print_photo_py():
    
    photo = f"photo/result_disney.png"
    return render_template("print_photo.html", photo=photo)

@app.route("/ani_print_photo_py")
def ani_print_photo_py():
    photo = f"photo/result_cartoon.png"
    return render_template("ani_print_photo.html", photo=photo)
