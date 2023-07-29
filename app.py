from flask import Flask, request, jsonify, send_file, url_for, send_from_directory
import models as ml
from werkzeug.exceptions import BadRequestKeyError
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'asset/signgesture/'

@app.route("/files/<path:path>")
def get_file(path):
    
    file_path = 'asset/signgesture/output/'+path+'/predict/'+path+'.mp4'
    
    return send_file(file_path, mimetype='video/mp4')
    

@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == "POST":

        try:
            # Mengambil data dari form input
            ekstensi_gambar = ['.jpg', '.jpeg', '.png','.bmp','.dng','.mpo','.tif','.tiff','.webp','.pfm']
            ekstensi_video = ['.mp4', '.avi', '.mkv','.asf','.gif']

            username = request.form['username']
            sign_data = request.files['sign']
            extension = os.path.splitext(sign_data.filename)[1]

            # Menyimpan data ke dalam folder
            directory = "asset/signgesture/"
            new_filename = username+extension

            sign_data.save(directory + new_filename)
            
            # menyimpan gambar grayscale
            if extension in ekstensi_gambar:
                ml.save_grayscale(directory, new_filename)
            
            # menyimpan video grayscale
            elif extension in ekstensi_video:
                # ml.gray_scale_video(directory, new_filename)
                # pass
                print("video")
            
            # Mengembalikan pesan jika ekstensi file tidak didukung
            else:
                output = {
                    "status": "error",
                    "message": "Ekstensi file tidak didukung",
                    "status_code": 400
                }
                return jsonify(output)

            # Exception handling
            try:
                # Melakukan deteksi menggunakan model yang telah dibuat
                hasil_deteksi = ml.sign_detection(directory + new_filename, username, extension)

                # Memperbaiki format hasil deteksi
                cari_kata = ml.input_repaired(hasil_deteksi)

                # Melakukan normalisasi menggunakan model yang telah dibuat
                # hasil_normalisasi = ml.normalization(hasil_deteksi)
                hasil_normalisasi = ml.lexnorm(cari_kata)

                # Mengembalikan pesan jika hasil deteksi kosong
                if hasil_deteksi == "":
                    hasil_deteksi = "Gerakan Bahasa Isyarat Tidak Terdeteksi"
                    hasil_normalisasi = "Tidak Ternormalisasi"
                
                # Mengembalikan hasil deteksi
                if extension in ekstensi_gambar:
                    output = {
                        "status": "success",
                        "message": "Berhasil melakukan deteksi",
                        "hasil": {
                                "sebelum_normalisasi": hasil_deteksi,
                                "sesudah_normalisasi": hasil_normalisasi
                        },
                        "status_code": 200
                    }
                elif extension in ekstensi_video:
                    lokasi_video = "http://192.168.100.46:8000/files/"+username
                    output = {
                        "status": "success",
                        "message": "Berhasil melakukan deteksi",
                        "hasil": {
                                "video_deteksi": lokasi_video,
                                "sebelum_normalisasi": hasil_deteksi,
                                "sesudah_normalisasi": hasil_normalisasi
                        },
                        "status_code": 200
                    }

            # Mengembalikan pesan error jika terjadi error
            except Exception as e:
                output = {
                        "status": "error",
                        "message": str(e),
                        "status_code": 500
                }

        # Mengembalikan pesan error jika data tidak lengkap
        except BadRequestKeyError:
            output = {
                "status": "error",
                "message": "Data tidak lengkap",
                "status_code": 400
            }

    # Mengembalikan pesan error jika method yang digunakan bukan POST
    else:
        output = {
            "status": "error",
            "message": "Method yang digunakan bukan POST",
            "status_code": 405
        }

    return jsonify(output)

if __name__ == '__main__':
    app.run(host='192.168.100.46',port=8000,debug=True)