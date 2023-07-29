from ultralytics import YOLO
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import cv2
from moviepy.editor import VideoFileClip
import moviepy.editor as mp
import shutil

def convert_avi_to_mp4(input_file, output_file):
    video = VideoFileClip(input_file)
    video.fx(mp.vfx.blackwhite)
    video.write_videofile(output_file, codec='libx264')


def save_grayscale(directory, new_filename):
    # Load image
    img = cv2.imread(directory+new_filename)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Save grayscale image
    cv2.imwrite(directory + new_filename, gray)

def save_video(directory, new_filename):
    # Membaca video menggunakan moviepy
    clip = VideoFileClip(directory+new_filename)

    duration = clip.duration

    # Mengonversi setiap frame ke grayscale
    gray_clip = clip.fx(mp.vfx.blackwhite)

    gray_clip = gray_clip.subclip(0, duration)

    gray_clip.write_videofile(directory+new_filename, codec="libx264", fps=clip.fps)

def gray_scale_video(directory, new_filename):
    source = cv2.VideoCapture(directory + new_filename)

    frame_width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))

    size = (frame_width, frame_height)

    result = cv2.VideoWriter(directory + new_filename,  
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, size, 0)

    while True:
        ret, img = source.read()
        
        if not ret:  # Break the loop if the frame is empty
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result.write(gray)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    source.release()



def sign_detection(input,username,extension):
    # Inisiasi variabel
    torch_class = []
    class_id = []
    detect_names = []
    names = ['A', 'Apa', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Halo', 'I', 'AkuCintaKamu', 'J', 'K', 'Kamu', 'L', 'M', 'Malam', 'N', 'Nama', 'O', 'P', 'Pagi', 'Q', 'R', 'S', 'Aku', 'Selamat', 'Siapa', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    daftar_kata_tidak_boleh_berulang = ['Apa', 'Aku' , 'Kamu', 'Kalian', 'Kita', 'Mereka', 'Siapa', 'Kenapa']

    # Load model
    model = YOLO('model/yolov8n.pt') 
    model = YOLO('model/best.pt')

    # Mengambil data dari form input
    ekstensi_gambar = ['.jpg', '.jpeg', '.png']
    ekstensi_video = ['.mp4', '.avi', '.mkv']

    # Membedakan proses prediksi berdasarkan ekstensi file
    if extension in ekstensi_gambar:
        # Inference
        results = model(input)

    elif extension in ekstensi_video:
        # menghapus folder sebelumnya
        if os.path.exists('asset/signgesture/output/'+username):
            shutil.rmtree('asset/signgesture/output/'+username)

        # Inference dan simpan sebagi mp4
        results = model.predict(input, save=True, project='asset/signgesture/output/'+username, conf=0.7)
        convert_avi_to_mp4('asset/signgesture/output/'+username+'/predict/'+username+'.avi', 'asset/signgesture/output/'+username+'/predict/'+username+'.mp4')

    # Looping untuk mengambil class torch hasil object detection
    for result in results:
        boxes = result.boxes
        for box in boxes:
            box = box.cpu()
            box = box.numpy()
            torch_class.append(box.cls)

    # Looping untuk mengambil class id dari class torch
    for i in range(len(torch_class)):
        class_id.append(torch_class[i][0])

    # Looping untuk mengambil nama dari class id
    for i in range(len(class_id)):
        for j in range(len(names)):
            if class_id[i] == j:
                # Kondisi untuk menghindari duplikat nama yang sama berurutan
                if len(detect_names) > 0:
                    if names[j] != detect_names[-1]:
                        detect_names.append(names[j])
                else:
                    detect_names.append(names[j])
    
    # Menggabungkan hasil looping menjadi satu kalimat
    result_sentence = ''.join(detect_names)

    # Return the result
    return result_sentence

def normalization(input_text):
    # Mengambil dataset
    df = pd.read_csv('asset/data/lexnorm.csv', sep=';')

    # Meload Model
    model = load_model('model/lexnorm.h5')

    # Menghapus tanda baca
    def punctual_removal(text):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for x in text.lower(): 
            if x in punctuations: 
                text = text.replace(x, " ") 
        return text

    # Menghapus tanda baca
    def labels_punctual_removal(text):
        punctuations = '''!()[]{};:'"\,<>./?@#$%^&*_~'''
        for x in text.lower(): 
            if x in punctuations: 
                text = text.replace(x, "") 
        return text
    
    df['text'] = df['text'].apply(punctual_removal)
    df['labels'] = df['labels'].apply(labels_punctual_removal)

    # Membuat kamus karakter unik dari data
    characters = sorted(list(set("".join(df['text'] + df['labels']))))
    char_to_index = {char: idx for idx, char in enumerate(characters)}
    index_to_char = {idx: char for idx, char in enumerate(characters)}

    # Mengubah data menjadi representasi numerik
    X = [[char_to_index[char] for char in data] for data in df['text']]
    y = [[char_to_index[char] for char in data] for data in df['labels']]

    # Padding sequence menjadi panjang maksimum
    max_length = max(max(len(x) for x in X), max(len(y) for y in y))

    def convert_text_to_input(text):
        input_sequence = [char_to_index[char] for char in text]
        input_sequence = pad_sequences([input_sequence], maxlen=max_length, padding='post')
        return input_sequence

    # Mengonversi representasi numerik menjadi kalimat output
    def convert_output_to_text(output_sequence):
        output_text = ''.join([characters[int(index)] for index in output_sequence])
        return output_text

    # Mengonversi input menjadi representasi numerik
    input_sequence = convert_text_to_input(input_text.lower())

    # Melakukan prediksi
    prediction = model.predict(input_sequence)
    
    # Mengonversi output menjadi kalimat
    output_text = convert_output_to_text(np.argmax(prediction, axis=2)[0])

    return output_text

def input_repaired(input_text):
    df = pd.read_csv('asset/data/dataset.csv',sep=";")

    # Menghapus tanda baca
    def punctual_removal(text):
        punctual = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for i in text.lower(): 
            if i in punctual: 
                text = text.replace(i, "") 
        return text

    # Mengubah kalimat menjadi huruf kecil
    def lower_case(text):
        return text.lower()

    # Apply function
    def preprocess(text):
        text = punctual_removal(text)
        text = lower_case(text)
        return text
    
    df['text'] = df['text'].apply(preprocess)
    label_to_dict = df['normalization'].apply(preprocess)

    words = sorted(list(set(" ".join(label_to_dict+df['text']).split())))

    chars = sorted(list(set(" ".join(df['text']+df['normalization']))))

    words = words + chars

    # Menghilangkan kata yang kurang dari 3 karakter
    words = [word for word in words if len(word) > 2]

    # mengubah urutan words dari kata terpanjang ke terpendek
    words = sorted(words, key=len, reverse=True)

    def pecah_kata_kalimat(kalimat):
        tokens = list(kalimat)
        return tokens

    input_text = preprocess(input_text)
    input_text = pecah_kata_kalimat(str(input_text))
    print(input_text)

    def is_valid_word(word):
        return word in words

    def menyaring_kata_kalimat(pecahan_huruf):
        n = len(pecahan_huruf)
        kalimat = []

        def backtrack(start, current_sentence):
            if start == n:
                kalimat.append(" ".join(current_sentence))
                return

            for end in range(start + 1, n + 1):
                word = ''.join(pecahan_huruf[start:end])
                if is_valid_word(word):
                    current_sentence.append(word)
                    backtrack(end, current_sentence)
                    current_sentence.pop()
            
        backtrack(0, [])
        
        if len(kalimat) == 0:
            kalimat.append("")

        return kalimat[0]
    
    hasil = menyaring_kata_kalimat(input_text)

    return hasil

def lexnorm(input_text):
    df = pd.read_csv('asset/data/dataset.csv',sep=";")

    # Menghapus tanda baca
    def punctual_removal(text):
        punctual = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for i in text.lower(): 
            if i in punctual: 
                text = text.replace(i, "") 
        return text

    # Mengubah kalimat menjadi huruf kecil
    def lower_case(text):
        return text.lower()

    # Apply function
    def preprocess(text):
        text = punctual_removal(text)
        text = lower_case(text)
        return text

    df['text'] = df['text'].apply(preprocess)
    normalization_kamus = df['normalization'].apply(preprocess)

    # Membuat kamus kata-kata unik
    words = sorted(list(set(" ".join(df['text']+normalization_kamus).split(" "))))

    # Membuat kamus huruf unik
    chars = sorted(list(set(" ".join(df['text']+df['normalization']))))

    # Menggabungkan kamus kata dan huruf
    vocab = words + chars

    word_to_index = {word:idx for idx, word in enumerate(vocab)}

    index_to_word = {idx:word for idx, word in enumerate(vocab)}

    # Mengubah data menjadi numerik
    X = [[word_to_index[char] for char in text] for text in df['text']]
    y = [[word_to_index[char] for char in text] for text in df['normalization']]

    # menghitung kalimat terpanjang dari text dan normalization jika digabungkan
    maxlen = max(max(len(x) for x in X), max(len(y) for y in y))

    model = load_model('model/lexnormv2.h5')

    # Mengonversi kalimat input menjadi representasi numerik
    def convert_text_to_input(text):
        input_sequence = [word_to_index[char] for char in text]
        input_sequence = pad_sequences([input_sequence], maxlen=maxlen, padding='post')
        return input_sequence

    # Mengonversi representasi numerik menjadi kalimat output
    def convert_output_to_text(output_sequence):
        output_text = ''.join([vocab[int(index)] for index in output_sequence])
        return output_text

    # Mengonversi input menjadi representasi numerik
    input_sequence = convert_text_to_input(input_text)

    # Melakukan prediksi menggunakan model
    output_sequence = model.predict(input_sequence)

    # Mengonversi output menjadi kalimat
    output_text = convert_output_to_text(np.argmax(output_sequence, axis=2)[0])

    return output_text






