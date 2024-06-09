import tkinter as tk #thư viện giao diện 
from tkinter import Message, Text, PhotoImage, Label, Button 
import cv2 
import os #thư viện cung cấp hàm
import csv #Thư viện này cho phép bạn thao tác với các tệp CSV (Comma Separated Values)
import numpy as np #Thư viện này cung cấp hỗ trợ cho các mảng và ma trận nhiều chiều
from PIL import Image, ImageTk #thư viện xử lí ảnh
import pandas as pd # thư viện cung cấp cấu trúc dữ liệu
import datetime #Thư viện này cung cấp các lớp và hàm để làm việc với thời gian và ngày tháng trong Python.
import time #Thư viện này cung cấp các hàm để làm việc với thời gian
import tkinter.ttk as ttk
import tkinter.font as font
from tensorflow.keras.preprocessing.image import ImageDataGenerator #công cụ để tạo ra dữ liệu để huấn luyện mô hình từ dữ liệu hình ảnh
#Thư viện này cung cấp các công cụ để xây dựng và huấn luyện mạng nơ-ron sâu. 
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

#Chức năng tiền xử lý và tăng cường hình ảnh
def augment_image(image):
    augmented_images = []
    rows, cols = image.shape

    # Xoay hình ảnh
    for angle in range(-15, 20, 5):
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(dst)

    # Dịch chuyển hình ảnh
    for shift in range(-5, 6, 5):
        M = np.float32([[1, 0, shift], [0, 1, shift]])
        dst = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(dst)

    return augmented_images


def preprocess_image(image): # định nghĩa hàm
    # Chuyển đổi ảnh sang xám
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #thư viện chuyển màu convertcolor
    image = cv2.equalizeHist(image) #độ tương phản
    return image


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s) #chuyển đổi một ký tự Unicode thành giá trị số tương ứng nếu có
        return True
    except (TypeError, ValueError):
        pass

    return False

def clear(): #tạo hàm xóa dữ liệu id và tên
    txt.delete(0, 'end')
    res = ""
    message.configure(text=res)

def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)

def TakeImages(): #lấy hình ảnh để train  
    Id = txt.get()
    name = txt2.get()
    if is_number(Id) and name.isalpha():
        # Tạo thư mục TrainingImage nếu chưa tồn tại
        if not os.path.exists("TrainingImage"):
            os.makedirs("TrainingImage")

        # Tạo thư mục con cho từng lớp (dựa trên ID và Tên)
        class_dir = os.path.join("TrainingImage", name)
        if not os.path.exists(class_dir): #kiểm tra thư mục con có tồn tại không
            os.makedirs(class_dir)

        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml" #Dòng này đặt đường dẫn đến tập tin XML chứa thông tin về bộ lọc Haarcascade được sử dụng để phát hiện khuôn mặt trong ảnh
        detector = cv2.CascadeClassifier(harcascadePath) #Dòng này tạo một đối tượng 
        sampleNum = 0 #đưa về 0 để đếm số lượng ảnh từ webcam

        while True:
            ret, img = cam.read() #đọc hình tù webcam
            if not ret:
                break

            img2 = preprocess_image(img) # tiền xử lý khung hình
            faces = detector.detectMultiScale(img2, scaleFactor=1.25, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            # phát hiện khuôn mặt xác định vị trí lưu ảnh.
            for (x, y, w, h) in faces:  
                cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 2) #2 là độ dày hình chữ nhật hình chữ nhật màu xanh dương
                sampleNum += 1 #đếm số lượng trong file
                face = img2[y:y + h, x:x + w] #cắt
                face = cv2.resize(face, (400, 400)) #thay đổi khuôn mặt 400x400 pixel
                augmented_faces = augment_image(face) #tăng cường dữ liệu hình ảnh khuôn mặt

                for i, aug_face in enumerate(augmented_faces): 
                    cv2.imwrite(f"{class_dir}/{name}.{Id}.{sampleNum}_{i}.jpg", aug_face)

                cv2.imshow('frame', img2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif sampleNum >= 500:
                break
        cam.release() #sử dụng để giải phóng tài nguyên 
        cv2.destroyAllWindows()

        res = "Ảnh đã được lưu với ID : " + Id + " - Tên : " + name #thông báo 
        row = [Id, name]
        with open('StudentDetails/StudentDetails.csv', 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=';') #chỉ định dấu phân cách được sử dụng giữa các trường dữ liệu.
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res) #tạo giao diện
    else:
        if is_number(Id):
            res = "Nhập tên là chữ cái"
            message.configure(text=res)
        if name.isalpha():
            res = "Nhập ID là số"
            message.configure(text=res)

def TrainImages():
    train_data_dir = 'TrainingImage'

    #Chuẩn bị tăng cường dữ liệu
    datagen = ImageDataGenerator(
        rescale=1. / 255, #Thực hiện một phép biến đổi tuyến tính lên tất cả các giá trị pixel trong hình ảnh để đảm bảo rằng chúng nằm trong khoảng [0,1]
        rotation_range=10, # xoay hình ảnh.
        width_shift_range=0.1,#ngẫu nhiên phép dịch hình ảnh theo chiều ngang
        height_shift_range=0.1,# ngẫu nhiên phép dịch hình ảnh theo chiều dọc
        shear_range=0.1,# phép cắt hình ảnh
        zoom_range=0.1,
        horizontal_flip=True, #lật
        fill_mode='nearest'#Chiến lược điền dữ liệu cho các vùng hình ảnh mới được tạo ra bằng cách sao chép các giá trị gần nhất từ các vùng lân cận.
    )

    train_generator = datagen.flow_from_directory(
        train_data_dir, #Đường dẫn tới thư mục chứa dữ liệu huấn luyện.
        target_size=(64, 64),#Kích thước mà các hình ảnh sẽ được thay đổi đến trước khi đưa vào mô hình. 
        color_mode='grayscale',# Chế độ màu của các hình ảnh
        batch_size=32,#Kích thước của các lô dữ liệu được tạo ra bởi ImageDataGenerator. số lượng hình ảnh cập nhật
        class_mode='categorical'
    )

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))#xác định kích thước của dữ liệu đầu vào. Trong trường hợp này, hình ảnh có kích thước 64x64 pixel và có một kênh màu (ảnh xám).
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))#sử dụng để tăng tính phi tuyến tính của mô hình.
    model.add(MaxPooling2D(pool_size=(2, 2))) #Lớp này thực hiện phép max pooling để giảm kích thước của đầu ra từ các lớp convolution trước đó.
    model.add(Flatten())#Lớp này chuyển đổi đầu ra từ các lớp convolution và pooling thành một vector 1D để truyền vào lớp kết nối đầy đủ 
    model.add(Dense(128, activation='relu'))#128 là số lượng neurons trong lớp này.,activation='relu' là hàm kích hoạt ReLU được sử dụng.
    model.add(Dropout(0.5)) #Lớp này thực hiện phép dropout để ngẫu nhiên "tắt" một số neurons trong quá trình huấn luyện, nhằm tránh hiện tượng overfitting, 50% số neurons sẽ được tắt ngẫu nhiên.
    model.add(Dense(train_generator.num_classes, activation='softmax')) #train_generator.num_classes là số lượng lớp (classes) trong dữ liệu huấn luyện, được lấy từ train_generator.
#activation='softmax' là hàm kích hoạt softmax được sử dụng để tính toán xác suất của các lớp đầu ra.
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#optimizer=Adam(learning_rate=0.001): Sử dụng thuật toán tối ưu hóa Adam với learning rate (tốc độ học) là 0.001. 
# loss='categorical_crossentropy': Sử dụng hàm mất mát categorical crossentropy, phù hợp với bài toán phân loại nhiều lớp. 
# metrics=['accuracy']: Đánh giá hiệu suất của mô hình bằng độ chính xác trong quá trình huấn luyện.
    model.fit(train_generator, epochs=10) # Số lượng epochs (vòng lặp huấn luyện) mà mô hình sẽ được huấn luyện.

    model.save('TrainingImageLabel/Trainner.h5') #LƯU DỮ liệu vào đường dẫn

    res = "Train thành công"
    message.configure(text=res)

def TrackImages():
    model = load_model('TrainingImageLabel/Trainner.h5')# sử dụng model ở train để check in khuôn mặt
    harcascadePath = "haarcascade_frontalface_default.xml" #đường dần lấy link xml
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    #tạo một đối tượng CascadeClassifier từ tệp XML chứa các thông số cần thiết để phát hiện khuôn mặt. 
    cam = cv2.VideoCapture(0) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    # Lấy tên các lớp từ thư mục TrainingImage
    class_labels = os.listdir('TrainingImage')
    class_dict = {class_labels[i]: i for i in range(len(class_labels))}

    while True:
        ret, im = cam.read() #đọc dữ liệu 
        if not ret:
            break

        im2 = preprocess_image(im) #chuyển đổi sang ảnh xám
        faces = faceCascade.detectMultiScale(im2, 1.25, 5) #im2 là hình ảnh đã xử lí, 1.2 là độ thu nhỏ, 5 minNeibor số lượng các điểm ảnh láng giềng

        for (x, y, w, h) in faces:
            cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2) #vẽ hình chữ nhật
            face = im2[y:y + h, x:x + w] #cắt
            face = cv2.resize(face, (64, 64)) #thay đổi kích thước
            face = face / 255.0 #chuấn gía trị
            face = np.expand_dims(face, axis=0) #mở rộng kích thước
            face = np.expand_dims(face, axis=-1)  # Thêm dòng này cho hình ảnh thang độ xám
            prediction = model.predict(face) #dự đoán lớp khuôn mặt
            class_index = np.argmax(prediction, axis=1)[0] # prediction của lớp dự đoán trả về giá trị lớn nhất

            if prediction[0][class_index] > 0.85: # xác suất lớp dự đoán lớn hơn 0.85 thì 
                name = class_labels[class_index]  #lấy tên lớp dưj đoán
                Id = class_dict[name] #lấy thông tin id
                ts = time.time() #lấy thời gian và ngày
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                tt = f"{Id}-{name}" #chuỗi kết quả hiển thị sau khi điểm danh 
            else:
                tt = "Unknown" 
                noOfFile = len(os.listdir("ImagesUnknown")) + 1 #tính toán số lượng tệp image và tangư lên 1
                if not os.path.exists("ImagesUnknown"):
                    os.makedirs("ImagesUnknown") #kiểm tra thư mục
                cv2.imwrite(f"ImagesUnknown/Image{noOfFile}.jpg", im[y:y + h, x:x + w])# lưu hình ảnh 

            cv2.putText(im, str(tt), (x, y + h + 30), font, 1, (255, 255, 255), 2)# hiển thị thông tin

        attendance = attendance.drop_duplicates(subset=['Id'], keep='first') #loại bỏ các bản ghi trùng lặp 
        cv2.imshow('im', im)
        if cv2.waitKey(1) == ord('q'):
            break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = f"Attendance/Attendance_{date}_{Hour}-{Minute}-{Second}.csv"
    attendance.to_csv(fileName, index=False)

    cam.release()
    cv2.destroyAllWindows()
    res = attendance
    message2.configure(text=res) #cập nhật nội dung


# GUI code
anhbia = tk.Tk()
anhbia.geometry('1280x600')
anhbia.title("Bao cao cuoi ki xu li anh")

picture = PhotoImage(file='baocaocuoiky.png')
mainpicture = Label(anhbia, image=picture)
mainpicture.place(x=0, y=0)

bt = Button(anhbia, command=lambda: anhbia.destroy(), text='START', font=("Arial", 18, 'bold'), fg='blue')
bt.place(x=640, y=300)

anhbia.mainloop()

window = tk.Tk()
window.title("Hệ thống nhận diện khuôn mặt")
window.geometry('1200x900')
window.configure(background='LightBlue4')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text="Hệ thống nhận diện khuôn mặt", bg="dark slate gray", fg="white", width=50, height=3, font=('times', 30, 'italic bold underline'))
message.place(x=200, y=20)

lbl = tk.Label(window, text="Nhập ID", width=20, height=2, fg="white", bg="gray25", font=('times', 15, ' bold '))
lbl.place(x=400, y=200)

txt = tk.Entry(window, width=20, bg="gray25", fg="white", font=('times', 15, ' bold '))
txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Nhập tên", width=20, fg="white", bg="gray25", height=2, font=('times', 15, ' bold '))
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window, width=20, bg="gray25", fg="white", font=('times', 15, ' bold '))
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Thông báo : ", width=20, fg="white", bg="gray25", height=2, font=('times', 15, ' bold underline '))
lbl3.place(x=400, y=400)

message = tk.Label(window, text="", bg="gray25", fg="white", width=30, height=2, activebackground="yellow", font=('times', 15, ' bold '))
message.place(x=700, y=400)

lbl3 = tk.Label(window, text="Thông tin điểm danh : ", width=20, fg="white", bg="gray25", height=2, font=('times', 15, ' bold  underline'))
lbl3.place(x=400, y=650)

message2 = tk.Label(window, text="", fg="white", bg="gray25", activeforeground="green", width=30, height=2, font=('times', 15, ' bold '))
message2.place(x=700, y=650)

clearButton = tk.Button(window, text="Xóa", command=clear, fg="steel blue", bg="OliveDrab1", width=20, height=2, activebackground="white", font=('times', 15, ' bold '))
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Xóa", command=clear2, fg="steel blue", bg="OliveDrab1", width=20, height=2, activebackground="white", font=('times', 15, ' bold '))
clearButton2.place(x=950, y=300)
takeImg = tk.Button(window, text="Chụp ảnh", command=TakeImages, fg="steel blue", bg="OliveDrab1", width=20, height=3, activebackground="white", font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
trainImg = tk.Button(window, text="Train ảnh", command=TrainImages, fg="steel blue", bg="OliveDrab1", width=20, height=3, activebackground="white", font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Nhận diện", command=TrackImages, fg="steel blue", bg="OliveDrab1", width=20, height=3, activebackground="white", font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Thoát", command=window.destroy, fg="steel blue", bg="OliveDrab1", width=20, height=3, activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0, font=('times', 30, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.configure(state="disabled", fg="white")
copyWrite.pack(side="left")
copyWrite.place(x=800, y=750)

window.mainloop()
