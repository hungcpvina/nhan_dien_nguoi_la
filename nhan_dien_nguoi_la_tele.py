import cv2
import time
import os
import sys
import tkinter as tk
from tkinter import filedialog
import logging
from multiprocessing import Process, current_process, set_start_method, freeze_support
import requests
from ultralytics import YOLO

# =================== CẤU HÌNH CHUNG ===================

# Các lớp đối tượng cần cảnh báo
classes_to_alert = ["nguoi_la", "nguoi_do"]

# Các hằng số xử lý
MAX_RETRY = 5 
PROCESSING_INTERVAL = 3    # Chạy inference mỗi 3 giây để giảm tải
ALERT_INTERVAL = 60        # Cách nhau 60 giây giữa các lần cảnh báo
REPEATED_ALERT_INTERVAL = 180  # 3 phút giữa các lần cảnh báo cùng một đối tượng
DISPLAY_WIDTH = 400        # Cửa sổ video hiển thị có độ rộng cố định (khoảng 4 inch - ví dụ 400 pixel)
DISPLAY_HEIGHT = 300       # Chiều cao hiển thị (bạn có thể điều chỉnh)
detected_objects = {}  # Lưu thông tin về các đối tượng đã phát hiện
last_alert_time = {}

# =================== HÀM GỬI CẢNH BÁO QUA TELEGRAM ===================
def send_alert_to_telegram(image, object_id):
    try:
        current_time = time.time()
        alert_interval = 180  # 3 phút giữa các lần cảnh báo cùng một đối tượng

        # Kiểm tra nếu đối tượng đã được cảnh báo gần đây
        if object_id in last_alert_time:
            if current_time - last_alert_time[object_id] < alert_interval:
                logging.info(f"Đối tượng {object_id} đã được cảnh báo gần đây. Chờ {alert_interval} giây trước khi gửi lại.")
                return False  # Không gửi cảnh báo ngay

        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'photo': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        data = {'chat_id': CHAT_ID, 'caption': '⚠️ Cảnh báo ! Phát hiện người lạ! ⚠️'}
        response = requests.post(url, files=files, data=data, timeout=10)

        if response.ok:
            logging.info("Đã gửi cảnh báo qua Telegram thành công.")
            last_alert_time[object_id] = current_time  # Cập nhật thời gian cảnh báo cho đối tượng
            return True
        else:
            logging.warning(f"Telegram trả về lỗi: {response.text}")
            return False
    except Exception as e:
        logging.error(f"Lỗi khi gửi cảnh báo Telegram: {e}")
        return False

# =================== HÀM GIAO DIỆN CHỌN THAM SỐ (Tkinter GUI) ===================
def select_parameters():
    params = {}
    root = tk.Tk()
    root.title("Cấu hình Camera & Mô hình")

     # Các biến chứa thông số của giao diện
    telegram_token_var = tk.StringVar()
    chat_id_var = tk.StringVar()
    rtsp_var1 = tk.StringVar()
    rtsp_var2 = tk.StringVar()
    rtsp_var3 = tk.StringVar()
    rtsp_var4 = tk.StringVar() #Thêm trường nhập link rtsp 4
    model_path_var = tk.StringVar()
    alert_folder_var = tk.StringVar(value=os.path.join(os.getcwd(), "alert_images"))
    video_path_var = tk.StringVar()  # 🆕 Thêm biến chứa đường dẫn video mẫu
    # Hàm chọn file mô hình
    def browse_model():
        fp = filedialog.askopenfilename(
            title="Chọn file mô hình (best.pt)",
            filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")]
        )
        if fp:
            model_path_var.set(fp)

    # Hàm chọn thư mục lưu ảnh cảnh báo
    def browse_folder():
        folder = filedialog.askdirectory(title="Chọn thư mục lưu ảnh cảnh báo")
        if folder:
            alert_folder_var.set(folder)

    # Hàm chọn file video mẫu
    def browse_video():
        file_path = filedialog.askopenfilename(
            title="Chọn video mẫu",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            video_path_var.set(file_path)

    # Giao diện nhập Telegram Token
    tk.Label(root, text="Telegram Token:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=telegram_token_var, width=60).grid(row=0, column=1, padx=5, pady=5)

    # Giao diện nhập Telegram Chat ID
    tk.Label(root, text="Telegram Chat ID:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=chat_id_var, width=60).grid(row=1, column=1, padx=5, pady=5)

    # Giao diện nhập 3 RTSP link
    tk.Label(root, text="RTSP Link 1:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=rtsp_var1, width=60).grid(row=2, column=1, padx=5, pady=5)

    tk.Label(root, text="RTSP Link 2:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=rtsp_var2, width=60).grid(row=3, column=1, padx=5, pady=5)

    tk.Label(root, text="RTSP Link 3:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=rtsp_var3, width=60).grid(row=4, column=1, padx=5, pady=5)

        # Giao diện nhập RTSP Link 4
    tk.Label(root, text="RTSP Link 4:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=rtsp_var4, width=60).grid(row=5, column=1, padx=5, pady=5)


        # Nhập đường dẫn file mô hình best.pt
    tk.Label(root, text="Model Path (best.pt):").grid(row=6, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=model_path_var, width=60).grid(row=6, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=browse_model).grid(row=6, column=2, padx=5, pady=5)

    # Nhập thư mục lưu ảnh cảnh báo
    tk.Label(root, text="Alert Image Folder:").grid(row=7, column=0, sticky="w", padx=5, pady=5)
    tk.Entry(root, textvariable=alert_folder_var, width=60).grid(row=7, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=browse_folder).grid(row=7, column=2, padx=5, pady=5)

    # 🆕 Nhập đường dẫn video mẫu (sửa bố trí để không bị đè)
    tk.Label(root, text="Video Test (nếu không có RTSP):").grid(row=8, column=0, sticky="w", padx=5, pady=10)
    tk.Entry(root, textvariable=video_path_var, width=60).grid(row=8, column=1, padx=5, pady=10)
    tk.Button(root, text="Browse", command=lambda: video_path_var.set(filedialog.askopenfilename(title="Chọn video mẫu", filetypes=[("Video Files", "*.mp4 *.avi *.mov")]))).grid(row=8, column=2, padx=5, pady=10)

       # 🆕 Nút Start: Kiểm tra RTSP, nếu trống thì dùng video mẫu
    def on_start():
        params["telegram_token"] = telegram_token_var.get()
        params["chat_id"] = chat_id_var.get()
        params["rtsp_links"] = [rtsp_var1.get(), rtsp_var2.get(), rtsp_var3.get()]
        params["model_path"] = model_path_var.get()
        params["alert_folder"] = alert_folder_var.get()
        params["video_path"] = video_path_var.get() if not any(params["rtsp_links"]) else None
        
        root.destroy()
        run_camera(params)  # 🆕 Gọi xử lý nhận diện ngay sau khi chọn xong

    tk.Button(root, text="Start", command=on_start, width=20).grid(row=9, column=1, pady=10)

    root.mainloop()

    # Kiểm tra nếu thiếu Token hoặc Chat ID thì không chạy tiếp
    if not params["telegram_token"] or not params["chat_id"]:
        logging.error("Lỗi: Telegram Token hoặc Chat ID chưa nhập!")
        return {}

    return params

# =================== HÀM XỬ LÝ CAMERA (Chạy trong tiến trình riêng, không có GUI) ===================
def send_alert_to_telegram(image, telegram_token, chat_id):
    if not telegram_token or not chat_id:
        logging.error("Lỗi: Telegram Token hoặc Chat ID chưa được thiết lập!")
        return False

    try:
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'photo': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
        url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
        data = {'chat_id': chat_id, 'caption': '⚠️ Cảnh báo ! Phát hiện người lạ! ⚠️'}
        response = requests.post(url, files=files, data=data, timeout=10)

        if response.ok:
            logging.info("Đã gửi cảnh báo qua Telegram thành công.")
            return True
        else:
            logging.warning(f"Telegram trả về lỗi: {response.text}")
            return False

    except Exception as e:
        logging.error(f"Lỗi khi gửi cảnh báo Telegram: {e}")
        return False
def run_camera(rtsp_url, window_name, model_path, alert_folder, processing_interval, telegram_token, chat_id, video_path=None):
    logging.info(f"[{window_name}] Khởi động...")

    try:
        model = YOLO(model_path)
        logging.info(f"[{window_name}] Mô hình được tải thành công!")
    except Exception as e:
        logging.error(f"[{window_name}] Lỗi tải mô hình: {e}")
        return

    retry_count = 0
    cap = None

    # 🔹 Kiểm tra nếu RTSP trống thì dùng video mẫu
    if rtsp_url:
        logging.info(f"[{window_name}] Đang kết nối RTSP: {rtsp_url}")
        while retry_count < MAX_RETRY:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
            
            if cap.isOpened():
                logging.info(f"[{window_name}] Kết nối RTSP thành công!")
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                logging.info(f"[{window_name}] FPS thực tế của luồng: {actual_fps}")
                break
            else:
                logging.warning(f"[{window_name}] Lỗi mở RTSP! Đang thử lại ({retry_count+1}/{MAX_RETRY})...")
                cap.release()
                time.sleep(3)
                retry_count += 1
    else:
        logging.info(f"[{window_name}] Đang chạy demo với video mẫu: {video_path}")
        cap = cv2.VideoCapture(video_path)

    if not cap or not cap.isOpened():
        logging.error(f"[{window_name}] Không thể mở nguồn video sau {MAX_RETRY} lần thử!")
        return

    paused = False
    last_inference = 0
    last_alert_time = 0

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    while True:
        time.sleep(0.1)
       # 🔹 Bỏ qua khung hình cũ để lấy khung hình mới nhất
        cap.grab()
        ret, frame = cap.retrieve()

        if not ret:
            if rtsp_url:
                logging.warning(f"[{window_name}] Mất kết nối RTSP, đang thử lại...")
                cap.release()
                time.sleep(3)
                retry_count = 0
                while retry_count < MAX_RETRY:
                    cap = cv2.VideoCapture(rtsp_url)
                    if cap.isOpened():
                        logging.info(f"[{window_name}] Kết nối RTSP thành công!")
                        break
                    else:
                        logging.warning(f"[{window_name}] Lỗi mở RTSP! Đang thử lại ({retry_count+1}/{MAX_RETRY})...")
                        cap.release()
                        time.sleep(3)
                        retry_count += 1
                if not cap or not cap.isOpened():
                    logging.error(f"[{window_name}] Không thể kết nối lại RTSP sau {MAX_RETRY} lần thử!")
                    break
                continue
            else:
                logging.error(f"[{window_name}] Video mẫu kết thúc!")
                break

        # 🔹 Giữ nguyên phần chạy inference mỗi vài giây để giảm tải
        current_time = time.time()
        display_frame = frame.copy()

        if current_time - last_inference >= processing_interval:
            last_inference = current_time
            try:
                results = model(display_frame)[0]
                alert_detected = False
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names.get(cls_id, "unknown") if isinstance(model.names, dict) else model.names[cls_id]
                    if class_name in classes_to_alert:
                        alert_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(display_frame, f"{class_name} {conf:.2f}",
                                    (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if alert_detected and (current_time - last_alert_time >= ALERT_INTERVAL):
                        filename = os.path.join(alert_folder, f"alert_{int(current_time)}.jpg")
                        cv2.imwrite(filename, display_frame)
                        logging.info(f"[{window_name}] Lưu ảnh cảnh báo: {filename}")
                        if send_alert_to_telegram(display_frame, telegram_token, chat_id):
                            logging.info(f"[{window_name}] Đã gửi cảnh báo qua Telegram.")
                        last_alert_time = current_time

            except Exception as e:
                logging.error(f"[{window_name}] Lỗi khi inference: {e}")
# Kiểm tra phím: 'p' để tạm dừng, 'q' để thoát
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            paused = not paused
        if key == ord('q'):
            logging.info(f"[{window_name}] Người dùng nhấn 'q'. Thoát luồng.")
            break

        if paused:
            pause_frame = frame.copy()
            cv2.putText(pause_frame, "TẠM DỪNG - Nhấn 'p' để tiếp tục", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(window_name, pause_frame)
            continue
    cap.release()
    cv2.destroyWindow(window_name)
    logging.info(f"[{window_name}] Đóng kết nối.")



# =================== HÀM MAIN ===================
def select_parameters():
    # Chỉ cho phép tiến trình chính chạy giao diện
    if current_process().name != "MainProcess":
        logging.error("Tiến trình con không được phép chạy giao diện!")
        return {}  # Ngăn tiến trình con mở lại giao diện

    params = {
        "rtsp_links": [],
        "model_path": "",
        "alert_folder": "",
        "telegram_token": "",
        "chat_id": "",
        "video_path": ""  # 🆕 Thêm biến chứa đường dẫn video mẫu
    }

    root = tk.Tk()
    root.title("Cấu hình Camera & Mô hình")

    # Nhập Telegram Token
    tk.Label(root, text="Telegram Token:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    telegram_token_var = tk.StringVar()
    tk.Entry(root, textvariable=telegram_token_var, width=60).grid(row=0, column=1, padx=5, pady=5)

    # Nhập Telegram Chat ID
    tk.Label(root, text="Telegram Chat ID:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    chat_id_var = tk.StringVar()
    tk.Entry(root, textvariable=chat_id_var, width=60).grid(row=1, column=1, padx=5, pady=5)

    # Nhập RTSP Link 1
    tk.Label(root, text="RTSP Link 1:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    rtsp_var1 = tk.StringVar(value="rtsp://admin:pa123456@camera1_url?rtsp_transport=tcp")
    tk.Entry(root, textvariable=rtsp_var1, width=60).grid(row=2, column=1, padx=5, pady=5)

    tk.Label(root, text="RTSP Link 2:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
    rtsp_var2 = tk.StringVar(value="rtsp://admin:pa123456@camera2_url?rtsp_transport=tcp")
    tk.Entry(root, textvariable=rtsp_var2, width=60).grid(row=3, column=1, padx=5, pady=5)

    tk.Label(root, text="RTSP Link 3:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
    rtsp_var3 = tk.StringVar(value="rtsp://admin:pa123456@camera3_url?rtsp_transport=tcp")
    tk.Entry(root, textvariable=rtsp_var3, width=60).grid(row=4, column=1, padx=5, pady=5)

        # Nhập RTSP Link 4
    tk.Label(root, text="RTSP Link 4:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
    rtsp_var4 = tk.StringVar(value="rtsp://admin:pa123456@camera4_url?rtsp_transport=tcp")
    tk.Entry(root, textvariable=rtsp_var4, width=60).grid(row=5, column=1, padx=5, pady=5)

    # Nhập đường dẫn file mô hình best.pt
    tk.Label(root, text="Model Path (best.pt):").grid(row=6, column=0, sticky="w", padx=5, pady=5)
    model_path_var = tk.StringVar()
    tk.Entry(root, textvariable=model_path_var, width=60).grid(row=6, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=lambda: model_path_var.set(filedialog.askopenfilename())).grid(row=6, column=2, padx=5, pady=5)

    # Nhập thư mục ảnh cảnh báo
    tk.Label(root, text="Alert Image Folder:").grid(row=7, column=0, sticky="w", padx=5, pady=5)
    alert_folder_var = tk.StringVar(value=os.path.join(os.getcwd(), "alert_images"))
    tk.Entry(root, textvariable=alert_folder_var, width=60).grid(row=7, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=lambda: alert_folder_var.set(filedialog.askdirectory())).grid(row=7, column=2, padx=5, pady=5)

    # 🆕 Nhập đường dẫn video mẫu
    tk.Label(root, text="Video Test (nếu không có RTSP):").grid(row=8, column=0, sticky="w", padx=5, pady=5)
    video_path_var = tk.StringVar()
    tk.Entry(root, textvariable=video_path_var, width=60).grid(row=8, column=1, padx=5, pady=5)
    tk.Button(root, text="Browse", command=lambda: video_path_var.set(filedialog.askopenfilename(title="Chọn video mẫu", filetypes=[("Video Files", "*.mp4 *.avi *.mov")]))).grid(row=8, column=2, padx=5, pady=5)

    # Nút Start: Lưu các thông tin và đóng cửa sổ
    def on_start():
        params["telegram_token"] = telegram_token_var.get()
        params["chat_id"] = chat_id_var.get()
        params["rtsp_links"] = [rtsp_var1.get(), rtsp_var2.get(), rtsp_var3.get(), rtsp_var4.get()]  # 🆕 Thêm RTSP Link 4
        params["model_path"] = model_path_var.get()
        params["alert_folder"] = alert_folder_var.get()
        params["video_path"] = video_path_var.get() if not any(params["rtsp_links"]) else None  # 🆕 Nếu không có RTSP, dùng video test
        
        root.destroy()
        run_camera(params)  # 🆕 Gọi xử lý nhận diện ngay sau khi chọn xong

    tk.Button(root, text="Start", command=on_start, width=20).grid(row=9, column=1, pady=15)

    root.mainloop()

    # Kiểm tra nếu thiếu Token hoặc Chat ID thì báo lỗi
    if not params["telegram_token"] or not params["chat_id"]:
        logging.error("Lỗi: Telegram Token hoặc Chat ID chưa nhập!")
        return {}

    logging.info(f"Dữ liệu đã nhập: {params}")
    return params
# =================== HÀM MAIN ===================
if __name__ == "__main__":
    # Dành cho việc chạy EXE trên Windows với multiprocessing
    freeze_support()
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    # Chỉ tiến trình chính mới thực hiện các bước dưới đây
    if current_process().name != "MainProcess":
        exit()

    params = select_parameters()
    if not params or not params.get("rtsp_links"):
        logging.error("Lỗi: Không nhận được dữ liệu từ select_parameters()")
        exit()
    telegram_token = params.get("telegram_token")
    chat_id = params.get("chat_id")
    rtsp_links = params.get("rtsp_links", [])
    model_path = params.get("model_path", "")
    alert_folder = params.get("alert_folder", "")
    video_path = params.get("video_path")  # 🆕 Thêm video mẫu

    if not os.path.exists(model_path):
        logging.error("Thiếu dữ liệu đầu vào!")
        exit()

    logging.info("Chương trình bắt đầu xử lý camera...")

    processes = []
    # 🔹 Nếu có RTSP, chạy từng camera
    if any(rtsp_links):
        for idx, rtsp_url in enumerate(rtsp_links):
            window_name = f"Camera {idx+1}"
            p = Process(target=run_camera, args=(rtsp_url, window_name, model_path, alert_folder, PROCESSING_INTERVAL, telegram_token, chat_id))
            p.start()
            processes.append(p)
    else:
        # 🔹 Nếu không có RTSP, chạy video mẫu
        window_name = "Demo Video"
        p = Process(target=run_camera, args=(None, window_name, model_path, alert_folder, PROCESSING_INTERVAL, telegram_token, chat_id, video_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()