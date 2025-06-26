import cv2
import time
import os
import math
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import logging
import numpy as np
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%H:%M:%S")
from multiprocessing import Process, current_process, set_start_method, freeze_support
import requests
from ultralytics import YOLO

# =================== CẤU HÌNH CHUNG ===================

# Các lớp đối tượng cần cảnh báo
classes_to_alert = ["nguoi_la", "nguoi_do"]

# Các hằng số xử lý
MAX_RETRY = 5
PROCESSING_INTERVAL = 0    # Chạy inference mỗi 1 giây để giảm tải
ALERT_INTERVAL = 60        # Cách nhau 60 giây giữa các lần cảnh báo
REPEATED_ALERT_INTERVAL = 300  # 5 phút giữa các lần cảnh báo cùng một đối tượng
window_name = "Camera Feed"
DISPLAY_WIDTH = 400        # Cửa sổ video hiển thị có độ rộng cố định (khoảng 4 inch - ví dụ 400 pixel)
DISPLAY_HEIGHT = 300       # Chiều cao hiển thị (bạn có thể điều chỉnh)
detected_objects = {}  # Lưu thông tin về các đối tượng đã phát hiện
last_alert_time = {}
has_lost_connection = False  # 🆕 Biến trạng thái để ngăn spam cảnh báo
motion_check_interval = 3
frame_idx = 0
full_threshold = 5000

DURATION_THRESH = 1      # giây
MOVE_THRESH     = 1000       # pixel: vùng di chuyển nhỏ
MATCH_THRESH    = 100       # pixel: để match box với tracker
MAX_IDLE        = 1      # giây: xóa tracker không thấy update


# =================== HÀM GỬI CẢNH BÁO QUA TELEGRAM ===================
def send_alert_to_telegram(image,telegram_token, chat_id, object_id=None):
    if not telegram_token or not chat_id:
        logging.error("Lỗi: Telegram Token hoặc Chat ID chưa được thiết lập!")
        return False
    try:
        current_time = time.time()
        alert_interval = 300  # 5 phút giữa các lần cảnh báo cùng một đối tượng
        # Kiểm tra nếu đối tượng đã được cảnh báo gần đây
        if object_id in last_alert_time:
            if current_time - last_alert_time[object_id] < alert_interval:
                logging.info(f"Đối tượng {object_id} đã được cảnh báo gần đây. Chờ {alert_interval} giây trước khi gửi lại.")
                return False  # Không gửi cảnh báo ngay

        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'photo': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
        url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
        data = {'chat_id': chat_id, 'caption': '⚠️ Cảnh báo ! Phát hiện người lạ!'}
        response = requests.post(url, files=files, data=data, timeout=10)

        if response.ok:
            logging.info("Đã gửi cảnh báo qua Telegram thành công.")
            last_alert_time[object_id] = current_time  # Cập nhật thời gian cảnh báo cho đối tượng
            return True
        else:
            logging.warning(f"Telegram trả về lỗi: {response.text}")
            return False
    except requests.RequestException as e:
        logging.error(f"Lỗi khi gửi ảnh Telegram (RequestException): {e}")
        return False
    except Exception as e:
        logging.error(f"Lỗi khi gửi cảnh báo Telegram: {e}")
        return False
def send_zone_alert_to_telegram(image, telegram_token, chat_id):
    if not telegram_token or not chat_id:
        logging.error("Thiếu Token hoặc Chat ID khi gửi alert zone!")
        return False
    try:
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'photo': ('zone_alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
        url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
        data = {'chat_id': chat_id, 'caption': '🚧 CẢNH BÁO: Có người đã vào vùng nguy hiểm!'}
        response = requests.post(url, files=files, data=data, timeout=10)
        return response.ok
    except Exception as e:
        logging.error(f"Lỗi khi gửi cảnh báo Zone: {e}")
        return False
# =================== HÀM GIAO DIỆN CHỌN THAM SỐ (Tkinter GUI) ===================
def make_transparent(image_path, opacity=30):  # 🆕 Độ mờ 30% (có thể chỉnh thấp hơn)
    image = Image.open(image_path).convert("RGBA")
    alpha = image.split()[3]
    alpha = alpha.point(lambda p: int(p * opacity / 100))
    image.putalpha(alpha)
    return ImageTk.PhotoImage(image)

def select_parameters():
    # 🆕 Khởi tạo params đúng cách
    params = {"telegram_token": "", "chat_id": "", "rtsp_links": [],"enable_zones": [], "model_path": "", "alert_folder": "", "video_path": ""}
    root = tk.Tk()
    root.title("Cấu hình Camera & Mô hình")
    root.configure(bg="#f0f0f0")  # Màu nền của cửa sổ
    root.geometry("800x600")
    style = ttk.Style()
    style.theme_use("clam")
     # 🆕 Các biến chứa thông số của giao diện
    telegram_token_var = tk.StringVar()
    chat_id_var = tk.StringVar()
    rtsp_var1 = tk.StringVar()
    rtsp_var2 = tk.StringVar()
    rtsp_var3 = tk.StringVar()
    rtsp_var4 = tk.StringVar()
    model_path_var = tk.StringVar()
    alert_folder_var = tk.StringVar()
    video_path_var = tk.StringVar()
    # ==== thêm biến để lưu trạng thái enable zone ====
    enable_zone_var1 = tk.BooleanVar(value=False)
    enable_zone_var2 = tk.BooleanVar(value=False)
    enable_zone_var3 = tk.BooleanVar(value=False)
    enable_zone_var4 = tk.BooleanVar(value=False)
    demo_zone_var = tk.BooleanVar(value=False)


# 🆕 Hàm chọn file mô hình
    def browse_model():
        fp = filedialog.askopenfilename(title="Chọn file mô hình (best.pt)", filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")])
        if fp:
            model_path_var.set(fp)
    # 🆕 Hàm chọn thư mục lưu ảnh cảnh báo
    def browse_folder():
        folder = filedialog.askdirectory(title="Chọn thư mục lưu ảnh cảnh báo")
        if folder:
            alert_folder_var.set(folder)
    # 🆕 Hàm chọn file video mẫu
    def browse_video():
        file_path = filedialog.askopenfilename(title="Chọn video mẫu", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if file_path:
            video_path_var.set(file_path)
      # 🆕 Hiển thị logo nền với độ trong suốt
    bg_image = make_transparent("D:/nhan_dien_nguoi_la/logo.jpg", opacity=30)
    # 🆕 Đặt logo làm nền bằng Label
    bg_label = tk.Label(root, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    bg_label.lower()  # 🆕 Đảm bảo logo không che phần nhập liệu
# 🆕 Giữ ảnh để tránh bị xoá (Đặt sau khi `bg_label` đã khởi tạo)
    bg_label.image = bg_image
     # 🆕 Tạo canvas để đặt ảnh nền phía dưới
    canvas = tk.Canvas(root, width=800, height=600)
    canvas.place(x=0, y=0, width=800, height=600)
    canvas.create_image(400, 300, anchor="center", image=bg_image)
    canvas.image = bg_image

   # 🆕 Thêm thông tin tác giả phần mềm vào góc bên phải (bạn có thể điều chỉnh vị trí x, y cho phù hợp)
    tk.Label(root,
         text="Phát triển bởi: C.P Vietnam Corporation \nSwine Veterinary and Biosecurity Department \n(Bộ phận Thú y và Phòng dịch)",
         bg="#f0f0f0",
         fg="black",
         font=("Helvetica", 10, "italic")).place(x=510, y=10)
    tk.Label(root, text="Telegram Token:", bg="#f0f0f0", fg="black").place(x=50, y=50)
    tk.Entry(root, textvariable=telegram_token_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=50)

    tk.Label(root, text="Telegram Chat ID:", bg="#f0f0f0", fg="black").place(x=50, y=100)
    tk.Entry(root, textvariable=chat_id_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=100)

    tk.Label(root, text="RTSP Link 1:", bg="#f0f0f0", fg="black").place(x=50, y=150)
    tk.Entry(root, textvariable=rtsp_var1, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=150)
    tk.Checkbutton(root,
               text="Vẽ vùng cảnh báo",
               variable=enable_zone_var1,
               bg="#f0f0f0")\
    .place(x=600, y=150)
    tk.Label(root, text="RTSP Link 2:", bg="#f0f0f0", fg="black").place(x=50, y=200)
    tk.Entry(root, textvariable=rtsp_var2, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=200)
    tk.Checkbutton(root,
               text="Vẽ vùng cảnh báo",
               variable=enable_zone_var2,
               bg="#f0f0f0")\
    .place(x=600, y=200)

    tk.Label(root, text="RTSP Link 3:", bg="#f0f0f0", fg="black").place(x=50, y=250)
    tk.Entry(root, textvariable=rtsp_var3, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=250)
    tk.Checkbutton(root,
               text="Vẽ vùng cảnh báo",
               variable=enable_zone_var3,
               bg="#f0f0f0")\
    .place(x=600, y=250)
    tk.Label(root, text="RTSP Link 4:", bg="#f0f0f0", fg="black").place(x=50, y=300)
    tk.Entry(root, textvariable=rtsp_var4, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=300)
    tk.Checkbutton(root,
               text="Vẽ vùng cảnh báo",
               variable=enable_zone_var4,
               bg="#f0f0f0")\
    .place(x=600, y=300)
    tk.Label(root, text="Model Path (best.pt):", bg="#f0f0f0", fg="black").place(x=50, y=350)
    tk.Entry(root, textvariable=model_path_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=350)
    tk.Button(root, text="Browse", command=browse_model, bg="#d3d3d3", fg="black", relief="raised", bd=3).place(x=510, y=348)

    tk.Label(root, text="Alert Image Folder:", bg="#f0f0f0", fg="black").place(x=50, y=400)
    tk.Entry(root, textvariable=alert_folder_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=400)
    tk.Button(root, text="Browse", command=browse_folder, bg="#d3d3d3", fg="black", relief="raised", bd=3).place(x=510, y=398)

    tk.Label(root, text="Video Test:", bg="#f0f0f0", fg="black").place(x=50, y=450)
    tk.Entry(root, textvariable=video_path_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=450)
    tk.Button(root, text="Browse", command=browse_video, bg="#d3d3d3", fg="black", relief="raised", bd=3).place(x=510, y=448)
    tk.Checkbutton(root,
                   text="Vẽ vùng cảnh báo",
                   variable=demo_zone_var,
                   bg="#f0f0f0")\
      .place(x=600, y=450)
    # 🆕 Nút Start
    def on_start():
        params["telegram_token"] = telegram_token_var.get()
        params["chat_id"] = chat_id_var.get()
        params["rtsp_links"] = [rtsp_var1.get(), rtsp_var2.get(), rtsp_var3.get(), rtsp_var4.get()]
        params["enable_zones"]  = [
        enable_zone_var1.get(),
        enable_zone_var2.get(),
        enable_zone_var3.get(),
        enable_zone_var4.get()]
        params["model_path"] = model_path_var.get()
        params["alert_folder"] = alert_folder_var.get()
        params["video_path"] = video_path_var.get() if not any(params["rtsp_links"]) else None
        params["demo_enable_zone"] = demo_zone_var.get()
        root.destroy()
        run_camera(params)
# Ví dụ sử dụng ttk.Button thay cho tk.Button:
    ttk.Button(root, text="Start", command=on_start, style="TButton").place(x=250, y=500)
    root.mainloop()

    # Kiểm tra nếu thiếu Token hoặc Chat ID thì không chạy tiếp
    if not params["telegram_token"] or not params["chat_id"]:
        logging.error("Lỗi: Telegram Token hoặc Chat ID chưa nhập!")
        return {}
    return params

# =================== HÀM XỬ LÝ CAMERA (Chạy trong tiến trình riêng, không có GUI) ===================
def send_alert_to_telegram(image, telegram_token, chat_id, object_id=None):
    if not telegram_token or not chat_id:
        logging.error("Lỗi: Telegram Token hoặc Chat ID chưa được thiết lập!")
        return False
    try:
        current_time = time.time()
        alert_interval = 300  # 5 phút giữa các lần cảnh báo cùng một đối tượng
        # Kiểm tra nếu đối tượng đã được cảnh báo gần đây
        if object_id in last_alert_time:
            if current_time - last_alert_time[object_id] < alert_interval:
                logging.info(f"Đối tượng {object_id} đã được cảnh báo gần đây. Chờ {alert_interval} giây trước khi gửi lại.")
                return False  # Không gửi cảnh báo ngay
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'photo': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
        url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
        data = {'chat_id': chat_id, 'caption': '⚠️ Cảnh báo ! Phát hiện người lạ!'}
        response = requests.post(url, files=files, data=data, timeout=10)
        if response.ok:
            logging.info("Đã gửi cảnh báo qua Telegram thành công.")
            return True
        else:
            logging.warning(f"Telegram trả về lỗi: {response.text}")
            return False
    except requests.RequestException as e:
        logging.error(f"Lỗi khi gửi ảnh Telegram (RequestException): {e}")
        return False
    except Exception as e:
        logging.error(f"Lỗi khi gửi ảnh Telegram (Unexpected): {e}")
        return False
def send_zone_alert_to_telegram(image, telegram_token, chat_id):
    if not telegram_token or not chat_id:
        logging.error("Thiếu Token hoặc Chat ID khi gửi alert zone!")
        return False
    try:
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'photo': ('zone_alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
        url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
        data = {'chat_id': chat_id, 'caption': '🚧 CẢNH BÁO: Có người đã vào vùng nguy hiểm!'}
        response = requests.post(url, files=files, data=data, timeout=10)
        return response.ok
    except Exception as e:
        logging.error(f"Lỗi khi gửi cảnh báo Zone: {e}")
        return False
# ================ Hàm mới: gửi tin nhắn văn bản ===================
def send_text_to_telegram(message, telegram_token, chat_id):
    if not telegram_token or not chat_id:
        logging.error("Lỗi: Telegram Token hoặc Chat ID chưa được thiết lập!")
        return False
    try:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': message}
        response = requests.post(url, json=payload, timeout=10)

        if response.ok:
            logging.info("Đã gửi tin nhắn qua Telegram thành công.")
            return True
        else:
            logging.warning(f"Telegram trả về lỗi: {response.text}")
            return False
    except requests.RequestException as e:
        logging.error(f"Lỗi khi gửi tin nhắn Telegram (RequestException): {e}")
        return False
    except Exception as e:
        logging.error(f"Lỗi khi gửi tin nhắn Telegram (Unexpected): {e}")
        return False
# ============== Hàm kiểm tra kết nối camera RTSP ================
def check_camera_connection(rtsp_url, window_name, telegram_token, chat_id):
    cap = cv2.VideoCapture(rtsp_url)
    opened = cap.isOpened()
    cap.release()
    if not opened:
        logging.error(f"❌ [{window_name}] Mất kết nối camera!")
        send_text_to_telegram(f"⚠️ Cảnh báo! Mất kết nối với {window_name}", telegram_token, chat_id)
        return False
    return True
# ===================== Hàm chính chạy camera =====================
def is_inside_zone(c, polygon):
    """Trả về True nếu point c=(x,y) nằm trong polygon."""
    # pointPolygonTest >=0 nghĩa nằm trong hoặc trên biên
    return cv2.pointPolygonTest(polygon, c, False) >= 0

def run_camera(params):
    # Trích xuất tham số từ dict
    rtsp_url = params.get("rtsp_url") # Thêm rtsp_url riêng nếu có
    window_name = params.get("window_name")
    model_path = params.get("model_path")
    alert_folder = params.get("alert_folder")
    processing_interval = params.get("processing_interval")
    telegram_token = params.get("telegram_token")
    chat_id = params.get("chat_id")
    enable_danger_zone = params.get("enable_danger_zone") # renamed for clarity
    video_path = params.get("video_path")

    logging.info(f"[{window_name}] Khởi động...")
    rtsp_url = rtsp_url.strip() if rtsp_url else None
    # Tải model
    try:
        model = YOLO(model_path)
        logging.info(f"[{window_name}] Mô hình được tải thành công!")
    except Exception as e:
        logging.error(f"[{window_name}] Lỗi tải mô hình: {e}")
        return
    # Mở kết nối RTSP hoặc video mẫu
    retry_count = 0
    cap = None
    if rtsp_url:
        if not check_camera_connection(rtsp_url, window_name, telegram_token, chat_id):
            return
        while retry_count < MAX_RETRY:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_FPS, 25)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                logging.info(f"[{window_name}] Kết nối RTSP thành công!")
                break
            cap.release()
            retry_count += 1
            time.sleep(3)
        if not cap or not cap.isOpened():
            logging.error(f"[{window_name}] Không thể mở RTSP sau {MAX_RETRY} lần thử!")
            send_text_to_telegram(
                f"❌ Không thể kết nối RTSP với {window_name} sau {MAX_RETRY} lần thử.",
                telegram_token, chat_id)
            return
    else:
        logging.info(f"[{window_name}] Chạy với video mẫu: {video_path}")
        cap = cv2.VideoCapture(video_path)
     # Biến trạng thái kết nối
    paused = False
    freeze_frame = None  # Biến lưu ảnh dừng (freeze) khi tạm dừng
    has_lost_connection = False
    last_inference = 0
    last_alert_time = 0
     # —— KHỞI TẠO CHO TRACKING STATIC FILTER ——
    trackers = []        # danh sách tracker cho mỗi camera
    next_id  = 0         # id tiếp theo cho tracker
    entered_zone_ids = set()
    entered_zone_log_ids = set()
    zone_last_alert = {}        # lưu last alert time cho mỗi box_id trong zone
    stranger_last_alert = {}    # lưu last alert time cho mỗi stranger (theo ID hoặc name)
    ZONE_REPEAT_INTERVAL = 300  # 5 phút
    STRANGER_REPEAT_INTERVAL = 300  # nếu cần, 5 phút giữa 2 alert stranger cùng 1 người

     # Thiết lập cửa sổ hiển thị
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # 🆕 Khởi tạo biến lưu trạng thái vẽ vùng cảnh báo
     # zone_data giờ sẽ lưu list điểm và polygon khi hoàn thành
    zone_data = {
        "points": [],     # list các đỉnh (x,y)
        "polygon": None}

    # 🆕 Hàm callback để vẽ vùng bằng chuột
    def draw_zone(event, x, y, flags, _):
    # Chỉ vẽ khi đang ở mode vẽ và chưa hoàn thành polygon
        if not enable_danger_zone or zone_data["polygon"] is not None:
            return

    # Left‐click: thêm đỉnh
        if event == cv2.EVENT_LBUTTONDOWN:
            zone_data["points"].append((x, y))
            logging.info(f"[{window_name}] +Vertex {(x,y)}")

    # Right‐click: kết thúc polygon khi >=4 điểm
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(zone_data["points"]) >= 4:
                zone_data["polygon"] = np.array(zone_data["points"], np.int32)
                logging.info(f"[{window_name}] Polygon done: {zone_data['polygon']}")
        else:
            logging.warning(f"[{window_name}] Cần ít nhất 4 điểm, hiện có {len(zone_data['points'])}")

    # 🆕 Gán callback, param = enable_danger_zone
    cv2.setMouseCallback(window_name, draw_zone)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"[{window_name}] FPS luồng: {cap_fps}")
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
    motion_check_interval = 1
    frame_idx = 0
    consecutive_detections = 0
    detection_required = 1
    min_confidence   = 0.6
    full_threshold = 5000
    entered_zone_ids = set()

    def centroid(x1, y1, x2, y2):
        return ((x1+x2)//2, (y1+y2)//2)
    def dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    while True:
        current_time = time.time()

        # 1) Nếu chưa đến lúc làm inference thì vứt 1 frame cũ và loop tiếp
        if current_time - last_inference < PROCESSING_INTERVAL:
            cap.grab()           # xả 1 frame
            time.sleep(0)
            continue
        # 2) Flush buffer: gọi grab liên tục để đẩy hết frame cũ
        cap.grab()
        # 3) Lấy frame “nóng” nhất
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            # Xử lý mất kết nối RTSP
            if rtsp_url:
                if not has_lost_connection:
                    logging.warning(f"[{window_name}] Mất kết nối RTSP, đang gửi thông báo...")
                    send_text_to_telegram(
                        f"⚠️ Camera {window_name} mất kết nối RTSP. Đang gửi thông báo...",
                        telegram_token, chat_id
                    )
                    has_lost_connection = True
                cap.release()
                time.sleep(3)
                # Thử reconnect tối đa MAX_RETRY lần
                retry = 0
                while retry < MAX_RETRY and not has_lost_connection:
                    cap = cv2.VideoCapture(rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if cap.isOpened():
                        logging.info(f"[{window_name}] Đã kết nối lại RTSP.")
                        send_text_to_telegram(
                            f"✅ Camera {window_name} đã kết nối lại thành công.",
                            telegram_token, chat_id
                        )
                        has_lost_connection = False
                        break
                    retry += 1
                    time.sleep(3)
                # nếu vẫn chưa connect được thì thoát loop chính
                if not cap or not cap.isOpened():
                    logging.error(f"[{window_name}] Không thể reconnect sau {MAX_RETRY} lần.")
                    break
                continue
            else:
                # video file hết
                logging.error(f"[{window_name}] Video mẫu đã kết thúc!")
                break
            # 🔹 Chuẩn bị frame & frame‐skipping
        current_time  = time.time()
        display_frame = frame.copy()
            # —— CHÈN VẼ VÙNG CẢNH BÁO ——
        if enable_danger_zone and zone_data["polygon"] is None:
        # 1) Hướng dẫn
                cv2.putText(display_frame,
                "Left-click: add vertex | Right-click: finish (>4 pts) | Q: skip",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        # 2) Vẽ các đoạn nối giữa các điểm đã chọn
                pts = zone_data["points"]
                if len(pts) >= 2:
                    arr = np.array(pts, np.int32)
                    cv2.polylines(display_frame,
                        [arr],
                        isClosed=False,
                        color=(0,255,255),
                        thickness=2)
        # 3) Vẽ các điểm đỉnh rõ hơn
                for (x,y) in pts:
                    cv2.circle(display_frame, (x,y), 4, (0,255,255), -1)
        # 4) Hiển thị & bắt phím
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    enable_danger_zone = False
                continue

    # ——————— Khi polygon đã hoàn thành ———————
        if zone_data["polygon"] is not None:
                cv2.polylines(display_frame,
                    [zone_data["polygon"]],
                    isClosed=True,   # đóng kín
                    color=(0,255,255),
                    thickness=2)

    # … tiếp code xử lý inference, alert, hiển thị cuối cùng …
        cv2.imshow(window_name, display_frame)
                # --- CHỈ SKIP MOTION/FRAME CHO RTSP ---
        if rtsp_url:
                    fgmask = fgbg.apply(frame)
                    motion_pixels = cv2.countNonZero(fgmask)
                    if motion_pixels < full_threshold:
                        logging.debug(
                            f"[{window_name}] Motion pixels={motion_pixels} < full_th={full_threshold}, skip")
                        cv2.imshow(window_name, display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue
                    frame_idx += 1
                    if frame_idx % motion_check_interval != 0:
                        cv2.imshow(window_name, display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue
                # 🔹 Chạy inference & tracking khi tới lượt
        if rtsp_url is None or current_time - last_inference >= processing_interval:
                    last_inference = current_time
                    detections_this_frame = False
                    try:
            # ─── inference & tracking mới ───────────────────────────────────────────
                        results = model(display_frame)[0]
                        logging.info(f"[{window_name}] Model trả về {len(results.boxes)} boxes")
                        new_trackers = []
                        for box in results.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id        = int(box.cls[0])
                            conf          = float(box.conf[0])
                            name          = model.names.get(cls_id, str(cls_id))
                            c             = ((x1 + x2)//2, (y1 + y2)//2)
                # 1) Match với tracker để lấy obj_id
                            matched = None
                            for tr in trackers:
                                if dist(tr['positions'][-1], c) < MATCH_THRESH:
                                    matched = tr
                                    break
                            if matched:
                                matched['positions'].append(c)
                                matched['last'] = current_time
                            else:
                                matched = {
                                    'id'       : next_id,
                                    'positions': [c],
                                    'start'    : current_time,
                                    'last'     : current_time,
                                    'static'   : False
                                }
                                next_id += 1
                            obj_id = matched['id']
                            new_trackers.append(matched)
                # 2) Luôn vẽ nhạt cho mọi "nguoi_*"
                            if name.startswith("nguoi"):
                                cv2.rectangle(display_frame, (x1,y1), (x2,y2), (200,200,200), 1)
                                cv2.putText(display_frame,
                                            f"{name} {conf:.2f}",
                                            (x1, y1-5),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (200,200,200), 1)
                # 3) Xác định trạng thái
                            in_zone        = (
                                enable_danger_zone
                                and zone_data["polygon"] is not None
                                and is_inside_zone(c, zone_data["polygon"])
                            )
                            is_zone_violate = in_zone and conf >= 0.1
                            is_stranger     = (not in_zone
                                and name in classes_to_alert
                                and conf >= 0.6)
                # 4) Zone‐alert: 1 lần/5′ theo obj_id
                            if is_zone_violate:
                                last_z = zone_last_alert.get(obj_id, 0)
                                if current_time - last_z >= ZONE_REPEAT_INTERVAL:
                                    zone_last_alert[obj_id] = current_time
                                    logging.info(
                                        f"[{window_name}] 🚧 Zone alert id={obj_id}, {name} at {c}"
                                    )
                                    cv2.rectangle(display_frame, (x1,y1),(x2,y2),(0,0,255),2)
                                    cv2.putText(display_frame, "DANGER ZONE",
                                                (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
                                    send_zone_alert_to_telegram(display_frame.copy(),
                                                telegram_token, chat_id)
                                else:
                                    cv2.rectangle(display_frame, (x1,y1),(x2,y2),(0,0,255),2)
                                    cv2.putText(display_frame, "DANGER ZONE",
                                                (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
                                continue
                # 5) Stranger‐alert ngoài zone: 1 lần/5′ theo obj_id
                            if is_stranger:
                                last_s = stranger_last_alert.get(obj_id, 0)
                                if current_time - last_s >= STRANGER_REPEAT_INTERVAL:
                                    stranger_last_alert[obj_id] = current_time
                                    logging.info(
                                        f"[{window_name}] 🔴 Stranger alert id={obj_id}, {name} at {c}"
                                    )
                                    cv2.rectangle(display_frame,(x1,y1),(x2,y2),(0,0,255),2)
                                    cv2.putText(display_frame,f"{name} {conf:.2f}",
                                                (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                                    send_alert_to_telegram(display_frame.copy(),
                                            telegram_token, chat_id)
                                else:
                                    cv2.rectangle(display_frame,(x1,y1),(x2,y2),(0,0,255),2)
                                    cv2.putText(display_frame,f"{name} {conf:.2f}",
                                                (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                                continue
                    # — 6) Static‐filter & vẽ dành riêng cho nguoi_la / nguoi_do
                            if name in classes_to_alert:
                                xs = [p[0] for p in matched['positions']]
                                ys = [p[1] for p in matched['positions']]
                                if (current_time - matched['start'] >= DURATION_THRESH
                                    and max(xs)-min(xs) <= MOVE_THRESH
                                    and max(ys)-min(ys) <= MOVE_THRESH):
                                    matched['static'] = True
                                if not matched['static']:
                                    detections_this_frame = True
                                    cv2.rectangle(display_frame,(x1,y1),(x2,y2),(0,0,255),2)
                                    cv2.putText(display_frame,
                                            f"{name} {conf:.2f}",
                                            (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)			
            # dọn tracker idle & cập nhật
                        trackers = [
                            tr for tr in new_trackers
                            if (current_time - tr['last']) <= MAX_IDLE
                        ]
                        
                    except Exception as e:
                        logging.error(f"[{window_name}] Lỗi khi inference: {e}")

            # — Kiểm tra phím: 'p' để tạm dừng, 'q' để thoát
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
                paused = not paused
                if paused:
                    freeze_frame = frame.copy()
        if key == ord('q'):
                logging.info(f"[{window_name}] Người dùng nhấn 'q'. Thoát luồng.")
                break

        if paused:
                pause_frame = (freeze_frame.copy()
                            if freeze_frame is not None else frame.copy())
                cv2.putText(pause_frame, "P to Pause - Q to Quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)
                cv2.imshow(window_name, pause_frame)
                continue
        cv2.imshow(window_name, display_frame)
    cap.release()
    cv2.destroyAllWindows()
    logging.info(f"[{window_name}] Đóng kết nối.")
# =================== HÀM MAIN ===================
def make_transparent(image_path, opacity=30):  # 🆕 Độ mờ 30% (có thể chỉnh thấp hơn)
    image = Image.open(image_path).convert("RGBA")
    alpha = image.split()[3]
    alpha = alpha.point(lambda p: int(p * opacity / 100))
    image.putalpha(alpha)
    return ImageTk.PhotoImage(image)

def select_parameters():
       # Chỉ cho phép tiến trình chính chạy giao diện
    if current_process().name != "MainProcess":
        logging.error("Tiến trình con không được phép chạy giao diện!")
        return {}  # Ngăn tiến trình con mở lại giao diện
    # 🆕 Khởi tạo params đúng cách
    params = {"telegram_token": "", "chat_id": "", "rtsp_links": [], "enable_zones": [], "model_path": "", "alert_folder": "", "video_path": ""}

    root = tk.Tk()
    root.title("Cấu hình Camera & Mô hình")
    root.configure(bg="#f0f0f0")  # Màu nền của cửa sổ
    root.geometry("800x600")  # Thiết lập kích thước cửa sổ

    style = ttk.Style()
    style.theme_use("clam")

  # 🆕 Các biến chứa thông số của giao diện
    telegram_token_var = tk.StringVar()
    chat_id_var = tk.StringVar()
    rtsp_var1 = tk.StringVar()
    rtsp_var2 = tk.StringVar()
    rtsp_var3 = tk.StringVar()
    rtsp_var4 = tk.StringVar()
    # 🆕 Khai báo BooleanVar trước khi dùng
    enable_zone_var1 = tk.BooleanVar(value=False)
    enable_zone_var2 = tk.BooleanVar(value=False)
    enable_zone_var3 = tk.BooleanVar(value=False)
    enable_zone_var4 = tk.BooleanVar(value=False)
    model_path_var = tk.StringVar()
    alert_folder_var = tk.StringVar(value=os.path.join(os.getcwd(), "alert_images"))
    video_path_var = tk.StringVar()
    # ==== thêm biến để lưu trạng thái enable zone ====
    enable_zone_var1 = tk.BooleanVar(value=False)
    enable_zone_var2 = tk.BooleanVar(value=False)
    enable_zone_var3 = tk.BooleanVar(value=False)
    enable_zone_var4 = tk.BooleanVar(value=False)
    demo_zone_var = tk.BooleanVar(value=False)
     # 🆕 Hàm chọn file mô hình
    def browse_model():
        fp = filedialog.askopenfilename(title="Chọn file mô hình (best.pt)", filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")])
        if fp:
            model_path_var.set(fp)
    # 🆕 Hàm chọn thư mục lưu ảnh cảnh báo
    def browse_folder():
        folder = filedialog.askdirectory(title="Chọn thư mục lưu ảnh cảnh báo")
        if folder:
            alert_folder_var.set(folder)
    # 🆕 Hàm chọn file video mẫu
    def browse_video():
        file_path = filedialog.askopenfilename(title="Chọn video mẫu", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if file_path:
            video_path_var.set(file_path)
    # 🆕 Đọc ảnh logo và resize nếu cần
    image = Image.open("D:/nhan_dien_nguoi_la/logo.jpg").resize((400, 300))
    bg_image = ImageTk.PhotoImage(image)
# 🆕 Hiển thị logo nền với độ trong suốt
    bg_image = make_transparent("D:/nhan_dien_nguoi_la/logo.jpg", opacity=30)
    bg_label = tk.Label(root, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    bg_label.lower()  # 🆕 Đảm bảo logo không che phần nhập liệu
 # 🆕 Đặt logo làm nền bằng Label
    bg_label = tk.Label(root, image=bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # 🆕 Đặt logo nền đúng vị trí
    bg_label.lower()  # 🆕 Đưa logo xuống dưới để tránh chặn ô nhập liệu
  # 🆕 Giữ ảnh để tránh bị xoá (Đặt sau khi `bg_label` đã khởi tạo)
    bg_label.image = bg_image
     # 🆕 Tạo canvas và đặt ảnh làm background
    canvas = tk.Canvas(root, width=800, height=600)
    canvas.place(x=0, y=0, width=800, height=600)
    canvas.create_image(400, 300, anchor="center", image=bg_image)
    canvas.image = bg_image
     # 🆕 Thêm thông tin tác giả phần mềm vào góc bên phải (bạn có thể điều chỉnh vị trí x, y cho phù hợp)
    tk.Label(root,
         text="Phát triển bởi: C.P Vietnam Corporation \nSwine Veterinary and Biosecurity Department \n(Bộ phận Thú y và Phòng dịch)",
         bg="#f0f0f0",
         fg="black",
         font=("Helvetica", 10, "italic")).place(x=510, y=10)

    tk.Label(root, text="Telegram Token:", bg="#f0f0f0", fg="black").place(x=50, y=50)
    tk.Entry(root, textvariable=telegram_token_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=50)

    tk.Label(root, text="Telegram Chat ID:", bg="#f0f0f0", fg="black").place(x=50, y=100)
    tk.Entry(root, textvariable=chat_id_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=100)

    tk.Label(root, text="RTSP Link 1:", bg="#f0f0f0", fg="black").place(x=50, y=150)
    tk.Entry(root, textvariable=rtsp_var1, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=150)
    tk.Checkbutton(root,
               text="Vẽ vùng cảnh báo",
               variable=enable_zone_var1,
               bg="#f0f0f0")\
    .place(x=600, y=150)
    tk.Label(root, text="RTSP Link 2:", bg="#f0f0f0", fg="black").place(x=50, y=200)
    tk.Entry(root, textvariable=rtsp_var2, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=200)
    tk.Checkbutton(root,
               text="Vẽ vùng cảnh báo",
               variable=enable_zone_var2,
               bg="#f0f0f0")\
    .place(x=600, y=200)

    tk.Label(root, text="RTSP Link 3:", bg="#f0f0f0", fg="black").place(x=50, y=250)
    tk.Entry(root, textvariable=rtsp_var3, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=250)
    tk.Checkbutton(root,
               text="Vẽ vùng cảnh báo",
               variable=enable_zone_var3,
               bg="#f0f0f0")\
    .place(x=600, y=250)
    tk.Label(root, text="RTSP Link 4:", bg="#f0f0f0", fg="black").place(x=50, y=300)
    tk.Entry(root, textvariable=rtsp_var4, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=300)
    tk.Checkbutton(root,
               text="Vẽ vùng cảnh báo",
               variable=enable_zone_var4,
               bg="#f0f0f0")\
    .place(x=600, y=300)

    tk.Label(root, text="Model Path (best.pt):", bg="#f0f0f0", fg="black").place(x=50, y=350)
    tk.Entry(root, textvariable=model_path_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=350)
    tk.Button(root, text="Browse", command=browse_model, bg="#d3d3d3", fg="black", relief="raised", bd=3).place(x=510, y=348)

    tk.Label(root, text="Alert Image Folder:", bg="#f0f0f0", fg="black").place(x=50, y=400)
    tk.Entry(root, textvariable=alert_folder_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=400)
    tk.Button(root, text="Browse", command=browse_folder, bg="#d3d3d3", fg="black", relief="raised", bd=3).place(x=510, y=398)

    tk.Label(root, text="Video Test:", bg="#f0f0f0", fg="black").place(x=50, y=450)
    tk.Entry(root, textvariable=video_path_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=450)
    tk.Button(root, text="Browse", command=browse_video, bg="#d3d3d3", fg="black", relief="raised", bd=3).place(x=510, y=448)
    # 🆕 Checkbox cho Demo Video
    tk.Checkbutton(root,
                   text="Vẽ vùng cảnh báo",
                   variable=demo_zone_var,
                   bg="#f0f0f0")\
      .place(x=600, y=450)
    # Nút Start: Lưu các thông tin và đóng cửa sổ
    def on_start():
        params["telegram_token"] = telegram_token_var.get()
        params["chat_id"] = chat_id_var.get()
        params["rtsp_links"] = [rtsp_var1.get(), rtsp_var2.get(), rtsp_var3.get(), rtsp_var4.get()]  # 🆕 Thêm RTSP Link 4
        params["enable_zones"]  = [
        enable_zone_var1.get(),
        enable_zone_var2.get(),
        enable_zone_var3.get(),
        enable_zone_var4.get()]
        params["model_path"] = model_path_var.get()
        params["alert_folder"] = alert_folder_var.get()
        params["video_path"] = video_path_var.get() if not any(params["rtsp_links"]) else None  # 🆕 Nếu không có RTSP, dùng video test
        params["demo_enable_zone"] = demo_zone_var.get()
        root.destroy()
        # The run_camera function now expects a dictionary of parameters
        # It will be called in the main block using multiprocessing
        # We don't call it directly here.

# Ví dụ sử dụng ttk.Button thay cho tk.Button:
    ttk.Button(root, text="Start", command=on_start, style="TButton").place(x=250, y=500)
    root.mainloop()
    # Kiểm tra nếu thiếu Token hoặc Chat ID thì báo lỗi
    if not params["telegram_token"] or not params["chat_id"]:
        logging.error("Lỗi: Telegram Token hoặc Chat ID chưa nhập!")
        return {}
    logging.info(f"Dữ liệu đã nhập: {params}")
    return params

# =================== HÀM MAIN ===================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")
    # Dành cho việc chạy EXE trên Windows với multiprocessing
    freeze_support()
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    # Chỉ tiến trình chính mới thực hiện các bước dưới đây
    # if current_process().name != "MainProcess": # This check is actually preventing sub-processes from running
    #     exit()                                 # Removed to allow multiprocessing to work as intended
    params = select_parameters()
    if not params or (not any(params.get("rtsp_links")) and not params.get("video_path")):
        logging.error("Lỗi: Không nhận được dữ liệu từ select_parameters() hoặc không có nguồn video/RTSP.")
        exit()
    telegram_token = params.get("telegram_token")
    chat_id = params.get("chat_id")
    rtsp_links = params.get("rtsp_links", [])
    enable_zones = params["enable_zones"]
    model_path = params.get("model_path", "")
    alert_folder = params.get("alert_folder", "")
    video_path = params.get("video_path")  # 🆕 Thêm video mẫu
    if not os.path.exists(model_path):
        logging.error("Thiếu dữ liệu đầu vào!")
        exit()
    logging.info("Chương trình bắt đầu xử lý camera...")
    processes = []
    # 🔹 Nếu có RTSP, chạy từng camera
    if any(rtsp_links) and not video_path: # Prioritize RTSP if provided and no video path explicitly selected
        for idx, rtsp_url in enumerate(rtsp_links):
            if not rtsp_url.strip():
                continue
            window_name_cam = f"Camera {idx+1}"
            enable_zone_for_cam = enable_zones[idx] if idx < len(enable_zones) else False
            process_params = {
                "rtsp_url": rtsp_url,
                "window_name": window_name_cam,
                "model_path": model_path,
                "alert_folder": alert_folder,
                "processing_interval": PROCESSING_INTERVAL,
                "telegram_token": telegram_token,
                "chat_id": chat_id,
                "enable_danger_zone": enable_zone_for_cam,
                "video_path": None # Ensure video_path is None for RTSP streams
            }
            p = Process(name=window_name_cam, target=run_camera, args=(process_params,))
            p.start()
            processes.append(p)
    else:
        # 🔹 Nếu không có RTSP hoặc video path được chọn, chạy video mẫu
        if video_path and os.path.exists(video_path):
            window_name_demo = "Demo Video"
            demo_enable = params.get("demo_enable_zone", False)
            process_params = {
                "rtsp_url": None, # Ensure rtsp_url is None for demo video
                "window_name": window_name_demo,
                "model_path": model_path,
                "alert_folder": alert_folder,
                "processing_interval": PROCESSING_INTERVAL,
                "telegram_token": telegram_token,
                "chat_id": chat_id,
                "enable_danger_zone": demo_enable,
                "video_path": video_path
            }
            p = Process(target=run_camera, args=(process_params,))
            p.start()
            processes.append(p)
        else:
            logging.error("Lỗi: Không có nguồn video/RTSP hợp lệ được cung cấp.")
            exit()

    for p in processes:
        p.join()