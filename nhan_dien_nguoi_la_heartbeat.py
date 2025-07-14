import sys
import cv2
import time
import os
import math
import threading
import socket
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

# Hàm tìm path chính xác cho cả lúc chạy .py lẫn .exe
def resource_path(rel_path):
    """
    Trả về đường dẫn tuyệt đối tới tài nguyên:
    - Khi chạy exe: _MEIPASS + rel_path
    - Khi chạy script: thư mục chứa file .py + rel_path
    """
    if hasattr(sys, '_MEIPASS'):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    full = os.path.join(base, rel_path)

    # Trong dev-mode: nếu không tìm thấy ở rel_path, thử fallback về cùng cấp file .py
    if not os.path.exists(full) and not hasattr(sys, '_MEIPASS'):
        full_fallback = os.path.join(base, os.path.basename(rel_path))
        if os.path.exists(full_fallback):
            full = full_fallback
    return full
# DEBUG: in ra đường dẫn đầy đủ và trạng thái tồn tại của file
logo_rel   = 'nhan_dien_nguoi_la/logo.jpg'
logo_abspath = resource_path(logo_rel)
print(f"[DEBUG]    rel         = {logo_rel}")
print(f"[DEBUG]    _MEIPASS    = {getattr(sys, '_MEIPASS', None)}")
print(f"[DEBUG]    script_dir  = {os.path.dirname(os.path.abspath(__file__))}")
print(f"[DEBUG]    full path   = {logo_abspath}")
print(f"[DEBUG]    exists      = {os.path.exists(logo_abspath)}")

# =================== CẤU HÌNH CHUNG ===================
# Các lớp đối tượng cần cảnh báo
classes_to_alert = ["nguoi_la", "nguoi_do", "con_cho", "con_meo", "con_chuot"]
# Các hằng số xử lý
RETRY_DELAY = 10 
MAX_RETRY = 500
RETRY_DELAY = 10
RECONNECT_INTERVAL = 300
RETRY_DELAY_BETWEEN = 5
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
MOVE_THRESH     = 100       # pixel: vùng di chuyển nhỏ
MATCH_THRESH    = 500       # pixel: để match box với tracker
MAX_IDLE        = 1      # giây: xóa tracker không thấy update


# =================== HÀM TIỆN ÍCH KIỂM TRA INTERNET ===================
def is_internet_available(host="api.telegram.org", port=443, timeout=5):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)\
              .connect((host, port))
        return True
    except Exception:
        return False

# Hàm helper lưu ảnh tạm
def _cache_alert_image(image, alert_folder, prefix, object_id=None):
    os.makedirs(alert_folder, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    oid = object_id if object_id is not None else "unknown"
    filename = f"{prefix}_{oid}_{ts}.jpg"
    path = os.path.join(alert_folder, filename)
    cv2.imwrite(path, image)
    logging.info(f"Offline cache: lưu ảnh cảnh báo tại {path}")



# =================== HÀM GỬI CẢNH BÁO ‘STRANGER’ ===================
def send_alert_to_telegram(image, telegram_token, chat_id,
                           alert_folder, object_id=None):
    """
    Gửi ảnh stranger alert qua Telegram.
    Nếu mất Internet hoặc gửi lỗi, tự lưu ảnh vào alert_folder.
    """
    if not telegram_token or not chat_id:
        logging.error("Lỗi: Telegram Token hoặc Chat ID chưa được thiết lập!")
        return False

    # Kiểm tra khoảng cách 2 lần cảnh báo cùng object
    current_time = time.time()
    alert_interval = 300  # 5 phút
    if object_id in last_alert_time:
        if current_time - last_alert_time[object_id] < alert_interval:
            logging.info(f"Đối tượng {object_id} đã cảnh báo gần đây, "
                         f"chờ thêm {alert_interval}s.")
            return False

    # Nếu không có mạng, lưu tạm và trả về False
    if not is_internet_available():
        _cache_alert_image(image, alert_folder, prefix="stranger", object_id=object_id)
        return False

    # Thử gửi
    _, img_encoded = cv2.imencode('.jpg', image)
    files = {'photo': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
    url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
    data = {'chat_id': chat_id, 'caption': '⚠️ Cảnh báo! Phát hiện người lạ!'}

    try:
        resp = requests.post(url, files=files, data=data, timeout=10)
        if resp.ok:
            logging.info("Đã gửi stranger alert qua Telegram thành công.")
            last_alert_time[object_id] = current_time
            return True
        else:
            logging.warning(f"Telegram API lỗi: {resp.text}")
            _cache_alert_image(image, alert_folder, prefix="stranger", object_id=object_id)
            return False

    except requests.RequestException as e:
        logging.error(f"Lỗi khi gửi stranger alert (RequestException): {e}")
        _cache_alert_image(image, alert_folder, prefix="stranger", object_id=object_id)
        return False
    except Exception as e:
        logging.error(f"Unexpected error sending stranger alert: {e}")
        _cache_alert_image(image, alert_folder, prefix="stranger", object_id=object_id)
        return False

# =================== HÀM GỬI CẢNH BÁO ‘ZONE’ ===================
def send_zone_alert_to_telegram(image, telegram_token, chat_id,
                                alert_folder, object_id=None):
    """
    Gửi ảnh zone violation qua Telegram.
    Nếu mất Internet hoặc gửi lỗi, tự lưu ảnh vào alert_folder.
    """
    if not telegram_token or not chat_id:
        logging.error("Thiếu Token hoặc Chat ID khi gửi zone alert!")
        return False

    # Nếu không có mạng, lưu tạm và trả về False
    if not is_internet_available():
        _cache_alert_image(image, alert_folder, prefix="zone", object_id=object_id)
        return False

    _, img_encoded = cv2.imencode('.jpg', image)
    files = {'photo': ('zone_alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
    url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
    data = {'chat_id': chat_id, 'caption': '🚧 CẢNH BÁO: Vùng nguy hiểm!'}

    try:
        resp = requests.post(url, files=files, data=data, timeout=10)
        if resp.ok:
            logging.info("Đã gửi zone alert qua Telegram thành công.")
            return True
        else:
            logging.warning(f"Telegram API lỗi (zone): {resp.text}")
            _cache_alert_image(image, alert_folder, prefix="zone", object_id=object_id)
            return False

    except requests.RequestException as e:
        logging.error(f"Lỗi khi gửi zone alert (RequestException): {e}")
        _cache_alert_image(image, alert_folder, prefix="zone", object_id=object_id)
        return False
    except Exception as e:
        logging.error(f"Unexpected error sending zone alert: {e}")
        _cache_alert_image(image, alert_folder, prefix="zone", object_id=object_id)
        return False

# =================== HÀM ĐẨY ẢNH TRỰC TIẾP KHÔNG GIỚI HẠN ===================
def _push_photo_to_telegram(image, telegram_token, chat_id, caption):
    try:
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'photo': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
        url   = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
        data  = {'chat_id': chat_id, 'caption': caption}
        resp  = requests.post(url, files=files, data=data, timeout=10)
        if resp.ok:
            logging.info("Resend thành công qua _push_photo_to_telegram.")
        else:
            logging.warning(f"Lỗi khi gửi ảnh resend: {resp.text}")
        return resp.ok
    except Exception as e:
        logging.error(f"Lỗi trong _push_photo_to_telegram: {e}")
        return False

# =================== HÀM GIAO DIỆN CHỌN THAM SỐ (Tkinter GUI) ===================
def make_transparent(img_or_path, opacity=30):  # 🆕 Độ mờ 30% (có thể chỉnh thấp hơn)
    # 1. Mở ảnh nếu được truyền path
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path).convert("RGBA")
    else:
        img = img_or_path.convert("RGBA")
    # 2. Điều chỉnh kênh alpha
    r, g, b, a = img.split()
    a = a.point(lambda p: int(p * opacity / 100))
    img.putalpha(a)
    return ImageTk.PhotoImage(img)

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
    bg_image = make_transparent(
    resource_path('nhan_dien_nguoi_la/logo.jpg'),
    opacity=30)
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
# HÀM GỬI CẢNH BÁO ‘STRANGER’
def send_alert_to_telegram(image, telegram_token, chat_id,
                           alert_folder, object_id=None):
    """
    Gửi ảnh stranger alert qua Telegram.
    Nếu mất Internet hoặc gửi lỗi, tự lưu ảnh vào alert_folder.
    """
    if not telegram_token or not chat_id:
        logging.error("Lỗi: Telegram Token hoặc Chat ID chưa được thiết lập!")
        return False

    # Kiểm tra khoảng cách 2 lần cảnh báo cùng object
    current_time = time.time()
    alert_interval = 300  # 5 phút
    if object_id in last_alert_time:
        if current_time - last_alert_time[object_id] < alert_interval:
            logging.info(f"Đối tượng {object_id} đã cảnh báo gần đây, "
                         f"chờ thêm {alert_interval}s.")
            return False

    # Nếu không có mạng, lưu tạm và trả về False
    if not is_internet_available():
        _cache_alert_image(image, alert_folder, prefix="stranger", object_id=object_id)
        return False

    # Thử gửi
    _, img_encoded = cv2.imencode('.jpg', image)
    files = {'photo': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
    url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
    data = {'chat_id': chat_id, 'caption': '⚠️ Cảnh báo! Phát hiện người lạ!'}

    try:
        resp = requests.post(url, files=files, data=data, timeout=10)
        if resp.ok:
            logging.info("Đã gửi stranger alert qua Telegram thành công.")
            last_alert_time[object_id] = current_time
            return True
        else:
            logging.warning(f"Telegram API lỗi: {resp.text}")
            _cache_alert_image(image, alert_folder, prefix="stranger", object_id=object_id)
            return False

    except requests.RequestException as e:
        logging.error(f"Lỗi khi gửi stranger alert (RequestException): {e}")
        _cache_alert_image(image, alert_folder, prefix="stranger", object_id=object_id)
        return False
    except Exception as e:
        logging.error(f"Unexpected error sending stranger alert: {e}")
        _cache_alert_image(image, alert_folder, prefix="stranger", object_id=object_id)
        return False
# =================== HÀM GỬI CẢNH BÁO ‘ZONE’ ===================
def send_zone_alert_to_telegram(image, telegram_token, chat_id,
                                alert_folder, object_id=None):
    """
    Gửi ảnh zone violation qua Telegram.
    Nếu mất Internet hoặc gửi lỗi, tự lưu ảnh vào alert_folder.
    """
    if not telegram_token or not chat_id:
        logging.error("Thiếu Token hoặc Chat ID khi gửi zone alert!")
        return False

    # Nếu không có mạng, lưu tạm và trả về False
    if not is_internet_available():
        _cache_alert_image(image, alert_folder, prefix="zone", object_id=object_id)
        return False

    _, img_encoded = cv2.imencode('.jpg', image)
    files = {'photo': ('zone_alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
    url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
    data = {'chat_id': chat_id, 'caption': '🚧 CẢNH BÁO: Vùng nguy hiểm!'}

    try:
        resp = requests.post(url, files=files, data=data, timeout=10)
        if resp.ok:
            logging.info("Đã gửi zone alert qua Telegram thành công.")
            return True
        else:
            logging.warning(f"Telegram API lỗi (zone): {resp.text}")
            _cache_alert_image(image, alert_folder, prefix="zone", object_id=object_id)
            return False

    except requests.RequestException as e:
        logging.error(f"Lỗi khi gửi zone alert (RequestException): {e}")
        _cache_alert_image(image, alert_folder, prefix="zone", object_id=object_id)
        return False
    except Exception as e:
        logging.error(f"Unexpected error sending zone alert: {e}")
        _cache_alert_image(image, alert_folder, prefix="zone", object_id=object_id)
        return False
# ================ Hàm gửi tin nhắn văn bản ===================
def send_text_to_telegram(message, telegram_token, chat_id):
    if not telegram_token or not chat_id:
        logging.error("Lỗi: Telegram Token hoặc Chat ID chưa được thiết lập!")
        return False
    # Nếu không có mạng, bỏ qua ngay
    if not is_internet_available():
        logging.warning("Offline: không thể gửi text alert, bỏ qua.")
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
# =================== HÀM ĐẨY ẢNH CŨ LÚC MẤT MẠNG ===================
def _push_photo_to_telegram(image, telegram_token, chat_id, caption):
    try:
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'photo': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
        url   = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
        data  = {'chat_id': chat_id, 'caption': caption}
        resp  = requests.post(url, files=files, data=data, timeout=10)
        if resp.ok:
            logging.info("Resend thành công qua _push_photo_to_telegram.")
        else:
            logging.warning(f"Lỗi khi gửi ảnh resend: {resp.text}")
        return resp.ok
    except Exception as e:
        logging.error(f"Lỗi trong _push_photo_to_telegram: {e}")
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
# ===================== Hàm vẽ Dangerzone=====================
def is_inside_zone(c, polygon):
    """Trả về True nếu point c=(x,y) nằm trong polygon."""
    # pointPolygonTest >=0 nghĩa nằm trong hoặc trên biên
    return cv2.pointPolygonTest(polygon, c, False) >= 0
# =================== Hàm gửi lại ảnh lưu tạm thời (khi mất mạng) ===================
def resend_cached_alerts(telegram_token, chat_id, alert_folder, interval=60):
    """
    Thread chạy nền: cứ mỗi `interval` giây, nếu có Internet,
    resend tất cả ảnh .jpg trong alert_folder và xóa file khi gửi thành công.
    """
    def _worker():
        while True:
            time.sleep(interval)
            # Chỉ resend khi có mạng
            if not is_internet_available():
                continue

            for fname in sorted(os.listdir(alert_folder)):
                if not fname.lower().endswith(".jpg"):
                    continue

                path = os.path.join(alert_folder, fname)
                img = cv2.imread(path)
                if img is None:
                    continue

                # Quy ước: zone_… → gọi send_zone, khác → send_alert
                if fname.startswith("zone_"):
                    ok = _push_photo_to_telegram(img, telegram_token, chat_id,
                                 caption="🚧 CẢNH BÁO: vùng nguy hiểm!")
                else:
                    ok = _push_photo_to_telegram(img, telegram_token, chat_id,
                                 caption="⚠️ Cảnh báo! Phát hiện người lạ!")
                if ok:
                    try:
                        os.remove(path)
                        logging.info(f"Resend thành công, xóa file cache: {fname}")
                    except Exception as e:
                        logging.warning(f"Không xóa được {fname}: {e}")
                else:
                    logging.warning(f"Resend thất bại, giữ lại: {fname}")

    t = threading.Thread(target=_worker, daemon=True, name="ResendCacheThread")
    t.start()
# =================== HÀM HEARTBEAT (GỬI TEXT MỖI 24h) ===================
def start_heartbeat(telegram_token, chat_id, interval=24*3600):
    """
    Thread nền: mỗi `interval` giây gửi tin báo alive.
    """
    def _hb_worker():
        while True:
            send_text_to_telegram(
                "🟢 Hệ thống giám sát vẫn hoạt động.",
                telegram_token, chat_id)
            time.sleep(interval)
    t = threading.Thread(
        target=_hb_worker,
        daemon=True,
        name="HeartbeatThread")
    t.start()
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
    video_ended = False
    last_frame  = None
    # Khởi động thread resend ảnh cache khi có Internet
    os.makedirs(alert_folder, exist_ok=True)
    resend_cached_alerts(telegram_token, chat_id, alert_folder, interval=60)
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
        while True:
        # kiểm tra RTSP server có phản hồi hay không
            if not check_camera_connection(rtsp_url, window_name, telegram_token, chat_id):
                logging.warning(f"[{window_name}] Chưa kết nối được RTSP, chờ {RETRY_DELAY}s rồi thử lại")
                time.sleep(RETRY_DELAY)
                continue
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_FPS, 25)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                logging.info(f"[{window_name}] Kết nối RTSP thành công!")
                break
        # nếu chưa mở được, release và chờ retry
            cap.release()
            logging.warning(f"[{window_name}] Mở RTSP thất bại, chờ {RETRY_DELAY}s rồi thử lại")
            time.sleep(RETRY_DELAY)
    else:
        logging.info(f"[{window_name}] Chạy với video mẫu: {video_path}")
        cap = cv2.VideoCapture(video_path)
    # Biến trạng thái kết nối
    paused = False
    freeze_frame = None  # Biến lưu ảnh dừng (freeze) khi tạm dừng
    has_lost_connection = False
    last_inference = 0
    last_alert_time = 0
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
    print(f"[DEBUG] register mouse callback on window: '{window_name}'")
    # 🆕 Khởi tạo biến lưu trạng thái vẽ vùng cảnh báo
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
    motion_check_interval = 3
    frame_idx = 0
    consecutive_detections = 0
    detection_required = 1
    min_confidence   = 0.65
    full_threshold = 5000
    entered_zone_ids = set()
    def centroid(x1, y1, x2, y2):
        return ((x1+x2)//2, (y1+y2)//2)
    def dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    print("[DEBUG] entering main loop, waiting for mouse events…")
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
                        telegram_token, chat_id)
                    has_lost_connection = True
                cap.release()
                time.sleep(1)
                # — Bắt đầu vòng lặp vô hạn cho reconnect —
                while True:
                    cap = cv2.VideoCapture(rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if cap.isOpened():
                        logging.info(f"[{window_name}] Đã kết nối lại RTSP.")
                        send_text_to_telegram(
                            f"✅ Camera {window_name} đã kết nối lại thành công.",
                            telegram_token, chat_id)
                        has_lost_connection = False
                        break    # thoát vòng lặp reconnect, quay về xử lý bình thường
                    logging.warning(f"[{window_name}] Reconnect thất bại, chờ {RETRY_DELAY}s rồi thử lại")
                    time.sleep(RETRY_DELAY)
                continue
            # DEMO MODE: video mẫu đã kết thúc
            if not video_ended:
                logging.info(f"[{window_name}] Demo video đã kết thúc. Chờ đóng cửa sổ hoặc kết nối lại Internet.")
                # Giữ lại khung cuối cùng để hiển thị
                last_frame = last_frame if last_frame is not None else np.zeros_like(display_frame)
                video_ended = True
            # Hiển thị khung cuối và thông báo
            disp = last_frame.copy()
            cv2.putText(disp,
                        "Demo video ended. Press Q to quit.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)
            cv2.imshow(window_name, disp)
            # Chờ key, vẫn cho thread resend chạy ngầm
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q'):
                logging.info(f"[{window_name}] Người dùng nhấn 'q'. Thoát.")
                break
            else:
                # không đẩy khung mới, chỉ lặp lại để chờ Q hoặc resend
                continue
        # Nếu ret=True, cập nhật last_frame
        last_frame = frame.copy()
        video_ended = False           
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
                    isClosed=True,
                    color=(0,255,255),
                    thickness=2)
        # … tiếp code xử lý inference, alert, hiển thị cuối cùng …
        cv2.imshow(window_name, display_frame)
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
                            # 0) Chuẩn bị các thông tin cơ bản
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            conf   = float(box.conf[0])
                            name   = model.names.get(cls_id, str(cls_id))
                            c      = ((x1+x2)//2, (y1+y2)//2)

                            # 1) Match tracker để lấy obj_id
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
                                    'last'     : current_time}
                                next_id += 1
                            obj_id = matched['id']
                            new_trackers.append(matched)

                            # 2) Tính net-displacement & static-filter
                            start_pt   = matched['positions'][0]
                            end_pt     = matched['positions'][-1]
                            disp       = math.hypot(end_pt[0]-start_pt[0], end_pt[1]-start_pt[1])
                            track_time = current_time - matched['start']
                            is_static  = (track_time >= DURATION_THRESH and disp < MOVE_THRESH)

                            # 3) Zone-alert (bỏ qua nếu static)
                            in_zone = enable_danger_zone and zone_data["polygon"] is not None \
                                    and is_inside_zone(c, zone_data["polygon"])
                            if in_zone and conf >= 0.3 and not is_static:
                                last_z = zone_last_alert.get(obj_id, 0)
                                if current_time - last_z >= ZONE_REPEAT_INTERVAL:
                                    zone_last_alert[obj_id] = current_time
                                    cv2.rectangle(display_frame, (x1,y1),(x2,y2),(0,0,255),2)
                                    cv2.putText(display_frame, "DANGER ZONE",
                                                (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
                                    send_zone_alert_to_telegram(
                                        display_frame.copy(), telegram_token, chat_id,
                                        alert_folder, object_id=obj_id
                                    )
                                else:
                                    cv2.rectangle(display_frame, (x1,y1),(x2,y2),(0,0,255),2)
                                    cv2.putText(display_frame, "DANGER ZONE",
                                                (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
                                continue

                            # 4) Stranger-alert ngoài zone (bỏ qua nếu static)
                            if not in_zone and name in classes_to_alert \
                            and conf >= min_confidence and not is_static:
                                last_s = stranger_last_alert.get(obj_id, 0)
                                if current_time - last_s >= STRANGER_REPEAT_INTERVAL:
                                    stranger_last_alert[obj_id] = current_time
                                    cv2.rectangle(display_frame,(x1,y1),(x2,y2),(0,0,255),2)
                                    cv2.putText(display_frame,f"{name} {conf:.2f}",
                                                (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                                    send_alert_to_telegram(
                                        display_frame.copy(), telegram_token, chat_id,
                                        alert_folder, object_id=obj_id
                                    )
                                else:
                                    cv2.rectangle(display_frame,(x1,y1),(x2,y2),(0,0,255),2)
                                    cv2.putText(display_frame,f"{name} {conf:.2f}",
                                                (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                                continue

                            # 5) Cuối cùng: vẽ cho đối tượng tĩnh hoặc các lớp không alert
                            if name in classes_to_alert:
                                # chỉ vẽ nếu chưa đánh dấu static
                                if not is_static:
                                    cv2.rectangle(display_frame,(x1,y1),(x2,y2),(0,0,255),2)
                                    cv2.putText(display_frame,
                                                f"{name} {conf:.2f}",
                                                (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                            else:
                                # các lớp khác hiển thị khung nhạt
                                cv2.rectangle(display_frame,(x1,y1),(x2,y2),(200,200,200),1)
                                cv2.putText(display_frame,
                                            f"{name} {conf:.2f}",
                                            (x1, y1-5),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)			
                        # dọn tracker idle & cập nhật
                        trackers = [
                            tr for tr in new_trackers
                            if (current_time - tr['last']) <= MAX_IDLE]
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
def make_transparent(img_or_path, opacity=30):
    """
    img_or_path: PIL.Image hoặc đường dẫn file
    opacity: % độ mờ  (0-100)
    Trả về ImageTk.PhotoImage ở chế độ RGBA với alpha đã điều chỉnh.
    """
    # 1. Mở ảnh nếu được truyền path
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path).convert("RGBA")
    else:
        img = img_or_path.convert("RGBA")
    # 2. Điều chỉnh kênh alpha
    r, g, b, a = img.split()
    a = a.point(lambda p: int(p * opacity / 100))
    img.putalpha(a)
    return ImageTk.PhotoImage(img)
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
    image = Image.open(resource_path('nhan_dien_nguoi_la/logo.jpg')).resize((400, 300))
    bg_image = ImageTk.PhotoImage(image)
    # 🆕 Hiển thị logo nền với độ trong suốt
    bg_image = make_transparent(
    resource_path('nhan_dien_nguoi_la/logo.jpg'),
    opacity=30)
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
    if telegram_token and chat_id:
        start_heartbeat(telegram_token, chat_id)
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
                "video_path": None}
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
                "video_path": video_path}
            p = Process(target=run_camera, args=(process_params,))
            p.start()
            processes.append(p)
        else:
            logging.error("Lỗi: Không có nguồn video/RTSP hợp lệ được cung cấp.")
            exit()
    for p in processes:
        p.join()