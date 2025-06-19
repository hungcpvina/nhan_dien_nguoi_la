import cv2
import time
import os
import sys
import math
from PIL import Image, ImageTk
from PIL import ImageFilter
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import logging
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
PROCESSING_INTERVAL = 3    # Chạy inference mỗi 3 giây để giảm tải
ALERT_INTERVAL = 120        # Cách nhau 60 giây giữa các lần cảnh báo
REPEATED_ALERT_INTERVAL = 300  # 5 phút giữa các lần cảnh báo cùng một đối tượng
window_name = "Camera Feed"
DISPLAY_WIDTH = 400        # Cửa sổ video hiển thị có độ rộng cố định (khoảng 4 inch - ví dụ 400 pixel)
DISPLAY_HEIGHT = 300       # Chiều cao hiển thị (bạn có thể điều chỉnh)
detected_objects = {}  # Lưu thông tin về các đối tượng đã phát hiện
last_alert_time = {}
has_lost_connection = False  # 🆕 Biến trạng thái để ngăn spam cảnh báo
motion_check_interval = 10
frame_idx = 0
full_threshold = 5000

DURATION_THRESH = 1      # giây
MOVE_THRESH     = 30       # pixel: vùng di chuyển nhỏ
MATCH_THRESH    = 50       # pixel: để match box với tracker
MAX_IDLE        = 2      # giây: xóa tracker không thấy update


# =================== HÀM GỬI CẢNH BÁO QUA TELEGRAM ===================
def send_alert_to_telegram(image, object_id):
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
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        data = {'chat_id': CHAT_ID, 'caption': '⚠️ Cảnh báo ! Phát hiện người lạ!'}
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
def make_transparent(image_path, opacity=30):  # 🆕 Độ mờ 30% (có thể chỉnh thấp hơn)
    image = Image.open(image_path).convert("RGBA")  
    alpha = image.split()[3]  
    alpha = alpha.point(lambda p: int(p * opacity / 100))  
    image.putalpha(alpha)  
    return ImageTk.PhotoImage(image)

def select_parameters():
    # 🆕 Khởi tạo params đúng cách
    params = {"telegram_token": "", "chat_id": "", "rtsp_links": [], "model_path": "", "alert_folder": "", "video_path": ""}
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

    tk.Label(root, text="RTSP Link 2:", bg="#f0f0f0", fg="black").place(x=50, y=200)
    tk.Entry(root, textvariable=rtsp_var2, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=200)

    tk.Label(root, text="RTSP Link 3:", bg="#f0f0f0", fg="black").place(x=50, y=250)
    tk.Entry(root, textvariable=rtsp_var3, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=250)

    tk.Label(root, text="RTSP Link 4:", bg="#f0f0f0", fg="black").place(x=50, y=300)
    tk.Entry(root, textvariable=rtsp_var4, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=300)

    tk.Label(root, text="Model Path (best.pt):", bg="#f0f0f0", fg="black").place(x=50, y=350)
    tk.Entry(root, textvariable=model_path_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=350)
    tk.Button(root, text="Browse", command=browse_model, bg="#d3d3d3", fg="black", relief="raised", bd=3).place(x=510, y=348)

    tk.Label(root, text="Alert Image Folder:", bg="#f0f0f0", fg="black").place(x=50, y=400)
    tk.Entry(root, textvariable=alert_folder_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=400)
    tk.Button(root, text="Browse", command=browse_folder, bg="#d3d3d3", fg="black", relief="raised", bd=3).place(x=510, y=398)

    tk.Label(root, text="Video Test:", bg="#f0f0f0", fg="black").place(x=50, y=450)
    tk.Entry(root, textvariable=video_path_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=450)
    tk.Button(root, text="Browse", command=browse_video, bg="#d3d3d3", fg="black", relief="raised", bd=3).place(x=510, y=448)

    # 🆕 Nút Start
    def on_start():
        params["telegram_token"] = telegram_token_var.get()
        params["chat_id"] = chat_id_var.get()
        params["rtsp_links"] = [rtsp_var1.get(), rtsp_var2.get(), rtsp_var3.get(), rtsp_var4.get()]
        params["model_path"] = model_path_var.get()
        params["alert_folder"] = alert_folder_var.get()
        params["video_path"] = video_path_var.get() if not any(params["rtsp_links"]) else None
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
def send_alert_to_telegram(image, telegram_token, chat_id):
    if not telegram_token or not chat_id:
        logging.error("Lỗi: Telegram Token hoặc Chat ID chưa được thiết lập!")
        return False
    try:
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
def run_camera(rtsp_url, window_name, model_path,
               alert_folder, processing_interval,
               telegram_token, chat_id, video_path=None):
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
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
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
     # Thiết lập cửa sổ hiển thị
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"[{window_name}] FPS luồng: {cap_fps}")
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
    motion_check_interval = 5
    frame_idx = 0
    consecutive_detections = 0
    detection_required = 2
    min_confidence   = 0.6
    full_threshold = 5000
    
    def centroid(x1, y1, x2, y2):
        return ((x1+x2)//2, (y1+y2)//2)
    def dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    while True:
        time.sleep(0.1)
        cap.grab()
        ret, frame = cap.retrieve()

        # ——— Xử lý mất kết nối ———
        if not ret:
            if rtsp_url:
                if not has_lost_connection:
                    logging.warning(f"[{window_name}] Mất kết nối RTSP, đang thử lại...")
                    send_text_to_telegram(
                        f"⚠️ Camera {window_name} mất kết nối RTSP. Đang tự động khôi phục...",
                        telegram_token, chat_id)
                    has_lost_connection = True
                cap.release()
                time.sleep(3)
                retry = 0
                while retry < MAX_RETRY:
                    cap = cv2.VideoCapture(rtsp_url)
                    if cap.isOpened():
                        logging.info(f"[{window_name}] Đã kết nối lại RTSP.")
                        send_text_to_telegram(
                            f"✅ Camera {window_name} đã kết nối lại thành công.",
                            telegram_token, chat_id)
                        has_lost_connection = False
                        break
                    cap.release()
                    retry += 1
                    time.sleep(3)
                if not cap or not cap.isOpened():
                    logging.error(f"[{window_name}] Không thể kết nối lại sau {MAX_RETRY} lần.")
                    break
                continue
            else:
                logging.error(f"[{window_name}] Video mẫu đã kết thúc!")
                break
        # 🔹 Chuẩn bị frame & frame‐skipping
        current_time  = time.time()
        display_frame = frame.copy()
        fgmask = fgbg.apply(frame)  
        motion_pixels = cv2.countNonZero(fgmask)
        # nếu chuyển động (số pixel đổi) chưa vượt full_threshold thì skip
        if motion_pixels < full_threshold:
            logging.debug(
                f"[{window_name}] Motion pixels={motion_pixels} < full_th={full_threshold}, skip")
            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue 
        frame_idx    += 1
        if frame_idx % motion_check_interval != 0:
            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        # 🔹 Chạy inference & tracking khi tới lượt
        if current_time - last_inference >= processing_interval:
            last_inference = current_time
            detections_this_frame = False
            try:
                results = model(display_frame)[0]
                logging.info(f"[{window_name}] Model trả về {len(results.boxes)} boxes")
                new_trackers = []
                # duyệt qua các box và chỉ quan tâm lớp trong classes_to_alert
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    name   = (model.names[cls_id] if 
                              not isinstance(model.names, dict) 
                              else model.names.get(cls_id))
                    logging.info(f"[{window_name}] Phát hiện {name} với confidence={conf:.2f}")
                    if name not in classes_to_alert or conf < min_confidence:
                        continue
                    # tọa độ và centroid
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    c = centroid(x1, y1, x2, y2)
                    # match với tracker cũ
                    matched = None
                    for tr in trackers:
                        if dist(tr['positions'][-1], c) < MATCH_THRESH:
                            matched = tr
                            break
                    # cập nhật hoặc tạo mới tracker
                    if matched:
                        matched['positions'].append(c)
                        matched['last'] = current_time
                        xs = [p[0] for p in matched['positions']]
                        ys = [p[1] for p in matched['positions']]
                        # nếu đứng yên đủ lâu & chỉ lắc nhỏ → static
                        if (current_time - matched['start'] >= DURATION_THRESH and
                            max(xs)-min(xs) <= MOVE_THRESH and
                            max(ys)-min(ys) <= MOVE_THRESH):
                            matched['static'] = True
                    else:
                        matched = {
                            'id': next_id,
                            'positions': [c],
                            'start': current_time,
                            'last': current_time,
                            'static': False}
                        next_id += 1
                    new_trackers.append(matched)
                    # nếu tracker chưa static → vẽ box và đánh dấu detection
                    if not matched['static']:
                        detections_this_frame = True
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2),
                                      (0, 0, 255), 2)
                        cv2.putText(display_frame,
                                    f"{name} {conf:.2f}",
                                    (x1, max(y1-10,10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                # dọn tracker idle
                trackers = [
                    tr for tr in new_trackers
                    if (current_time - tr['last']) <= MAX_IDLE]
                # cập nhật counter liên tiếp
                if detections_this_frame:
                    consecutive_detections += 1
                    logging.info(f"[{window_name}] consecutive detections = {consecutive_detections}")
                else:
                    consecutive_detections = 0
                logging.debug(
                    f"[{window_name}] check alert: cons={consecutive_detections}, "
                    f"elapsed={(current_time-last_alert_time):.1f}s, "
                    f"threshold={ALERT_INTERVAL}s")
                # nếu đủ liên tiếp & qua ALERT_INTERVAL → lưu ảnh + gửi Telegram
                if (consecutive_detections >= detection_required and
                    (current_time - last_alert_time) >= ALERT_INTERVAL):
                    logging.info(f"[{window_name}] → Điều kiện alert thỏa, gửi Telegram")
                    filename = os.path.join(
                        alert_folder, f"alert_{int(current_time)}.jpg")
                    cv2.imwrite(filename, display_frame)
                    logging.info(f"[{window_name}] Lưu ảnh cảnh báo: {filename}")
                    ok = send_alert_to_telegram(display_frame, telegram_token, chat_id)
                    logging.info(f"[{window_name}] send_alert_to_telegram returned: {ok}")
                    last_alert_time = current_time
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
    cv2.destroyWindow(window_name)
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
    params = {"telegram_token": "", "chat_id": "", "rtsp_links": [], "model_path": "", "alert_folder": "", "video_path": ""}

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
    model_path_var = tk.StringVar()
    alert_folder_var = tk.StringVar(value=os.path.join(os.getcwd(), "alert_images"))
    video_path_var = tk.StringVar()

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

    tk.Label(root, text="RTSP Link 2:", bg="#f0f0f0", fg="black").place(x=50, y=200)
    tk.Entry(root, textvariable=rtsp_var2, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=200)

    tk.Label(root, text="RTSP Link 3:", bg="#f0f0f0", fg="black").place(x=50, y=250)
    tk.Entry(root, textvariable=rtsp_var3, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=250)

    tk.Label(root, text="RTSP Link 4:", bg="#f0f0f0", fg="black").place(x=50, y=300)
    tk.Entry(root, textvariable=rtsp_var4, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=300)

    tk.Label(root, text="Model Path (best.pt):", bg="#f0f0f0", fg="black").place(x=50, y=350)
    tk.Entry(root, textvariable=model_path_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=350)
    tk.Button(root, text="Browse", command=browse_model, bg="#d3d3d3", fg="black", relief="raised", bd=3).place(x=510, y=348)

    tk.Label(root, text="Alert Image Folder:", bg="#f0f0f0", fg="black").place(x=50, y=400)
    tk.Entry(root, textvariable=alert_folder_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=400)
    tk.Button(root, text="Browse", command=browse_folder, bg="#d3d3d3", fg="black", relief="raised", bd=3).place(x=510, y=398)

    tk.Label(root, text="Video Test:", bg="#f0f0f0", fg="black").place(x=50, y=450)
    tk.Entry(root, textvariable=video_path_var, width=50, bg="#e0e0e0", fg="black", relief="ridge", bd=3).place(x=200, y=450)
    tk.Button(root, text="Browse", command=browse_video, bg="#d3d3d3", fg="black", relief="raised", bd=3).place(x=510, y=448)

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
        for idx, rtsp_url in enumerate(rtsp_links, 1):
            window_name = f"Camera {idx+1}"
            p = Process(name=window_name, target=run_camera, args=(rtsp_url, window_name, model_path, alert_folder, PROCESSING_INTERVAL, telegram_token, chat_id))
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