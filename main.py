# K230端：控制与 AI 同步主线程 + WiFi 独立推流后台线程，解决掉帧与断连闪退问题
import os, sys, gc, socket, ustruct, time, network, _thread
import uctypes
from machine import Pin, UART, FPIOA, TOUCH

sys.path.append('./sdcard/')

from media.sensor import *
from media.display import *
from media.media import *
from media.vencoder import *
from libs.YOLO import YOLO11
import image

# ================= 状态常量 =================
STATE_STANDBY = 0
STATE_MODE1   = 1

# ================= 配置 =================
WIFI_SSID   = "Keng230_AP"
WIFI_PWD    = "51850107"
SERVER_PORT = 6001
PACKET_MAGIC = 0x55AA55AA

STREAM_W, STREAM_H = 320, 240
AI_W,     AI_H     = 320, 240
LCD_W, LCD_H = 800, 480

KMODEL_PATH = "/sdcard/myball_v5n.kmodel"
LABELS      = ['steel']

IDR_INTERVAL = 60
DET_SEND_MS  = 66
MAX_DETS     = 20
OUT_BUFS     = 8

ROI_X = 1; ROI_Y = 196; ROI_W = 800; ROI_H = 51  # 下底 y=247
AI_ROI_X1 = int(ROI_X * AI_W / LCD_W)
AI_ROI_Y1 = int(ROI_Y * AI_H / LCD_H)
AI_ROI_X2 = int((ROI_X + ROI_W) * AI_W / LCD_W)
AI_ROI_Y2 = int((ROI_Y + ROI_H) * AI_H / LCD_H)

UART_TX_PIN = 11; UART_RX_PIN = 12
UART_BAUD   = 115200
UART_SEND_MS = 10

ORIGIN_X_LCD  = 371
PIXELS_PER_CM = 25.0
POS_5CM_X  = int(ORIGIN_X_LCD + 5.0 * PIXELS_PER_CM)
POS_N5CM_X = int(ORIGIN_X_LCD - 5.0 * PIXELS_PER_CM)

def align16(v): return (v // 16) * 16
if STREAM_W != align16(STREAM_W) or STREAM_H != align16(STREAM_H):
    STREAM_W = align16(STREAM_W); STREAM_H = align16(STREAM_H)

SCALE_X = STREAM_W / AI_W
SCALE_Y = STREAM_H / AI_H

g_current_state = STATE_STANDBY
g_conf_thresh   = 0.45
g_nms_thresh    = 0.45
g_sensor        = None
g_wifi_running  = True

# 线程间通信变量
g_shared_jpg = None
g_ctrl_cmd = None

# GPIO
fpioa = FPIOA()
for _p, _f in ((62, FPIOA.GPIO62), (20, FPIOA.GPIO20), (63, FPIOA.GPIO63)):
    fpioa.set_function(_p, _f)
LED_R = Pin(62, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
LED_G = Pin(20, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
LED_B = Pin(63, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
LED_R.high(); LED_G.high(); LED_B.high()

# UART
fpioa.set_function(UART_RX_PIN, FPIOA.UART2_RXD)
fpioa.set_function(UART_TX_PIN, FPIOA.UART2_TXD)
uart = UART(UART.UART2, baudrate=UART_BAUD, bits=UART.EIGHTBITS,
            parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)

# ================= AP 热点 =================
def setup_ap(ssid="Keng230_AP", password="12345678"):
    # 增加重试机制，应对脱机上电时WiFi芯片苏醒慢的问题
    for i in range(5):
        try:
            wlan = network.WLAN(network.AP_IF)
            wlan.active(True)
            time.sleep(1) # 给硬件一点反应时间
            # 脱机时为了确保稳定性，去掉 security 参数，使用最兼容的配置
            wlan.config(ssid=ssid, key=password)
            print("\n" + "="*50)
            print("✅ AP 热点已成功建立!")
            print("👉 手机/电脑请连接 WiFi: %s" % ssid)
            print("👉 密码: %s" % password)
            print("👉 打开浏览器访问: http://%s:%d" % (wlan.ifconfig()[0], SERVER_PORT))
            print("="*50 + "\n")
            return wlan
        except Exception as e:
            print("❌ AP 热点启动失败，正在重试 (%d/5): %s" % (i+1, e))
            time.sleep(1)
    return None

# ================= 触屏 UI =================
class ModeTouchUI:
    BTN_W = 140
    BTN_H = 45
    BTN_Y = 10

    def __init__(self, display_size=(800, 480)):
        self.buttons = {
            'mode1':   (20,  self.BTN_Y, self.BTN_W, self.BTN_H),
            'standby': (180, self.BTN_Y, self.BTN_W, self.BTN_H),
        }
        self._last_touch_time = 0

    def draw(self, img, current_state):
        btn_titles = {'mode1': "Start", 'standby': "Standby"}
        for name, (bx, by, bw, bh) in self.buttons.items():
            is_selected = ((name == 'mode1' and current_state == STATE_MODE1) or
                           (name == 'standby' and current_state == STATE_STANDBY))
            bg_color = (0, 180, 0) if is_selected else (50, 50, 50)
            text_color = (255, 255, 255)
            img.draw_rectangle(bx, by, bw, bh, color=bg_color, fill=True)
            img.draw_rectangle(bx, by, bw, bh, color=(255, 255, 255), thickness=2)
            img.draw_string_advanced(bx + 15, by + 10, 24, btn_titles[name], color=text_color)

    def process_touch(self, tp_dev):
        if tp_dev is None: return None
        now = time.ticks_ms()
        if time.ticks_diff(now, self._last_touch_time) < 300: return None
        try: p = tp_dev.read()
        except: return None
        if p and len(p) > 0:
            tx, ty = p[0].x, p[0].y
            for name, (bx, by, bw, bh) in self.buttons.items():
                if bx <= tx <= bx + bw and by <= ty <= by + bh:
                    self._last_touch_time = now
                    return name
        return None

# ================= HttpStreamer =================
class HttpStreamer:
    def __init__(self, port):
        self.port = port
        self.server_sock = None
        self.client_sock = None
        self.tcp_connect = False

    def start_server(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(('0.0.0.0', self.port))
        self.server_sock.listen(1)
        self.server_sock.setblocking(False)
        print("[HTTP] 等待浏览器访问 http://192.168.169.1:%d ..." % self.port)

    def accept_client(self):
        try:
            cl, addr = self.server_sock.accept()
            cl.setblocking(False)
            
            # 尝试读取 HTTP 请求头
            req = b""
            t0 = time.ticks_ms()
            while time.ticks_diff(time.ticks_ms(), t0) < 500:
                try:
                    data = cl.recv(1024)
                    if data:
                        req += data
                        if b"\r\n\r\n" in req: break
                except OSError as e:
                    if e.args[0] == 11: time.sleep_ms(10)
                    else: break
            
            # 如果访问根目录，返回带有自动重连 JS 的 HTML 页面
            if req.startswith(b"GET / ") or req.startswith(b"GET /?"):
                html = (
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
                    "<!DOCTYPE html><html><head><meta name='viewport' content='width=device-width, initial-scale=1.0'>"
                    "<style>"
                    "body{background:linear-gradient(135deg,#1e1e24 0%,#000 100%);margin:0;padding:20px;font-family:'Segoe UI',sans-serif;color:#fff;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;}"
                    ".card{background:rgba(255,255,255,0.05);padding:20px;border-radius:16px;box-shadow:0 4px 30px rgba(0,0,0,0.5);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.1);text-align:center;max-width:840px;width:100%;box-sizing:border-box;}"
                    "h2{margin-top:0;font-weight:600;letter-spacing:1px;color:#00d2ff;}"
                    "img{width:100%;border-radius:8px;box-shadow:0 8px 16px rgba(0,0,0,0.6);}"
                    ".status{margin-top:15px;font-size:14px;color:#ccc;display:flex;align-items:center;justify-content:center;gap:8px;}"
                    ".dot{width:10px;height:10px;background:#00ff00;border-radius:50%;box-shadow:0 0 10px #00ff00;animation:blink 1.5s infinite;}"
                    "@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.4;}}"
                    "</style></head><body>"
                    "<div class='card'>"
                    "<h2>YOLO11 Live Stream</h2>"
                    "<img id='v' src='/stream'>"
                    "<div class='status'><div class='dot'></div> Live Tracking</div>"
                    "</div>"
                    "<script>"
                    "var img = document.getElementById('v');"
                    "img.onerror = function() {"
                    "    setTimeout(function() { img.src = '/stream?' + new Date().getTime(); }, 1500);"
                    "};"
                    "</script></body></html>"
                )
                try:
                    cl.setblocking(True)
                    cl.send(html.encode('utf-8'))
                    cl.close()
                except: pass
                return False

            self.client_sock = cl
            self.tcp_connect = True
            print("[HTTP] 浏览器已连接(流): %s" % str(addr))
            
            headers = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: close\r\n\r\n"
            )
            self._send_nonblocking(headers.encode('utf-8'))
            return True
        except OSError:
            return False
        except Exception:
            return False

    def _drop(self, why=""):
        print("[HTTP] 断开 %s" % why)
        if self.client_sock:
            try: self.client_sock.close()
            except: pass
        self.client_sock = None
        self.tcp_connect = False

    def _send_nonblocking(self, data):
        """核心机制：非阻塞发送 + 睡眠让出 GIL，完美解决 AI 卡顿"""
        if not self.client_sock: return False
        data_view = memoryview(data)
        total = len(data_view)
        sent = 0
        timeout_ms = 3000 # 增加到 3 秒，防止网络稍微一波动就断开
        t0 = time.ticks_ms()
        while sent < total:
            if time.ticks_diff(time.ticks_ms(), t0) > timeout_ms:
                raise OSError("Timeout")
            try:
                n = self.client_sock.send(data_view[sent:])
                if n > 0:
                    sent += n
                    t0 = time.ticks_ms()
            except OSError as e:
                if e.args[0] == 11:
                    time.sleep_ms(15) # 让出更多算力
                else:
                    raise e
        return True

    def send_jpeg(self, jpeg_data):
        if not self.client_sock: return False
        try:
            sz = len(jpeg_data)
            header = (
                "--frame\r\n"
                "Content-Type: image/jpeg\r\n"
                "Content-Length: %d\r\n\r\n" % sz
            ).encode('utf-8')
            
            self._send_nonblocking(header)
            self._send_nonblocking(jpeg_data)
            self._send_nonblocking(b"\r\n")
            return True
        except OSError as e:
            self._drop("send err %s" % (e.args[0] if e.args else "timeout"))
            return False
        except Exception as e:
            self._drop("send %s" % e)
            return False

    def destroy(self):
        self._drop()
        if self.server_sock:
            try: self.server_sock.close()
            except: pass

# ================= WiFi 后台线程 =================
def wifi_worker_thread(streamer):
    global g_shared_jpg
    print("[WiFi Thread] 启动")
    while g_wifi_running:
        if not streamer.client_sock:
            streamer.accept_client()
        if g_shared_jpg:
            streamer.send_jpeg(g_shared_jpg)
            g_shared_jpg = None
        time.sleep_ms(10)

# ================= 主函数 =================
def main():
    global g_sensor, g_current_state, g_conf_thresh, g_nms_thresh
    global g_shared_jpg, g_ctrl_cmd
    print("YOLO11 MJPEG Web 开启")
    wlan = setup_ap(WIFI_SSID, WIFI_PWD)
    if not wlan: return
    ap_ip = wlan.ifconfig()[0]

    gc.collect()
    try: gc.threshold(gc.mem_free() // 4 + gc.mem_alloc())
    except: pass

    streamer = None
    try:
        g_sensor = Sensor(id=2, fps=30)
        g_sensor.reset()
        g_sensor.set_framesize(width=STREAM_W, height=STREAM_H, chn=CAM_CHN_ID_0, buffer_num=3)
        g_sensor.set_pixformat(PIXEL_FORMAT_RGB_565, chn=CAM_CHN_ID_0)
        g_sensor.set_framesize(width=LCD_W, height=LCD_H, chn=CAM_CHN_ID_1, buffer_num=3)
        g_sensor.set_pixformat(PIXEL_FORMAT_RGB_565, chn=CAM_CHN_ID_1)
        g_sensor.set_framesize(width=AI_W, height=AI_H, chn=CAM_CHN_ID_2, buffer_num=2)
        g_sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)

        streamer = HttpStreamer(SERVER_PORT)
        streamer.start_server()
        print("[AP] 地址: %s:%d" % (ap_ip, SERVER_PORT))

        try:
            Display.init(Display.ST7701, width=LCD_W, height=LCD_H, to_ide=True)
            print("[Display] ST7701 就绪")
        except:
            try:
                Display.init(Display.LCD, width=LCD_W, height=LCD_H, to_ide=True)
                print("[Display] LCD 就绪")
            except Exception as e:
                print("[Display] 失败: %s" % e)

        g_sensor.run()
    except Exception as e:
        print("[Init] 失败: %s" % e)
        try: Display.deinit()
        except: pass
        try: MediaManager.deinit()
        except: pass
        return

    try:
        tp = TOUCH(0)
        print("[TOUCH] 触摸屏就绪")
    except:
        tp = None
        print("[TOUCH] 无触摸屏")

    mode_ui = ModeTouchUI(display_size=(LCD_W, LCD_H))

    print("[AI] 正在初始化 YOLO...")
    yolo = None
    try:
        yolo = YOLO11(task_type="detect", mode="video", kmodel_path=KMODEL_PATH, labels=LABELS,
                      rgb888p_size=[AI_W, AI_H], model_input_size=[320, 320],
                      display_size=[AI_W, AI_H],
                      conf_thresh=g_conf_thresh, nms_thresh=g_nms_thresh, max_boxes_num=MAX_DETS, debug_mode=0)
        yolo.config_preprocess()
    except Exception as e:
        print("[AI] 初始化失败: %s" % e)
        return

    print("=== 系统就绪（主控+WiFi双线程架构） ===")

    fps_t0 = time.ticks_ms(); fps_n = 0
    stat_t0 = time.ticks_ms()
    uart_send_t0 = 0
    jpg_t0 = 0
    delta_x_cm = 0.0

    # 启动 WiFi 线程
    _thread.start_new_thread(wifi_worker_thread, (streamer,))

    try:
        while True:
            os.exitpoint()
            now = time.ticks_ms()

            # 1. 触屏检测
            touch_btn = mode_ui.process_touch(tp)
            if touch_btn == 'mode1' and g_current_state != STATE_MODE1:
                g_current_state = STATE_MODE1
                g_ctrl_cmd = 1
            elif touch_btn == 'standby' and g_current_state != STATE_STANDBY:
                g_current_state = STATE_STANDBY
                g_ctrl_cmd = 0

            # 2. 抓取图像进行 AI 推理，并立即获取显示帧，确保完全同步
            dets_now = []
            img = None
            if g_current_state == STATE_MODE1:
                if hasattr(yolo, 'conf_thresh'): yolo.conf_thresh = g_conf_thresh
                if hasattr(yolo, 'confidence_threshold'): yolo.confidence_threshold = g_conf_thresh
                if hasattr(yolo, 'nms_thresh'): yolo.nms_thresh = g_nms_thresh
                if hasattr(yolo, 'nms_threshold'): yolo.nms_threshold = g_nms_thresh

                try:
                    # 连续抓取两个通道，保证同一时间的画面 (消除拖影的关键)
                    frame = g_sensor.snapshot(chn=CAM_CHN_ID_2)
                    try: img = g_sensor.snapshot(chn=CAM_CHN_ID_1)
                    except: pass

                    res = yolo.run(frame.to_numpy_ref())

                    if res and len(res) >= 3:
                        boxes, cids, scores = res[0], res[1], res[2]
                        n = len(boxes)
                        if n > MAX_DETS: n = MAX_DETS
                        for i in range(n):
                            b = boxes[i]
                            x1, y1, x2, y2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            if AI_ROI_X1 <= cx <= AI_ROI_X2 and AI_ROI_Y1 <= cy <= AI_ROI_Y2:
                                dets_now.append((x1, y1, x2, y2, scores[i], cids[i]))
                except Exception as e:
                    print("[AI] 推理异常: %s" % e)
            else:
                try: img = g_sensor.snapshot(chn=CAM_CHN_ID_1)
                except: pass

            # 同步给 WiFi 线程
            g_shared_dets = dets_now

            # 3. 计算与串口发送 (使用刚刚算出的最新 dets_now)
            if g_current_state == STATE_MODE1:
                if dets_now:
                    ball_cx_ai = (dets_now[0][0] + dets_now[0][2]) / 2.0
                    ball_cx_lcd = ball_cx_ai * LCD_W / AI_W

                    delta_x_cm = (ball_cx_lcd - ORIGIN_X_LCD) / PIXELS_PER_CM

                    if time.ticks_diff(now, uart_send_t0) >= UART_SEND_MS:
                        uart.write("{:.2f}\r\n".format(delta_x_cm))
                        uart_send_t0 = now

            # 4. 在刚刚抓取的 img 上叠加绘制并显示
            if img is not None:
                img.draw_rectangle(ROI_X, ROI_Y, ROI_W, ROI_H, color=(0, 255, 0), thickness=2)
                img.draw_line(POS_N5CM_X, ROI_Y, POS_N5CM_X, ROI_Y + ROI_H, color=(0, 0, 255), thickness=2)
                img.draw_string_advanced(POS_N5CM_X - 22, ROI_Y - 25, 20, "-5cm", color=(0, 0, 255))
                img.draw_line(ORIGIN_X_LCD, ROI_Y, ORIGIN_X_LCD, ROI_Y + ROI_H, color=(255, 255, 0), thickness=1)
                img.draw_string_advanced(ORIGIN_X_LCD - 15, ROI_Y - 25, 20, "0cm", color=(255, 255, 0))
                img.draw_line(POS_5CM_X, ROI_Y, POS_5CM_X, ROI_Y + ROI_H, color=(0, 0, 255), thickness=2)
                img.draw_string_advanced(POS_5CM_X - 15, ROI_Y - 25, 20, "+5cm", color=(0, 0, 255))

                if g_current_state == STATE_MODE1:
                    start_y = 65
                    if len(dets_now) > 0:
                        img.draw_circle(25, start_y + 10, 10, color=(255, 0, 0), fill=True)
                        img.draw_string_advanced(45, start_y, 24, "Track", color=(255, 0, 0))

                    for d in dets_now:
                        x1 = int(d[0] * LCD_W / AI_W)
                        y1 = int(d[1] * LCD_H / AI_H)
                        x2 = int(d[2] * LCD_W / AI_W)
                        y2 = int(d[3] * LCD_H / AI_H)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        r = max(2, (x2 - x1 + y2 - y1) // 4)
                        img.draw_circle(cx, cy, r, color=(255, 0, 0), thickness=2)
                        img.draw_cross(cx, cy, color=(0, 255, 255), size=6, thickness=2)

                    delta_text = "X: {:.2f} cm".format(delta_x_cm)
                    img.draw_string_advanced(22, start_y + 35, 24, delta_text, color=(0, 0, 0))
                    img.draw_string_advanced(20, start_y + 33, 24, delta_text, color=(0, 255, 0))

                elif g_current_state == STATE_STANDBY:
                    img.draw_rectangle(259, 291, 282, 53, color=(0, 0, 0), fill=True)
                    img.draw_string_advanced(289, 304, 26, "Standby Mode", color=(255, 255, 255))

                mode_ui.draw(img, g_current_state)

                try:
                    Display.show_image(img)
                    fps_n += 1
                except: pass

                # 独立推流通道 (320x240)：保证 15 FPS 的流畅画面且几乎不占 AI 算力
                if streamer.tcp_connect and time.ticks_diff(now, jpg_t0) >= 66:
                    try:
                        stream_img = g_sensor.snapshot(chn=CAM_CHN_ID_0)
                        
                        # 在监视小图上绘制基础追踪框 (极速绘制)
                        if g_current_state == STATE_MODE1:
                            for d in dets_now:
                                sx1 = int(d[0] * STREAM_W / AI_W)
                                sy1 = int(d[1] * STREAM_H / AI_H)
                                sx2 = int(d[2] * STREAM_W / AI_W)
                                sy2 = int(d[3] * STREAM_H / AI_H)
                                scx, scy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
                                sr = max(2, (sx2 - sx1 + sy2 - sy1) // 4)
                                stream_img.draw_circle(scx, scy, sr, color=(255, 0, 0), thickness=2)
                                stream_img.draw_cross(scx, scy, color=(0, 255, 255), size=6, thickness=1)

                            # 绘制中心参考线
                            roi_sy = int(ROI_Y * STREAM_H / LCD_H)
                            roi_sh = int(ROI_H * STREAM_H / LCD_H)
                            stream_img.draw_rectangle(0, roi_sy, STREAM_W, roi_sh, color=(0, 255, 0), thickness=1)
                            o_x = int(ORIGIN_X_LCD * STREAM_W / LCD_W)
                            stream_img.draw_line(o_x, roi_sy, o_x, roi_sy + roi_sh, color=(255, 255, 0), thickness=1)

                        jpg = stream_img.compress(quality=35)
                        if hasattr(jpg, "to_bytes"):
                            g_shared_jpg = jpg.to_bytes()
                        else:
                            g_shared_jpg = bytes(jpg)
                    except Exception as e:
                        print("Stream err:", e)
                    jpg_t0 = now

            if time.ticks_diff(now, stat_t0) >= 3000:
                d = time.ticks_diff(now, fps_t0)
                fps = (fps_n * 1000 / d) if d > 0 else 0
                print("FPS:%.1f det=%d" % (fps, len(dets_now)))
                fps_t0 = now; fps_n = 0; stat_t0 = now

    except KeyboardInterrupt:
        print("stop by user")
    except Exception as e:
        print("Err: %s" % e)
    finally:
        global g_wifi_running
        g_wifi_running = False
        try: yolo.deinit()
        except: pass
        time.sleep(1)
        if g_sensor: g_sensor.stop()
        if streamer: streamer.destroy()
        uart.deinit()
        Display.deinit()
        MediaManager.deinit()
        gc.collect()

if __name__ == "__main__":
    main()
