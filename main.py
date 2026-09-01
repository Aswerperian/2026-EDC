# ==============================================================================
# 2026年全国大学生电子设计竞赛（TI杯）H题：车载平衡滚球运动控制系统
# K230 (庐山派) 视觉定位与无线图传核心处理程序
# 
# 作者: Aswerperian
# 平台: CanMV K230 / K230-CanMV-OS (MicroPython)
# 
# 核心功能模块:
# 1. 多通道传感器硬件流水线 (Sensor Multi-Channel Pipeline):
#    - CH0 (320x240 RGB565): 专用低分辨率/低带宽 WiFi MJPEG 图传流
#    - CH1 (800x480 RGB565): ST7701 4寸高清液晶触摸屏实时渲染与 UI 交互
#    - CH2 (320x240 RGB888P): KPU 硬件加速器专用 YOLO 深度学习目标检测输入
# 2. YOLOv5 nano 硬件加速滚球检测与 ROI 空间约束 (AI Inference & ROI Filter):
#    - 定制轻量化 YOLOv5 nano 钢球检测模型 (myball_v5n.kmodel)
#    - 水平凹槽 ROI 动态过滤，杜绝反光与车体背景干扰
# 3. 几何标定与坐标转换算法 (Calibration & Physical Coordinate Mapping):
#    - 像素坐标到物理真实尺寸(cm)的线性映射: 25.0 pixel/cm
#    - 实时计算钢球相对于摆杆中点(O点)的物理偏差 Delta X (cm)
# 4. 100Hz 高频低时延串口控制通信 (UART Control Stream):
#    - 向上位机/平衡控制底盘(STM32/MSPM0)高速发送 Delta X 位置偏差
# 5. 双线程独立并发架构 (Dual-Thread Decoupled Architecture):
#    - 主线程: 视觉采集 + YOLO 推理 + 坐标解算 + 屏幕渲染 + UART 发送 (保证控制实时性)
#    - 后台线程: WiFi AP 热点维护 + Socket HTTP/MJPEG 视频流分发 (让出GIL，网络IO不阻塞控制)
# 6. ST7701 电容触屏人机交互界面 (Touchscreen GUI):
#    - Start (Mode1 运行)/Standby (待机) 一键触控切换与状态机管理
# ==============================================================================

import os, sys, gc, socket, ustruct, time, network, _thread
import uctypes
from machine import Pin, UART, FPIOA, TOUCH

# 添加 SD 卡依赖库搜索路径
sys.path.append('./sdcard/')

# K230 硬件多媒体与 AI 驱动组件
from media.sensor import *
from media.display import *
from media.media import *
from media.vencoder import *
from libs.YOLO import YOLO11
import image

# ==============================================================================
# 一、 系统状态机常量定义 (State Machine Constants)
# ==============================================================================
STATE_STANDBY = 0  # 待机模式：关闭 AI 运算，仅保持基础图像显示与待机 UI，节省功耗
STATE_MODE1   = 1  # 运行模式：开启 YOLO 视觉检测、坐标解算、UART 偏差数据发送及实时图传

# ==============================================================================
# 二、 系统全局参数与硬件配置 (System Configuration)
# ==============================================================================

# --- 2.1 无线网络与 HTTP 图传配置 ---
WIFI_SSID    = "Keng230_AP"     # K230 作为 AP 热点发射的 SSID 名称
WIFI_PWD     = "51850107"       # AP 热点密码 (至少8位)
SERVER_PORT  = 6001             # HTTP MJPEG 视频流服务端口
PACKET_MAGIC = 0x55AA55AA       # 备用网络通信魔数包头

# --- 2.2 多通道视频分辨率配置 ---
STREAM_W, STREAM_H = 320, 240   # 图传通道分辨率 (Channel 0): 保持低带宽与低时延
AI_W,     AI_H     = 320, 240   # AI 运算输入分辨率 (Channel 2): 匹配 YOLO 输入模型
LCD_W,    LCD_H    = 800, 480   # 屏幕显示分辨率 (Channel 1): 匹配 ST7701 800x480 LCD

# --- 2.3 YOLO 深度学习模型与推理参数 ---
KMODEL_PATH  = "/sdcard/myball_v5n.kmodel"  # KPU 部署的 kmodel 钢球检测模型路径
LABELS       = ['steel']                    # 模型类别标签列表 (检测目标: 钢球)
IDR_INTERVAL = 60                           # 关键帧间隔
DET_SEND_MS  = 66                           # 检测帧最小发送间隔 (~15fps)
MAX_DETS     = 20                           # 单帧最大允许目标检测框数量
OUT_BUFS     = 8                            # 缓冲队列深度

# --- 2.4 水平摆杆凹槽感兴趣区域 (ROI: Region of Interest) ---
# 摆杆在 800x480 LCD 坐标系下的有效物理范围，用于过滤非摆杆区域的无关反光与干扰
ROI_X = 1; ROI_Y = 196; ROI_W = 800; ROI_H = 51  # 凹槽底部 y=247
# 将 LCD 坐标系下的 ROI 转换到 AI (320x240) 坐标系，用于在推理结果中进行快速过滤
AI_ROI_X1 = int(ROI_X * AI_W / LCD_W)
AI_ROI_Y1 = int(ROI_Y * AI_H / LCD_H)
AI_ROI_X2 = int((ROI_X + ROI_W) * AI_W / LCD_W)
AI_ROI_Y2 = int((ROI_Y + ROI_H) * AI_H / LCD_H)

# --- 2.5 串口通信配置 (UART Protocol) ---
UART_TX_PIN  = 11               # 庐山派引出 GPIO 11 -> 复用为 UART2_TXD
UART_RX_PIN  = 12               # 庐山派引出 GPIO 12 -> 复用为 UART2_RXD
UART_BAUD    = 115200           # 波特率 115200 bps
UART_SEND_MS = 10               # 串口发送最小周期: 10ms (100Hz 刷新率，匹配控制环路)

# --- 2.6 物理坐标与像素空间标定参数 (Calibration Parameters) ---
ORIGIN_X_LCD  = 371             # 摆杆物理中心 O 点在 LCD (800x480) 坐标系下的 X 像素坐标
PIXELS_PER_CM = 25.0            # 空间比例尺: 每厘米对应的像素数量 (25.0 pixels/cm)
# 计算 +5cm 与 -5cm 关键位置对应的 LCD X 坐标，用于屏幕辅助标定线绘制
POS_5CM_X  = int(ORIGIN_X_LCD + 5.0 * PIXELS_PER_CM)   # +5cm 位置 (右侧)
POS_N5CM_X = int(ORIGIN_X_LCD - 5.0 * PIXELS_PER_CM)   # -5cm 位置 (左侧)

# 确保硬件编码器 16 字节对齐
def align16(v): 
    """将尺寸对齐到 16 的倍数，满足硬件编码器及 KPU 步进要求"""
    return (v // 16) * 16

if STREAM_W != align16(STREAM_W) or STREAM_H != align16(STREAM_H):
    STREAM_W = align16(STREAM_W)
    STREAM_H = align16(STREAM_H)

SCALE_X = STREAM_W / AI_W
SCALE_Y = STREAM_H / AI_H

# ==============================================================================
# 三、 全局运行状态与线程间共享数据 (Global Runtime & Shared IPC Data)
# ==============================================================================
g_current_state = STATE_STANDBY  # 当前工作状态 (默认开机 Standby)
g_conf_thresh   = 0.45           # YOLO 置信度阈值
g_nms_thresh    = 0.45           # YOLO NMS 非极大值抑制 IoU 阈值
g_sensor        = None           # 摄像头全局句柄
g_wifi_running  = True           # WiFi 后台线程工作标志

# --- 线程间共享通信变量 (Thread-Safe Shared Variables) ---
g_shared_jpg    = None           # 主线程编码完毕的最新 JPEG 数据，供 WiFi 线程非阻塞推流
g_ctrl_cmd      = None           # UI 触控控制指令通知

# ==============================================================================
# 四、 硬件外设初始化 (GPIO / UART Pinmux)
# ==============================================================================
# 4.1 引脚功能复用与 RGB 状态指示灯 (LED)
fpioa = FPIOA()
for _p, _f in ((62, FPIOA.GPIO62), (20, FPIOA.GPIO20), (63, FPIOA.GPIO63)):
    fpioa.set_function(_p, _f)
LED_R = Pin(62, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
LED_G = Pin(20, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
LED_B = Pin(63, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
# 默认拉高熄灭 RGB LED (低电平点亮)
LED_R.high(); LED_G.high(); LED_B.high()

# 4.2 UART2 串口引脚复用与初始化
fpioa.set_function(UART_RX_PIN, FPIOA.UART2_RXD)
fpioa.set_function(UART_TX_PIN, FPIOA.UART2_TXD)
uart = UART(UART.UART2, baudrate=UART_BAUD, bits=UART.EIGHTBITS,
            parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)

# ==============================================================================
# 五、 网络服务与 WiFi AP 建立模块 (WiFi Access Point Setup)
# ==============================================================================
def setup_ap(ssid="Keng230_AP", password="12345678"):
    """
    配置并启动板载 WiFi 为 AP 热点模式，提供鲁棒的重试机制以支持赛场脱机上电。
    
    参数:
        ssid (str): WiFi 热点名称
        password (str): WiFi 热点密码
    返回:
        network.WLAN: 成功时返回 WLAN 对象句柄，失败返回 None
    """
    # 增加 5 次重试机制，应对脱机上电时 WiFi 射频芯片上电时序偏慢的问题
    for i in range(5):
        try:
            wlan = network.WLAN(network.AP_IF)
            wlan.active(True)
            time.sleep(1) # 给射频硬件充分的复位初始化时间
            # 赛场脱机环境采用最兼容配置
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

# ==============================================================================
# 六、 屏幕触摸 UI 交互模块 (Touchscreen GUI Component)
# ==============================================================================
class ModeTouchUI:
    """
    基于 ST7701 触摸屏的简易交互式控制面板。
    支持在屏幕顶部显示 'Start' 与 'Standby' 虚拟触控按键，并进行防抖判定。
    """
    BTN_W = 140
    BTN_H = 45
    BTN_Y = 10

    def __init__(self, display_size=(800, 480)):
        """初始化按键响应区域"""
        self.buttons = {
            'mode1':   (20,  self.BTN_Y, self.BTN_W, self.BTN_H),
            'standby': (180, self.BTN_Y, self.BTN_W, self.BTN_H),
        }
        self._last_touch_time = 0  # 软件消抖时间戳

    def draw(self, img, current_state):
        """
        在显示帧缓冲上绘制当前 UI 状态按钮。
        
        参数:
            img: 待绘制的目标 Image 对象 (800x480 RGB565)
            current_state: 当前系统工作状态 (STATE_MODE1 / STATE_STANDBY)
        """
        btn_titles = {'mode1': "Start", 'standby': "Standby"}
        for name, (bx, by, bw, bh) in self.buttons.items():
            is_selected = ((name == 'mode1' and current_state == STATE_MODE1) or
                           (name == 'standby' and current_state == STATE_STANDBY))
            # 激活状态为绿色，非激活状态为深灰底色
            bg_color = (0, 180, 0) if is_selected else (50, 50, 50)
            text_color = (255, 255, 255)
            img.draw_rectangle(bx, by, bw, bh, color=bg_color, fill=True)
            img.draw_rectangle(bx, by, bw, bh, color=(255, 255, 255), thickness=2)
            img.draw_string_advanced(bx + 15, by + 10, 24, btn_titles[name], color=text_color)

    def process_touch(self, tp_dev):
        """
        读取触摸芯片坐标，检测是否命中虚拟按钮（含 300ms 软件防抖）。
        
        参数:
            tp_dev: TOUCH 设备驱动句柄
        返回:
            str: 命中的按键标识 ('mode1' / 'standby' / None)
        """
        if tp_dev is None: return None
        now = time.ticks_ms()
        if time.ticks_diff(now, self._last_touch_time) < 300: return None
        try: 
            p = tp_dev.read()
        except: 
            return None
        if p and len(p) > 0:
            tx, ty = p[0].x, p[0].y
            for name, (bx, by, bw, bh) in self.buttons.items():
                if bx <= tx <= bx + bw and by <= ty <= by + bh:
                    self._last_touch_time = now
                    return name
        return None

# ==============================================================================
# 七、 高性能 HTTP MJPEG 图传流服务模块 (HTTP MJPEG Streaming Server)
# ==============================================================================
class HttpStreamer:
    """
    非阻塞式 HTTP MJPEG 图传服务器。
    特点:
    1. 支持浏览器直接访问根目录 http://<ip>:port 呈现美观的 Web 流监控页面。
    2. 提供标准的 multipart/x-mixed-replace MJPEG 视频数据流。
    3. 核心机制: 全程非阻塞 Socket 发送 + 主动微秒级 sleep 让出 Python GIL，
       从根本上消除了网络 I/O 导致 AI 视觉推理与底盘控制卡顿的痛点。
    """
    def __init__(self, port):
        self.port = port
        self.server_sock = None
        self.client_sock = None
        self.tcp_connect = False

    def start_server(self):
        """创建非阻塞 TCP 监听套接字"""
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(('0.0.0.0', self.port))
        self.server_sock.listen(1)
        self.server_sock.setblocking(False)
        print("[HTTP] 等待浏览器访问 http://192.168.169.1:%d ..." % self.port)

    def accept_client(self):
        """
        处理客户端连接接入与 HTTP 请求解析。
        返回 True 表示已建立 MJPEG 视频流传输通道。
        """
        try:
            cl, addr = self.server_sock.accept()
            cl.setblocking(False)
            
            # 读取 HTTP 请求头 (超时 500ms)
            req = b""
            t0 = time.ticks_ms()
            while time.ticks_diff(time.ticks_ms(), t0) < 500:
                try:
                    data = cl.recv(1024)
                    if data:
                        req += data
                        if b"\r\n\r\n" in req: break
                except OSError as e:
                    if e.args[0] == 11: # EAGAIN / EWOULDBLOCK
                        time.sleep_ms(10)
                    else: 
                        break
            
            # 若浏览器请求根目录，返回带有自适应 CSS 与自动重连 JS 的 Web 界面
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
                    "<h2>YOLOv5 nano Live Stream</h2>"
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

            # 请求 /stream 视频流通道: 发送 MJPEG 标准响应头
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
        """安全释放客户端连接资源"""
        print("[HTTP] 断开 %s" % why)
        if self.client_sock:
            try: self.client_sock.close()
            except: pass
        self.client_sock = None
        self.tcp_connect = False

    def _send_nonblocking(self, data):
        """
        核心发送机制：使用 memoryview 进行零拷贝切片发送，
        在遇见 EAGAIN 阻塞时主动 sleep_ms(15) 让出 GIL，避免单核过度占用
        """
        if not self.client_sock: return False
        data_view = memoryview(data)
        total = len(data_view)
        sent = 0
        timeout_ms = 3000 # 3 秒通信超时判定
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
                if e.args[0] == 11: # 缓冲区满，让出 GIL 算力
                    time.sleep_ms(15)
                else:
                    raise e
        return True

    def send_jpeg(self, jpeg_data):
        """
        打包并向客户端推送一帧 JPEG 图片。
        
        参数:
            jpeg_data (bytes): 压缩后的 JPEG 二进制字节流
        """
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
        """关闭服务器及所有 Socket"""
        self._drop()
        if self.server_sock:
            try: self.server_sock.close()
            except: pass

# ==============================================================================
# 八、 独立 WiFi 推流后台线程 (WiFi Worker Thread)
# ==============================================================================
def wifi_worker_thread(streamer):
    """
    WiFi 异步服务后台工作线程函数。
    独立于主控制循环运行，负责接收客户端连接请求以及从共享缓冲区获取 JPEG 推送，
    保证网络丢包、重传或等待连接时不影响主控线程的 100Hz 采样与串口通信。
    """
    global g_shared_jpg
    print("[WiFi Thread] 启动")
    while g_wifi_running:
        # 1. 如果没有客户端连接，尝试监听并接入
        if not streamer.client_sock:
            streamer.accept_client()
        # 2. 如果主线程生成了新的 JPEG 图像帧，则推送给客户端
        if g_shared_jpg:
            streamer.send_jpeg(g_shared_jpg)
            g_shared_jpg = None # 消费完成，置空
        time.sleep_ms(10)

# ==============================================================================
# 九、 系统主程序入口与控制闭环 (Main Application Lifecycle)
# ==============================================================================
def main():
    """
    主控制函数：
    1. 硬件外设与多媒体管道初始化
    2. YOLOv5 nano 模型加载与 KPU 配置
    3. 启动 WiFi 异步后台线程
    4. 执行主控制闭环：触控检测 -> 视觉采集 -> YOLO目标检测 -> 空间物理转换 -> UART发送 -> 屏幕渲染
    """
    global g_sensor, g_current_state, g_conf_thresh, g_nms_thresh
    global g_shared_jpg, g_ctrl_cmd
    
    print("=== 2026-EDC H题：车载平衡滚球视觉主控系统启动 (YOLOv5 nano) ===")
    
    # 9.1 启动 AP 热点
    wlan = setup_ap(WIFI_SSID, WIFI_PWD)
    if not wlan: 
        print("❌ WiFi 启动失败，程序终止")
        return
    ap_ip = wlan.ifconfig()[0]

    # 垃圾回收与内存整理
    gc.collect()
    try: gc.threshold(gc.mem_free() // 4 + gc.mem_alloc())
    except: pass

    streamer = None
    try:
        # 9.2 初始化 Sensor 摄像头并配置 3 路硬件独立视频通道
        g_sensor = Sensor(id=2, fps=30)
        g_sensor.reset()
        
        # Channel 0: 320x240 RGB565 -> WiFi 压缩推流
        g_sensor.set_framesize(width=STREAM_W, height=STREAM_H, chn=CAM_CHN_ID_0, buffer_num=3)
        g_sensor.set_pixformat(PIXEL_FORMAT_RGB_565, chn=CAM_CHN_ID_0)
        
        # Channel 1: 800x480 RGB565 -> 本地 LCD 显示与 UI 渲染
        g_sensor.set_framesize(width=LCD_W, height=LCD_H, chn=CAM_CHN_ID_1, buffer_num=3)
        g_sensor.set_pixformat(PIXEL_FORMAT_RGB_565, chn=CAM_CHN_ID_1)
        
        # Channel 2: 320x240 RGB888 Planar -> YOLOv5 nano KPU AI 推理
        g_sensor.set_framesize(width=AI_W, height=AI_H, chn=CAM_CHN_ID_2, buffer_num=2)
        g_sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)

        # 9.3 初始化 HTTP 视频流服务器
        streamer = HttpStreamer(SERVER_PORT)
        streamer.start_server()
        print("[AP] 地址: %s:%d" % (ap_ip, SERVER_PORT))

        # 9.4 初始化 LCD 屏幕显示输出
        try:
            Display.init(Display.ST7701, width=LCD_W, height=LCD_H, to_ide=True)
            print("[Display] ST7701 屏幕驱动就绪")
        except:
            try:
                Display.init(Display.LCD, width=LCD_W, height=LCD_H, to_ide=True)
                print("[Display] 默认 LCD 驱动就绪")
            except Exception as e:
                print("[Display] 初始化失败: %s" % e)

        # 启动摄像头硬件采集流水线
        g_sensor.run()
    except Exception as e:
        print("[Init] 硬件初始化失败: %s" % e)
        try: Display.deinit()
        except: pass
        try: MediaManager.deinit()
        except: pass
        return

    # 9.5 初始化触摸屏设备
    try:
        tp = TOUCH(0)
        print("[TOUCH] 触摸屏驱动就绪")
    except:
        tp = None
        print("[TOUCH] 未检测到触摸屏")

    mode_ui = ModeTouchUI(display_size=(LCD_W, LCD_H))

    # 9.6 加载并初始化 YOLOv5 nano 目标检测模型
    print("[AI] 正在加载 YOLOv5 nano KPU 模型: %s" % KMODEL_PATH)
    yolo = None
    try:
        yolo = YOLO11(task_type="detect", mode="video", kmodel_path=KMODEL_PATH, labels=LABELS,
                      rgb888p_size=[AI_W, AI_H], model_input_size=[320, 320],
                      display_size=[AI_W, AI_H],
                      conf_thresh=g_conf_thresh, nms_thresh=g_nms_thresh, max_boxes_num=MAX_DETS, debug_mode=0)
        yolo.config_preprocess()
        print("[AI] YOLOv5 nano 模型装载成功")
    except Exception as e:
        print("[AI] 初始化失败: %s" % e)
        return

    print("=== 系统就绪（主控+WiFi双线程并发架构） ===")

    # 性能统计与控制计时变量
    fps_t0 = time.ticks_ms(); fps_n = 0
    stat_t0 = time.ticks_ms()
    uart_send_t0 = 0
    jpg_t0 = 0
    delta_x_cm = 0.0

    # 9.7 启动后台 WiFi 工作线程
    _thread.start_new_thread(wifi_worker_thread, (streamer,))

    try:
        while True:
            os.exitpoint()
            now = time.ticks_ms()

            # -------------------------------------------------------------
            # 步骤 1: 触屏事件检测与状态机切换
            # -------------------------------------------------------------
            touch_btn = mode_ui.process_touch(tp)
            if touch_btn == 'mode1' and g_current_state != STATE_MODE1:
                g_current_state = STATE_MODE1
                g_ctrl_cmd = 1
            elif touch_btn == 'standby' and g_current_state != STATE_STANDBY:
                g_current_state = STATE_STANDBY
                g_ctrl_cmd = 0

            # -------------------------------------------------------------
            # 步骤 2: 图像抓取与 YOLO AI 推理 (连续抓取两路，消除时滞拖影)
            # -------------------------------------------------------------
            dets_now = []
            img = None
            if g_current_state == STATE_MODE1:
                # 动态同步置信度与 NMS 阈值
                if hasattr(yolo, 'conf_thresh'): yolo.conf_thresh = g_conf_thresh
                if hasattr(yolo, 'confidence_threshold'): yolo.confidence_threshold = g_conf_thresh
                if hasattr(yolo, 'nms_thresh'): yolo.nms_thresh = g_nms_thresh
                if hasattr(yolo, 'nms_threshold'): yolo.nms_threshold = g_nms_thresh

                try:
                    # 关键技术：连续抓取 AI 通道 (CHN2) 和 显示通道 (CHN1)，确保显示与检测完全同帧同相
                    frame = g_sensor.snapshot(chn=CAM_CHN_ID_2)
                    try: 
                        img = g_sensor.snapshot(chn=CAM_CHN_ID_1)
                    except: 
                        pass

                    # KPU 硬件加速推理
                    res = yolo.run(frame.to_numpy_ref())

                    # 解析推理边界框并应用摆杆水平 ROI 空间滤波
                    if res and len(res) >= 3:
                        boxes, cids, scores = res[0], res[1], res[2]
                        n = len(boxes)
                        if n > MAX_DETS: n = MAX_DETS
                        for i in range(n):
                            b = boxes[i]
                            x1, y1, x2, y2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            # ROI 区域空间约束：只保留落在摆杆凹槽区域内的目标，过滤外部背景误检
                            if AI_ROI_X1 <= cx <= AI_ROI_X2 and AI_ROI_Y1 <= cy <= AI_ROI_Y2:
                                dets_now.append((x1, y1, x2, y2, scores[i], cids[i]))
                except Exception as e:
                    print("[AI] 推理异常: %s" % e)
            else:
                # 待机模式：跳过 AI 推理，仅抓取显示帧
                try: 
                    img = g_sensor.snapshot(chn=CAM_CHN_ID_1)
                except: 
                    pass

            # -------------------------------------------------------------
            # 步骤 3: 几何空间映射与 UART 高速闭环发送 (100Hz 节拍)
            # -------------------------------------------------------------
            if g_current_state == STATE_MODE1:
                if dets_now:
                    # 提取首个有效钢球的中心坐标 (AI 坐标系 -> LCD 坐标系)
                    ball_cx_ai = (dets_now[0][0] + dets_now[0][2]) / 2.0
                    ball_cx_lcd = ball_cx_ai * LCD_W / AI_W

                    # 核心标定公式: Delta X (cm) = (X_pixel - X_origin) / Pixels_Per_Cm
                    delta_x_cm = (ball_cx_lcd - ORIGIN_X_LCD) / PIXELS_PER_CM

                    # 10ms 周期串口发送位置偏差（单位：cm，保留两位小数）
                    if time.ticks_diff(now, uart_send_t0) >= UART_SEND_MS:
                        uart.write("{:.2f}\r\n".format(delta_x_cm))
                        uart_send_t0 = now

            # -------------------------------------------------------------
            # 步骤 4: LCD 屏幕 OSD 画面叠加渲染与触摸 UI 绘制
            # -------------------------------------------------------------
            if img is not None:
                # 4.1 绘制摆杆物理参考系 (绿色 ROI 框与 0cm / ±5cm 刻度标记线)
                img.draw_rectangle(ROI_X, ROI_Y, ROI_W, ROI_H, color=(0, 255, 0), thickness=2)
                
                # -5cm 物理标定线 (蓝色)
                img.draw_line(POS_N5CM_X, ROI_Y, POS_N5CM_X, ROI_Y + ROI_H, color=(0, 0, 255), thickness=2)
                img.draw_string_advanced(POS_N5CM_X - 22, ROI_Y - 25, 20, "-5cm", color=(0, 0, 255))
                
                # 0cm 中心平衡点 O 标定线 (黄色)
                img.draw_line(ORIGIN_X_LCD, ROI_Y, ORIGIN_X_LCD, ROI_Y + ROI_H, color=(255, 255, 0), thickness=1)
                img.draw_string_advanced(ORIGIN_X_LCD - 15, ROI_Y - 25, 20, "0cm", color=(255, 255, 0))
                
                # +5cm 物理标定线 (蓝色)
                img.draw_line(POS_5CM_X, ROI_Y, POS_5CM_X, ROI_Y + ROI_H, color=(0, 0, 255), thickness=2)
                img.draw_string_advanced(POS_5CM_X - 15, ROI_Y - 25, 20, "+5cm", color=(0, 0, 255))

                # 4.2 运行模式下的追踪目标绘制
                if g_current_state == STATE_MODE1:
                    start_y = 65
                    if len(dets_now) > 0:
                        img.draw_circle(25, start_y + 10, 10, color=(255, 0, 0), fill=True)
                        img.draw_string_advanced(45, start_y, 24, "Track", color=(255, 0, 0))

                    # 绘制检测框与十字瞄准准星
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

                    # 实时物理偏差数值显示
                    delta_text = "X: {:.2f} cm".format(delta_x_cm)
                    img.draw_string_advanced(22, start_y + 35, 24, delta_text, color=(0, 0, 0))       # 黑色阴影
                    img.draw_string_advanced(20, start_y + 33, 24, delta_text, color=(0, 255, 0))     # 绿色高亮

                # 4.3 待机模式提示
                elif g_current_state == STATE_STANDBY:
                    img.draw_rectangle(259, 291, 282, 53, color=(0, 0, 0), fill=True)
                    img.draw_string_advanced(289, 304, 26, "Standby Mode", color=(255, 255, 255))

                # 4.4 绘制触摸按键 UI
                mode_ui.draw(img, g_current_state)

                # 4.5 刷新输出到 ST7701 液晶屏
                try:
                    Display.show_image(img)
                    fps_n += 1
                except: pass

                # -------------------------------------------------------------
                # 步骤 5: 独立推流通道 (Channel 0, 320x240) 帧压缩与共享
                # -------------------------------------------------------------
                # 控制图传帧率为 ~15 FPS (66ms 周期)，画质 35%，兼顾流畅度与无线信道稳定性
                if streamer.tcp_connect and time.ticks_diff(now, jpg_t0) >= 66:
                    try:
                        stream_img = g_sensor.snapshot(chn=CAM_CHN_ID_0)
                        
                        # 在监视小图上绘制基础追踪框与刻度线 (极速轻量绘制)
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

                        # 压缩为 JPEG 并共享给 WiFi 后台工作线程
                        jpg = stream_img.compress(quality=35)
                        if hasattr(jpg, "to_bytes"):
                            g_shared_jpg = jpg.to_bytes()
                        else:
                            g_shared_jpg = bytes(jpg)
                    except Exception as e:
                        print("Stream err:", e)
                    jpg_t0 = now

            # -------------------------------------------------------------
            # 步骤 6: 性能监视与控制台调试输出 (每 3 秒统计一次 FPS)
            # -------------------------------------------------------------
            if time.ticks_diff(now, stat_t0) >= 3000:
                d = time.ticks_diff(now, fps_t0)
                fps = (fps_n * 1000 / d) if d > 0 else 0
                print("FPS:%.1f det=%d" % (fps, len(dets_now)))
                fps_t0 = now; fps_n = 0; stat_t0 = now

    except KeyboardInterrupt:
        print("🛑 用户中断退出")
    except Exception as e:
        print("❌ 运行异常: %s" % e)
    finally:
        # 系统资源安全释放
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
        print("✅ 硬件资源与服务已安全释放")

if __name__ == "__main__":
    main()

