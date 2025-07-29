# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 15:24:20 2025

@author: Chiakai
"""

import os, base64, hashlib, webbrowser, threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError
import numpy as np, cv2
from deepface import DeepFace

'''
人臉辨識參考: https://github.com/serengil/deepface
pip install deepface
'''


# ---------- 工具 ----------
def imread_unicode(path: str) -> np.ndarray:
    """支援中文路徑讀取影像，回傳 BGR ndarray（cv2 格式）"""
    with open(path, "rb") as f:
        data = bytearray(f.read())
    img = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2 讀取失敗：{path}")
    return img

def img_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def calc_similarity(distance: float, threshold: float) -> tuple:
    similarity_a = max(0, (1 - (distance / threshold)) * 100)
    similarity_b = max(0, (1 - distance) * 100)
    return similarity_a, similarity_b

def gauge_color_html(value: float) -> str:
    if value >= 80:
        return "green"
    elif value >= 50:
        return "orange"
    else:
        return "red"


# ---------- GUI ----------
class DeepFaceGUI:
    ver = '20250729_1'
    
    def __init__(self, master):
        self.master = master
        master.title(f"DeepFace 臉部比對工具 ({self.ver})")

        self.img1_path = self.img2_path = None
        self.img1_arr  = self.img2_arr  = None
        
        ui_show = f'''
DeepFace 臉部比對工具 ({self.ver})
(參考: https://github.com/serengil/deepface)
UI、報告格式之設計: Chiakai
        '''.strip()
        
        tk.Label(master, text=ui_show, fg="gray").pack(padx=20, pady=10)
        
        prev_frame = tk.Frame(master); prev_frame.pack()
        self.prev1 = tk.Label(prev_frame, text="尚未選擇圖片 1", bd=2, relief="groove")
        self.prev1.grid(row=0, column=0, padx=8, pady=4)
        self.prev2 = tk.Label(prev_frame, text="尚未選擇圖片 2", bd=2, relief="groove")
        self.prev2.grid(row=0, column=1, padx=8, pady=4)

        btn_frame = tk.Frame(master); btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="選擇圖片 1", command=lambda: self.choose_image(1)).grid(row=0, column=0, padx=15)
        tk.Button(btn_frame, text="選擇圖片 2", command=lambda: self.choose_image(2)).grid(row=0, column=1, padx=15)

        self.btn_compare = tk.Button(master, text="開始比對", command=self.compare, state="disabled")
        self.btn_compare.pack(pady=8)

        self.msg = tk.Label(master, text="", fg="blue"); self.msg.pack(pady=6)

    def choose_image(self, idx):
        path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path: return
        try:
            pil_img = Image.open(path)
            pil_img.thumbnail((240, 240))
            tk_img = ImageTk.PhotoImage(pil_img)
        except UnidentifiedImageError:
            messagebox.showerror("錯誤", "無法讀取圖片（格式不支援）")
            return

        try:
            arr = imread_unicode(path)
        except Exception as e:
            messagebox.showerror("錯誤", f"cv2 讀圖失敗：{e}")
            return

        if idx == 1:
            self.img1_path, self.img1_arr = path, arr
            self.prev1.configure(image=tk_img, text=""); self.prev1.image = tk_img
        else:
            self.img2_path, self.img2_arr = path, arr
            self.prev2.configure(image=tk_img, text=""); self.prev2.image = tk_img

        if self.img1_arr is not None and self.img2_arr is not None:
            self.btn_compare.config(state="normal")

    def compare(self):
        self.msg.config(text="比對中，請稍候…", fg="orange")
        self.btn_compare.config(state="disabled")
        threading.Thread(target=self._verify, daemon=True).start()

    def _verify(self):
        try:
            result = DeepFace.verify(
                img1_path=self.img1_arr,
                img2_path=self.img2_arr,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )

            dist = result["distance"]
            thresh = result["threshold"]
            sim_a, sim_b = calc_similarity(dist, thresh)
            status = "✅ 同一人" if result["verified"] else "❌ 不同人"
            color = "green" if result["verified"] else "red"

            self.msg.config(
                text=f"{status}\n距離: {dist:.4f} (閾值 {thresh:.4f})\n"
                     f"相似度A: {sim_a:.1f}% / 相似度B: {sim_b:.1f}%",
                fg=color
            )

            report = self._make_report(result, sim_a, sim_b)
            self.msg.config(text=f"{status}\n報告: {report}", fg=color)

        except Exception as e:
            self.msg.config(text=f"比對發生錯誤：{e}", fg="red")
        finally:
            self.btn_compare.config(state="normal")

    def _make_report(self, res, sim_a, sim_b):
        sim_a_color = gauge_color_html(sim_a)
        sim_b_color = gauge_color_html(sim_b)

        html = f"""
        <!DOCTYPE html>
        <html lang="zh">
        <meta charset="utf-8">
        <title>DeepFace Report</title>
        <style>
            body {{ font-family:"Segoe UI",Arial,sans-serif; background:#f7f7f7; margin:0; }}
            .container {{ max-width: 960px; margin: auto; background:#fff; padding: 24px; 
                         box-shadow:0 4px 12px rgba(0,0,0,.1); }}
            h2 {{ text-align:center; margin-top:0; }}
            .faces {{ display:flex; justify-content:space-between; gap:20px; margin:20px 0; }}
            .card {{ flex:1; text-align:center; border:1px solid #ddd; border-radius:8px; padding:12px; }}
            .card img {{ width:100%; height:auto; max-height:320px; object-fit:contain; border-radius:4px; }}
            .hash {{ font-size:12px; color:#555; word-break:break-all; margin-top:4px; }}
            @media (max-width: 768px) {{ .faces {{ flex-direction:column; }} }}
        </style>
        <body>
        <div class="container">
            <h2>DeepFace 臉部比對報告</h2>
            <p><b>生成時間：</b>{datetime.now():%Y-%m-%d %H:%M:%S}</p>

            <div class="faces">
                <div class="card">
                    <h3>圖片 1：{os.path.basename(self.img1_path)}</h3>
                    <img src="data:image/jpeg;base64,{img_b64(self.img1_path)}">
                    <div class="hash">SHA256: {img_sha256(self.img1_path)}</div>
                </div>
                <div class="card">
                    <h3>圖片 2：{os.path.basename(self.img2_path)}</h3>
                    <img src="data:image/jpeg;base64,{img_b64(self.img2_path)}">
                    <div class="hash">SHA256: {img_sha256(self.img2_path)}</div>
                </div>
            </div>

            <h3>比對結果</h3>
            <p><b>系統判定：</b> {"✅ 同一人" if res['verified'] else "❌ 不同人"}</p>
            <ul>
                <li><b>distance：</b>{res['distance']:.4f}（數值愈小愈像）</li>
                <li><b>threshold：</b>{res['threshold']:.4f}（門檻值）</li>
            </ul>

            <h3>相似度換算</h3>
            <ol>
                <li>方法 A（與門檻值比較）：<b style="color:{sim_a_color};">{sim_a:.1f}%</b>
                    <br><small>越高代表越安全通過門檻</small></li>
                <li>方法 B（反距離法）：<b style="color:{sim_b_color};">{sim_b:.1f}%</b>
                    <br><small>直觀的相似百分比，僅供參考</small></li>
            </ol>

            <p style="color:#666;">提示：<u>系統判定 (✅ / ❌) 才是主要依據</u>，相似度 A 看餘裕，B 為粗略百分比。</p>
        </div>
        </body>
        </html>
        """

        report_name = "deepface_report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".html"
        save_dir = os.path.dirname(self.img1_path) or os.getcwd()
        fp = os.path.join(save_dir, report_name)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(html)
        webbrowser.open(f"file://{fp}")
        return fp


# ---------- 執行 ----------
if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    root = tk.Tk()
    DeepFaceGUI(root)
    root.mainloop()
