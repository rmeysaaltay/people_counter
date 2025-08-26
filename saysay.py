import cv2
from ultralytics import YOLO
import numpy as np
import time
import gradio as gr
from PIL import Image
import io
import threading
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os


class AVMPeopleCounter:
    def __init__(self, person_model_path, gender_model_path):
        self.person_model = YOLO(person_model_path)
        self.gender_model = YOLO(gender_model_path)

        self.total_count = 0
        self.male_count = 0
        self.female_count = 0
        self.unknown_count = 0

        self.tracked_objects = {}
        self.next_id = 0

        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0

        # Gradio için ek değişkenler
        self.is_running = False
        self.current_frame = None

        # Zaman serisi veriler için
        self.time_series_data = []
        self.start_time = time.time()

        # Video analizi için frame veriler
        self.video_analysis_data = []

    def reset_counters(self):
        """Sayaçları sıfırla"""
        self.total_count = 0
        self.male_count = 0
        self.female_count = 0
        self.unknown_count = 0
        self.tracked_objects = {}
        self.next_id = 0
        self.time_series_data = []
        self.video_analysis_data = []
        self.start_time = time.time()
        return "Sayaçlar sıfırlandı!"

    def detect_persons(self, frame):
        results = self.person_model(frame, conf=0.5)
        persons = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                persons.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': conf
                })

        return persons

    def detect_gender(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)

        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size > 0:
            results = self.gender_model(person_crop, conf=0.3)

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                best_result = None
                best_conf = 0

                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if conf > best_conf:
                        best_conf = conf
                        best_result = cls

                if best_result is not None:
                    gender = 'Erkek' if best_result == 0 else 'Kadın'
                    return gender, best_conf

        return 'Bilinmiyor', 0.0

    def simple_tracking(self, current_persons, max_distance=50):
        matched_ids = []

        for person in current_persons:
            center = person['center']
            best_match_id = None
            best_distance = float('inf')

            for obj_id, obj_info in self.tracked_objects.items():
                if obj_id not in matched_ids:
                    distance = np.sqrt((center[0] - obj_info['center'][0]) ** 2 +
                                       (center[1] - obj_info['center'][1]) ** 2)

                    if distance < best_distance and distance < max_distance:
                        best_distance = distance
                        best_match_id = obj_id

            if best_match_id is not None:
                self.tracked_objects[best_match_id].update({
                    'center': center,
                    'bbox': person['bbox'],
                    'last_seen': time.time()
                })
                person['id'] = best_match_id
                matched_ids.append(best_match_id)
            else:
                person['id'] = self.next_id
                self.tracked_objects[self.next_id] = {
                    'center': center,
                    'bbox': person['bbox'],
                    'last_seen': time.time(),
                    'counted': True,
                    'gender': None,
                    'gender_conf': 0.0
                }

                self.total_count += 1
                gender, conf = self.detect_gender(self.current_frame, person['bbox'])
                self.tracked_objects[self.next_id]['gender'] = gender
                self.tracked_objects[self.next_id]['gender_conf'] = conf

                if gender == 'Erkek':
                    self.male_count += 1
                elif gender == 'Kadın':
                    self.female_count += 1
                else:
                    self.unknown_count += 1

                print(f"Yeni kişi bulundu! Toplam: {self.total_count}, Cinsiyet: {gender} (%{conf * 100:.1f})")

                self.next_id += 1

        current_time = time.time()
        to_remove = []
        for obj_id, obj_info in self.tracked_objects.items():
            if current_time - obj_info['last_seen'] > 5:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.tracked_objects[obj_id]

        # Zaman serisi verilerini kaydet
        elapsed_time = current_time - self.start_time
        self.time_series_data.append({
            'time': elapsed_time,
            'total': self.total_count,
            'male': self.male_count,
            'female': self.female_count,
            'unknown': self.unknown_count,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })

    def calculate_fps(self):
        self.fps_counter += 1
        if time.time() - self.fps_time >= 1:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = time.time()

    def draw_interface(self, frame, persons):
        height, width = frame.shape[:2]

        for person in persons:
            if 'id' in person:
                x1, y1, x2, y2 = person['bbox']
                person_id = person['id']

                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"ID: {person_id}"
                if person_id in self.tracked_objects and self.tracked_objects[person_id]['gender']:
                    gender = self.tracked_objects[person_id]['gender']
                    conf = self.tracked_objects[person_id]['gender_conf']
                    label += f" - {gender} ({conf * 100:.1f}%)"

                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        panel_height = 150
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)

        cv2.putText(panel, f"TOPLAM KISI: {self.total_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, f"ERKEK: {self.male_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 150, 255), 2)
        cv2.putText(panel, f"KADIN: {self.female_count}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 150), 2)
        cv2.putText(panel, f"BILINMEYEN: {self.unknown_count}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        cv2.putText(panel, f"FPS: {self.current_fps}", (width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        combined = np.vstack([panel, frame])
        return combined

    def process_frame(self, frame, frame_number=None):
        """Tek frame işle - Gradio için"""
        if frame is None:
            return None

        self.current_frame = frame
        persons = self.detect_persons(frame)
        self.simple_tracking(persons)
        self.calculate_fps()
        processed_frame = self.draw_interface(frame, persons)

        # Video analizi için frame verilerini kaydet
        if frame_number is not None:
            self.video_analysis_data.append({
                'frame': frame_number,
                'total': self.total_count,
                'male': self.male_count,
                'female': self.female_count,
                'unknown': self.unknown_count,
                'persons_in_frame': len(persons)
            })

        # OpenCV BGR'den RGB'ye çevir
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        return processed_frame_rgb

    def get_stats(self):
        """Anlık istatistikleri döndür"""
        return {
            "Toplam Kişi": self.total_count,
            "Erkek": self.male_count,
            "Kadın": self.female_count,
            "Bilinmeyen": self.unknown_count,
            "FPS": self.current_fps
        }

    def get_dataframe(self):
        """İstatistikleri DataFrame olarak döndür"""
        data = {
            'Kategori': ['Toplam', 'Erkek', 'Kadın', 'Bilinmeyen'],
            'Sayı': [self.total_count, self.male_count, self.female_count, self.unknown_count],
            'Yüzde': [
                100.0 if self.total_count > 0 else 0,
                (self.male_count / self.total_count * 100) if self.total_count > 0 else 0,
                (self.female_count / self.total_count * 100) if self.total_count > 0 else 0,
                (self.unknown_count / self.total_count * 100) if self.total_count > 0 else 0
            ]
        }
        return pd.DataFrame(data)

    def create_charts(self):
        """Grafikleri oluştur"""
        # Pasta grafiği
        if self.total_count > 0:
            pie_fig = go.Figure(data=[go.Pie(
                labels=['Erkek', 'Kadın', 'Bilinmeyen'],
                values=[self.male_count, self.female_count, self.unknown_count],
                hole=0.3,
                marker_colors=['#3498db', '#e74c3c', '#95a5a6']
            )])
            pie_fig.update_layout(
                title="Cinsiyet Dağılımı",
                height=400,
                showlegend=True
            )
        else:
            pie_fig = go.Figure()
            pie_fig.add_annotation(text="Henüz veri yok", x=0.5, y=0.5, showarrow=False)
            pie_fig.update_layout(title="Cinsiyet Dağılımı", height=400)

        # Bar grafiği
        bar_fig = go.Figure(data=[
            go.Bar(name='Sayı', x=['Erkek', 'Kadın', 'Bilinmeyen'],
                   y=[self.male_count, self.female_count, self.unknown_count],
                   marker_color=['#3498db', '#e74c3c', '#95a5a6'])
        ])
        bar_fig.update_layout(
            title="Cinsiyet Bazında Sayılar",
            xaxis_title="Cinsiyet",
            yaxis_title="Kişi Sayısı",
            height=400
        )

        # Zaman serisi grafiği
        if len(self.time_series_data) > 1:
            df_time = pd.DataFrame(self.time_series_data)
            time_fig = go.Figure()
            time_fig.add_trace(go.Scatter(x=df_time['time'], y=df_time['total'],
                                          mode='lines+markers', name='Toplam', line=dict(color='#2ecc71')))
            time_fig.add_trace(go.Scatter(x=df_time['time'], y=df_time['male'],
                                          mode='lines+markers', name='Erkek', line=dict(color='#3498db')))
            time_fig.add_trace(go.Scatter(x=df_time['time'], y=df_time['female'],
                                          mode='lines+markers', name='Kadın', line=dict(color='#e74c3c')))
            time_fig.update_layout(
                title="Zaman İçinde Kişi Sayısı",
                xaxis_title="Zaman (saniye)",
                yaxis_title="Kişi Sayısı",
                height=400
            )
        else:
            time_fig = go.Figure()
            time_fig.add_annotation(text="Henüz zaman serisi verisi yok", x=0.5, y=0.5, showarrow=False)
            time_fig.update_layout(title="Zaman İçinde Kişi Sayısı", height=400)

        return pie_fig, bar_fig, time_fig


# Global counter instance
counter = None

# Hazır demo videoları (örnek yollar - kendi video dosyalarınızı ekleyin)
DEMO_VIDEOS = {
    "Video 1 - Market Girişi": "demo_videos/market_entrance.mp4",
    "Video 2 - AVM Koridoru": "demo_videos/mall_corridor.mp4",
    "Video 3 - Mağaza İçi": "demo_videos/store_interior.mp4"
}


def initialize_counter():
    """Counter'ı başlat"""
    global counter
    person_model_path = r"C:\Users\msi-nb\Downloads\person_model_best.pt"
    gender_model_path = r"C:\Users\msi-nb\Downloads\gender_model_best.pt"

    try:
        counter = AVMPeopleCounter(person_model_path, gender_model_path)
        return "✅ Modeller başarıyla yüklendi!"
    except Exception as e:
        return f"❌ Model yükleme hatası: {str(e)}"


def process_webcam_frame(frame):
    """Webcam frame'ini işle"""
    global counter
    if counter is None:
        return frame, {"Durum": "Modeller yüklenmedi"}, None, None, None, None

    try:
        processed_frame = counter.process_frame(frame)
        stats = counter.get_stats()
        df = counter.get_dataframe()
        pie_chart, bar_chart, time_chart = counter.create_charts()
        return processed_frame, stats, df, pie_chart, bar_chart, time_chart
    except Exception as e:
        return frame, {"Hata": str(e)}, None, None, None, None


def process_video_file(video_path, progress=gr.Progress()):
    """Video dosyasını tam olarak işle"""
    global counter
    if counter is None:
        return None, {"Durum": "Modeller yüklenmedi"}, None, None, None, None

    try:
        # Sayaçları sıfırla
        counter.reset_counters()

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames_processed = 0
        last_frame = None

        progress(0, desc="Video işleniyor...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = counter.process_frame(frame, frames_processed)
            last_frame = processed_frame
            frames_processed += 1

            # Progress güncellemesi
            if frames_processed % 10 == 0:
                progress_val = frames_processed / total_frames
                progress(progress_val, desc=f"İşlenen frame: {frames_processed}/{total_frames}")

            # Performans için her 5. frame'i işle
            for _ in range(4):
                ret, _ = cap.read()
                if not ret:
                    break
                frames_processed += 4

        cap.release()

        if last_frame is not None:
            stats = counter.get_stats()
            df = counter.get_dataframe()
            pie_chart, bar_chart, time_chart = counter.create_charts()

            # Video analiz sonuçlarını dataframe'e çevir
            video_df = pd.DataFrame(counter.video_analysis_data) if counter.video_analysis_data else pd.DataFrame()

            progress(1.0, desc="Video analizi tamamlandı!")
            return last_frame, stats, df, pie_chart, bar_chart, time_chart
        else:
            return None, {"Durum": "Video işlenemedi"}, None, None, None, None

    except Exception as e:
        return None, {"Hata": str(e)}, None, None, None, None


def process_demo_video(demo_video_name):
    """Hazır demo videosunu işle"""
    if demo_video_name and demo_video_name in DEMO_VIDEOS:
        video_path = DEMO_VIDEOS[demo_video_name]
        if os.path.exists(video_path):
            return process_video_file(video_path)
        else:
            return None, {"Hata": f"Demo video bulunamadı: {video_path}"}, None, None, None, None
    return None, {"Durum": "Demo video seçilmedi"}, None, None, None, None


def process_single_image(image):
    """Tek görüntüyü işle"""
    global counter
    if counter is None:
        return None, {"Durum": "Modeller yüklenmedi"}, None, None, None, None

    if image is None:
        return None, {"Durum": "Görüntü yüklenmedi"}, None, None, None, None

    try:
        # Reset counter for single image
        counter.reset_counters()
        processed_frame = counter.process_frame(image)
        stats = counter.get_stats()
        df = counter.get_dataframe()
        pie_chart, bar_chart, time_chart = counter.create_charts()
        return processed_frame, stats, df, pie_chart, bar_chart, time_chart
    except Exception as e:
        return None, {"Hata": str(e)}, None, None, None, None


def reset_system():
    """Sistemi sıfırla"""
    global counter
    if counter is not None:
        message = counter.reset_counters()
        stats = counter.get_stats()
        df = counter.get_dataframe()
        pie_chart, bar_chart, time_chart = counter.create_charts()
        return message, stats, df, pie_chart, bar_chart, time_chart
    return "Model yüklenmedi", {}, None, None, None, None


def get_current_stats():
    """Mevcut istatistikleri al"""
    global counter
    if counter is not None:
        stats = counter.get_stats()
        df = counter.get_dataframe()
        pie_chart, bar_chart, time_chart = counter.create_charts()
        return stats, df, pie_chart, bar_chart, time_chart
    return {}, None, None, None, None


# Gradio arayüzü
def create_interface():
    with gr.Blocks(title="AVM İnsan Sayma ve Cinsiyet Tespiti", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏬 AVM İnsan Sayma ve Cinsiyet Tespiti Sistemi")
        gr.Markdown("Bu sistem YOLOv8 kullanarak insanları tespit eder ve cinsiyetlerini belirler.")

        # Başlatma butonu
        with gr.Row():
            init_btn = gr.Button("🚀 Sistemi Başlat", variant="primary", size="lg")
            reset_btn = gr.Button("🔄 Sıfırla", variant="secondary")
            stats_btn = gr.Button("📊 İstatistikleri Güncelle", variant="secondary")

        init_status = gr.Textbox(label="Durum", interactive=False)

        # Ana arayüz
        with gr.Tabs():
            # Webcam sekmesi
            with gr.TabItem("📷 Webcam"):
                with gr.Row():
                    with gr.Column():
                        webcam_input = gr.Image(
                            label="Webcam Girişi",
                            sources=["webcam"],
                            streaming=True
                        )
                    with gr.Column():
                        webcam_output = gr.Image(label="İşlenmiş Görüntü")
                        webcam_stats = gr.JSON(label="Anlık İstatistikler")
            # Video yükleme sekmesi
            with gr.TabItem("📹 Video Yükle"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Video Dosyası Yükle")
                        process_video_btn = gr.Button("▶️ Video İşle", variant="primary")
                    with gr.Column():
                        video_output = gr.Image(label="İşlenmiş Frame")
                        video_stats = gr.JSON(label="Video İstatistikleri")

            # Manuel görüntü yükleme sekmesi
            with gr.TabItem("🖼️ Görüntü Yükle"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Görüntü Yükle",
                            sources=["upload"],
                            type="numpy"
                        )
                        process_image_btn = gr.Button("🔍 Görüntüyü İşle", variant="primary")
                    with gr.Column():
                        image_output = gr.Image(label="İşlenmiş Görüntü")
                        image_stats = gr.JSON(label="Görüntü İstatistikleri")

        # İstatistik ve grafik paneli
        gr.Markdown("## 📊 Analiz Sonuçları")

        with gr.Row():
            with gr.Column():
                stats_table = gr.Dataframe(
                    label="📋 Detaylı İstatistikler",
                    headers=["Kategori", "Sayı", "Yüzde"],
                    datatype=["str", "number", "number"],
                    interactive=False
                )
            with gr.Column():
                current_stats = gr.JSON(label="📈 Anlık Durum")

        # Grafikler
        gr.Markdown("## 📈 Grafiksel Analiz")

        with gr.Row():
            pie_chart = gr.Plot(label="🥧 Cinsiyet Dağılımı (Pasta)")
            bar_chart = gr.Plot(label="📊 Cinsiyet Bazında Sayılar (Bar)")

        with gr.Row():
            time_chart = gr.Plot(label="⏱️ Zaman İçinde Değişim")

        # Özet istatistikler
        with gr.Row():
            total_display = gr.Number(label="Toplam Kişi", interactive=False)
            male_display = gr.Number(label="Erkek", interactive=False)
            female_display = gr.Number(label="Kadın", interactive=False)
            unknown_display = gr.Number(label="Bilinmeyen", interactive=False)
            fps_display = gr.Number(label="FPS", interactive=False)

        # Event handlers
        init_btn.click(
            initialize_counter,
            outputs=[init_status]
        )

        # Webcam streaming
        webcam_input.stream(
            process_webcam_frame,
            inputs=[webcam_input],
            outputs=[webcam_output, webcam_stats, stats_table, pie_chart, bar_chart, time_chart],
            time_limit=60,
            stream_every=0.1
        )

        # Video processing
        process_video_btn.click(
            process_video_file,
            inputs=[video_input],
            outputs=[video_output, video_stats, stats_table, pie_chart, bar_chart, time_chart]
        )

        # Image processing
        process_image_btn.click(
            process_single_image,
            inputs=[image_input],
            outputs=[image_output, image_stats, stats_table, pie_chart, bar_chart, time_chart]
        )

        # Reset functionality
        reset_btn.click(
            reset_system,
            outputs=[init_status, current_stats, stats_table, pie_chart, bar_chart, time_chart]
        )

        # Stats update
        stats_btn.click(
            get_current_stats,
            outputs=[current_stats, stats_table, pie_chart, bar_chart, time_chart]
        )

        # İstatistikleri güncelle
        def update_stats():
            if counter is not None:
                stats = counter.get_stats()
                return (
                    stats.get("Toplam Kişi", 0),
                    stats.get("Erkek", 0),
                    stats.get("Kadın", 0),
                    stats.get("Bilinmeyen", 0),
                    stats.get("FPS", 0)
                )
            return 0, 0, 0, 0, 0

        # Periyodik güncelleme için timer
        timer = gr.Timer(value=2)  # Her 2 saniyede güncelle
        timer.tick(
            update_stats,
            outputs=[total_display, male_display, female_display, unknown_display, fps_display]
        )

        # Kullanım talimatları
        gr.Markdown("""
        ## 📝 Kullanım Talimatları:
        1. **Sistemi Başlat** butonuna tıklayın
        2. **Webcam** sekmesinde gerçek zamanlı sayım yapın
        3. **Demo Videolar** sekmesinde hazır videoları test edin
        4. **Video Yükle** sekmesinde kendi video dosyalarınızı analiz edin  
        5. **Görüntü Yükle** sekmesinde tek görüntü analizi yapın
        6. **İstatistikleri Güncelle** butonu ile tabloları ve grafikleri yenileyin
        7. **Sıfırla** butonu ile sayaçları sıfırlayın
        
        
        ## 🔧 Teknik Notlar:
        - Demo videolar için `demo_videos/` klasörü oluşturun
        - Model dosyalarının doğru konumda olduğundan emin olun
        - Webcam izni vermeyi unutmayın
        - Büyük videolar için işlem süresi artabilir
        """)

    return demo


if __name__ == "__main__":

    # Gradio arayüzünü başlat
    demo = create_interface()
    demo.queue()  # Çoklu kullanıcı desteği
    demo.launch(
        share=True,  # Genel erişim için link oluştur
        inbrowser=True,  # Otomatik tarayıcı açma
        server_name="0.0.0.0",  # Tüm IP'lerden erişim
        server_port=7860  # Port
    )