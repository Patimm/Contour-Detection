# pip install opencv-python numpy pillow tkinter
# Python 3.10


"""
Qualitätskontrolle aktueller Satur/ Ergänzungen


1. Kamera-Setup & Anti-Flicker
   - Manuelle Belichtungszeit, Auto-Exposure, Gain, Weißabgleich deaktiviert
   - Backlight-Kompensation und Gamma-Korrektur gesetzt
   - Power-Line-Frequenz (50 Hz) konfiguriert
   - Exposure-Slider & Reset-Button in der GUI

2. ROI-Erkennung nur innerhalb der Lichtquelle
   - Mehrfache Versuche mit Threshold=230 
   - Mindestflächen-Check (10 % des Bilds), ROI-Expansion, Debug-Output
   - Fallback auf zentrierten Bereich

3. Flacker-Filter
   - Exponential Moving Average (EMA) Filter in drei Modi (Hintergrund, Live, Kalibrierung)
   - Glättet Bildrauschen 
   - GUI-Toggle zum Ein-/Ausschalten und Live-Anpassung des Glättungsfaktors

4. Beleuchtungskorrektur & Kontrast
   - Optional homomorphes Filtern (Frequenzraum-HPF)
   - CLAHE (lokales Histogramm-Equalizing) anstelle globaler EqualizeHist
   - GUI-Toggle für CLAHE

5. Robuste Konturerkennung
   - Kombination aus Differenzbild + Gradientenanalyse
   - Multi-Scale Canny, adaptive Otsu-Thresholding, Morphologie mit Ellipsen-Kernen
   - Fallback auf Standardpipeline

6. Shape-Matching
   - Weighted Score aus Hu-Momenten, Flächen-/Umfangs-Ratios, Kompaktheit
   - ToDo: Echte Bildregistrierung & pixelgenaue Differenz (register_and_diff)

7. Interaktive GUI-Erweiterungen
   - Toggle-Buttons für alle Filter-Pipelines
   - Debug-Button zum Speichern von Zwischenergebnissen
   - Live-Annotation des ROI und Status-Text im Video-Frame

TODO:
- Umsetzung von Sebastians Vorschlägen: Ersatz für Hu-Moment-Matching durch echte Bildregistrierung & Differenzmessung

1. Kontur-Analyse und Ausrichtung
   - Berechne Zentroid und Hauptachse (z.B. via cv2.moments + cv2.minAreaRect)
   - Drehe das Testbild so, dass seine Hauptachse mit der Referenzkontur übereinstimmt.

2. Translation (Map-Shift)
   - Verschiebe das gedrehte Testbild, sodass die Zentroiden von Referenz und Testdeckung übereinanderliegen

3. Binarisierung und Maskenerstellung
   - Erzeuge binäre Masken von Referenz- und Testbild (z.B. Otsu oder adaptive Threshold)
   - Entferne kleine Artefakte am Bildrand (regionprops-Filterung)

4. Affine vs. einfache Transformation
   - Verzicht auf perspektivische Verzerrung, nutze reine Rotation+Translation (cv2.warpAffine)
   - Bevorzuge nearest-neighbor Interpolation, um binäre Masken beizubehalten

5. Pixelweise Differenzberechnung
   - Berechne cv2.absdiff zwischen referenziertem Test-Mask und Referenz-Mask
   - Zähle alle abweichenden Pixel (cv2.countNonZero) → Fehlerpixel

6. Physikalische Parametrisierung (optional)
   - Wenn Kalibrierung vorhanden, skaliere Pixel → mm (Fehlerfläche in mm²)

7. Erweiterungen (optional)
   - Regionprops für jede zusammenhängende Fehlerregion (Fläche, Länge, Lage)
   - Report der Fehlermetriken und schwellwertbasierte OK/NOK-Entscheidung

Aktuelles Problem:
	1. Teilweise Abbruch/Kamerafreezing nach Kalibrierung
	2. Rotation/Translation der Objekte wird noch nicht richtig erkannt
	3. Vergleich zwischen Referenzobjekten und Vergleichsobjekt misslingt. Es wird nicht richtig erkannt, dass das Objekt nun andere Kanten hat.
	4. Leichte Probleme mit Schatten

    
Quality Control – Current Status / Planned Extensions

1. Camera Setup & Anti-Flicker
   - Manual exposure time, auto-exposure, gain, and white balance disabled
   - Backlight compensation and gamma correction configured
   - Power-line frequency set to 50 Hz
   - Exposure slider & reset button integrated into the GUI

2. ROI Detection within Light Source Only
   - Multiple attempts with threshold = 230
   - Minimum area check (10% of the image), ROI expansion, debug output
   - Fallback to centered region if detection fails

3. Flicker Filter
   - Exponential Moving Average (EMA) filter in three modes (background, live, calibration)
   - Smooths image noise
   - GUI toggle to enable/disable and adjust smoothing factor in real time

4. Illumination Correction & Contrast Enhancement
   - Optional homomorphic filtering (high-pass in frequency domain)
   - CLAHE (local histogram equalization) instead of global EqualizeHist
   - GUI toggle for CLAHE

5. Robust Contour Detection
   - Combines difference image + gradient analysis
   - Multi-scale Canny, adaptive Otsu thresholding, morphology with elliptical kernels
   - Fallback to standard pipeline if detection fails

6. Shape Matching
   - Weighted score based on Hu moments, area/perimeter ratios, compactness
   - ToDo: True image registration & pixel-wise difference (register_and_diff)

7. Interactive GUI Enhancements
   - Toggle buttons for all filter pipelines
   - Debug button to save intermediate results
   - Live ROI annotation and status text in video frame

TODO:
- Implement Sebastian’s suggestions: replace Hu moment matching with true image registration & difference computation

1. Contour Analysis and Alignment
   - Compute centroid and major axis (e.g., via cv2.moments + cv2.minAreaRect)
   - Rotate the test image to align its major axis with the reference contour

2. Translation (Map Shift)
   - Shift the rotated test image so that its centroid aligns with the reference centroid

3. Binarization and Mask Creation
   - Generate binary masks from reference and test images (e.g., Otsu or adaptive threshold)
   - Remove small edge artifacts (e.g., using regionprops filtering)

4. Affine vs. Simple Transformation
   - Avoid perspective distortion; use only rotation + translation (cv2.warpAffine)
   - Prefer nearest-neighbor interpolation to preserve binary masks

5. Pixel-Wise Difference Calculation
   - Use cv2.absdiff between aligned test and reference masks
   - Count all differing pixels (cv2.countNonZero) → error pixel count

6. Physical Parameterization (optional)
   - If calibration is available, convert pixels to mm (error area in mm²)

7. Extensions (optional)
   - Use regionprops for each contiguous error region (area, length, location)
   - Report error metrics and derive OK/NOK decision via thresholding

Current Issues:
    1. Occasional crash or camera freeze after calibration
    2. Object rotation/translation not yet correctly detected
    3. Comparison between reference and test objects fails — new edge structures go undetected
    4. Minor issues with shadows
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import tkinter as tk
from datetime import datetime

# === Debug-Ordner-Verwaltung ===
current_debug_folder = None

def create_debug_folder():
    """
    Erstellt einen neuen Debug-Ordner mit Zeitstempel für die aktuelle Debug-Session
    Returns: Pfad zum erstellten Debug-Ordner
    """
    global current_debug_folder
    
    # Erstelle Basis-Debug-Ordner falls nicht vorhanden
    base_debug_dir = "Debug_Sessions"
    if not os.path.exists(base_debug_dir):
        os.makedirs(base_debug_dir)
    
    # Zeitstempel für aktuellen Debug-Ordner
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_folder_name = f"Debug_{timestamp}"
    current_debug_folder = os.path.join(base_debug_dir, debug_folder_name)
    
    # Erstelle Debug-Ordner
    if not os.path.exists(current_debug_folder):
        os.makedirs(current_debug_folder)
        print(f"Debug-Ordner erstellt: {current_debug_folder}")
    
    return current_debug_folder

def save_debug_image(image, filename, subfolder=None):
    """
    Speichert ein Debug-Bild im aktuellen Debug-Ordner
    
    Args:
        image: Das zu speichernde Bild (numpy array oder OpenCV Mat)
        filename: Dateiname (ohne Pfad)
        subfolder: Optionaler Unterordner im Debug-Ordner
    """
    global current_debug_folder
    
    # Erstelle Debug-Ordner falls noch nicht vorhanden
    if current_debug_folder is None:
        create_debug_folder()
    
    # Bestimme Zielpfad
    if subfolder:
        subfolder_path = os.path.join(current_debug_folder, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        file_path = os.path.join(subfolder_path, filename)
    else:
        file_path = os.path.join(current_debug_folder, filename)
    
    # Speichere Bild
    try:
        cv2.imwrite(file_path, image)
        print(f"Debug-Bild gespeichert: {file_path}")
        return file_path
    except Exception as e:
        print(f"Fehler beim Speichern von {filename}: {e}")
        return None







# === Homomorphic Filter für Beleuchtungskorrektur ===
def homomorphic_filter(gray, sigma=30):
    """
    Homomorphes Filtern trennt Beleuchtung (niedrige Frequenzen) von Reflexion (hohe Frequenzen). Ist in dem Case ber irrelevant, da die Beleuchtung konstant ist.
    """
    # 1) Log-Transform
    img_log = np.log1p(np.array(gray, dtype="float") / 255.0)
    # 2) Fourier-Transform
    dft = cv2.dft(img_log, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # 3) High-Pass-Filter im Frequenzraum
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols, 2), np.float32)
    r = sigma
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= r*r
    mask[mask_area] = 0    # Sperre tiefe Frequenzen
    # 4) Anwenden und zurücktransformieren
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    # 5) Exponentielle Rücktransformation und Normierung
    img_hom = np.expm1(img_back)
    img_hom = cv2.normalize(img_hom, None, 0, 255, cv2.NORM_MINMAX)
    return img_hom.astype(np.uint8)

### Erkennt Region of Interest

# === 1. Kamera öffnen ===
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # Fallback auf erste Kamera
    if not cap.isOpened():
        raise Exception("Keine Webcam gefunden.")

# Auflösung einstellen (640×360)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# === Kamera-Einstellungen gegen Flackern ===
def configure_camera_anti_flicker():
    """Konfiguriert Kamera-Einstellungen um Flackern zu reduzieren"""
    print("Konfiguriere Kamera gegen Flackern...")
    
    # Manuelle Belichtungszeit (wichtigster Parameter gegen Flackern)
    # Deaktiviere Auto-Belichtung
    if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25):  # 0.25 = manueller Modus
        print(" Auto-Belichtung deaktiviert")
        # Setze feste Belichtungszeit (Wert ist kamera-abhängig, typisch -1 bis -13)
        if cap.set(cv2.CAP_PROP_EXPOSURE, -6):  # Experimenteller Startwert
            print("Manuelle Belichtungszeit gesetzt")
    
    # Deaktiviere Auto-Weißabgleich
    if cap.set(cv2.CAP_PROP_AUTO_WB, 0):
        print("Auto-Weißabgleich deaktiviert")
    
    # Setze feste Verstärkung (Gain)
    if cap.set(cv2.CAP_PROP_GAIN, 0):
        print("Verstärkung auf 0 gesetzt")
    
    # Deaktiviere Backlight-Kompensation
    if cap.set(cv2.CAP_PROP_BACKLIGHT, 0):
        print("Backlight-Kompensation deaktiviert")
    
    # Setze Gamma-Korrektur
    if cap.set(cv2.CAP_PROP_GAMMA, 100):
        print("Gamma-Korrektur gesetzt")
    
    # Power Line Frequency (50Hz Europa, 60Hz USA) - wichtig gegen Netzbrummen!
    if cap.set(cv2.CAP_PROP_SETTINGS, 1):  # Öffnet Kamera-Einstellungen
        print("Kamera-Einstellungen verfügbar")
    
    # Zeige aktuelle Einstellungen
    print(f"Aktuelle Belichtung: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"Aktuelle Verstärkung: {cap.get(cv2.CAP_PROP_GAIN)}")
    print(f"Auto-Belichtung: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")

# Kamera konfigurieren
configure_camera_anti_flicker()

print("Kamera erfolgreich geöffnet und konfiguriert.")

# === 1.1 Automatische ROI-Erkennung des weißen Kastens ===
print("Warte auf stabile Kamerabilder für ROI-Erkennung...")

# Warte einige Frames für stabile Kameraeinstellungen
for i in range(10):
    ret, _ = cap.read()
    if not ret:
        raise Exception(f"Kamera liefert kein Bild beim Warmup (Frame {i+1}).")

# Versuche ROI-Erkennung mit mehreren Frames
roi_x, roi_y, roi_w, roi_h = None, None, None, None
best_contour_area = 0

for attempt in range(5):  # 5 Versuche für ROI-Erkennung
    ret, frame0 = cap.read()
    if not ret:
        print(f"ROI-Erkennungsversuch {attempt+1} fehlgeschlagen: Kein Kamerabild.")
        continue
    
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    
    # Threshold für sehr helle Flächen (angepasst)
    _, mask0 = cv2.threshold(gray0, 230, 255, cv2.THRESH_BINARY)  # Niedrigerer Threshold
    
    # Morphologische Nachbearbeitung
    kernel0 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask0 = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, kernel0)
    mask0 = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, kernel0)
    
    # Größte Kontur finden
    contours0, _ = cv2.findContours(mask0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours0:
        cnt0 = max(contours0, key=cv2.contourArea)
        area = cv2.contourArea(cnt0)
        
        # Mindestgröße für ROI (verhindert kleine Störungen)
        min_roi_area = (frame0.shape[0] * frame0.shape[1]) * 0.1  # Mindestens 10% des Bildes
        
        if area > min_roi_area and area > best_contour_area:
            best_contour_area = area
            roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(cnt0)
            print(f"ROI-Kandidat gefunden (Versuch {attempt+1}): x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}, Fläche={area:.0f}")

# ROI-Ergebnis auswerten
if roi_x is not None:
    print(f"ROI erfolgreich erkannt: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
    
    # ROI-Validierung: Größe und Position prüfen
    h0, w0 = frame0.shape[:2]
    if roi_w < w0 * 0.2 or roi_h < h0 * 0.2:  # ROI zu klein
        print("Warnung: ROI ist sehr klein, verwende größeren Bereich")
        # Erweitere ROI um 20%
        expand_x = int(roi_w * 0.2)
        expand_y = int(roi_h * 0.2)
        roi_x = max(0, roi_x - expand_x)
        roi_y = max(0, roi_y - expand_y)
        roi_w = min(w0 - roi_x, roi_w + 2 * expand_x)
        roi_h = min(h0 - roi_y, roi_h + 2 * expand_y)
        print(f"Erweiterte ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
else:
    # Fallback: Verwende mittleren Bereich als ROI
    h0, w0 = frame0.shape[:2]
    roi_w, roi_h = int(w0 * 0.6), int(h0 * 0.6)  # 60% der Bildgröße
    roi_x, roi_y = int(w0 * 0.2), int(h0 * 0.2)  # Zentriert
    print(f"Keine ROI erkannt, verwende zentrierten Bereich: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

# Debug: Speichere ROI-Erkennungsbild
if 'frame0' in locals():
    debug_frame = frame0.copy()
    cv2.rectangle(debug_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 3)
    save_debug_image(debug_frame, "roi_detection_debug.png", "ROI_Detection")
    print("ROI-Debug-Bild gespeichert in Debug-Ordner")

# === 2. Globale Variablen ===
background_image = None       # Graustufen-Hintergrund (ROI)
background_image_raw = None   # Ungefilterter Hintergrund für homomorphic Filter
reference_image = None        # Graustufen-Referenz des perfekten Teils (ROI)
reference_mask = None         # Maske des Referenzteils (ROI)
reference_contour = None      # Kontur des Referenzteils (ROI)

current_frame = None          # Letztes eingelesenes ROI-Frame (BGR)
previous_frame = None         # Vorheriges ROI-Frame (RGB) für Bewegungserkennung

part_present_state = False
error_state = False

motion_threshold = 15          # Reduziert von 30 → sensibler für kleine Helligkeitsänderungen
motion_pixels_threshold = 2000  # Reduziert von 5000 → weniger Pixel nötig für Bewegungserkennung
motion_frames_stable = 8        # Reduziert von 10 → schnellere Stabilisierung
stable_counter = 0

shape_tolerance = 0.03

# Debug-Option für homomorphic filter
use_homomorphic_filter = False  # Zunächst deaktiviert für Tests

# === CLAHE Filter für lokale Kontrastverbesserung ===
def apply_clahe_filter(gray_image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Verbessert lokalen Kontrast ohne Überkorrektur
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)

use_clahe_filter = True  # CLAHE standardmäßig aktiviert

# === Robuste Konturerkennung für bessere Segmentierung ===
def multi_scale_edge_detection(gray_image):
    """Multi-Scale Kantenerkennung für robustere Konturen"""
    edges_combined = np.zeros_like(gray_image)
    
    # Verschiedene Gauß-Kernel für verschiedene Detailgrade
    scales = [1, 2, 4]
    for scale in scales:
        # Gaussian Blur mit verschiedenen Sigmas
        blurred = cv2.GaussianBlur(gray_image, (0, 0), scale)
        
        # Canny mit adaptiven Schwellwerten
        high_thresh = np.percentile(blurred, 90)
        low_thresh = high_thresh * 0.5
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        # Kombiniere Kanten mit Gewichtung
        weight = 1.0 / scale  # Kleinere Skalen bekommen höheres Gewicht
        edges_combined = cv2.addWeighted(edges_combined, 1.0, edges, weight, 0)
    
    return edges_combined.astype(np.uint8)

def improved_contour_detection(gray_image, background_image):
    """Verbesserte Konturerkennung mit Gradientenanalyse"""
    # 1. Gradientenbasierte Segmentierung
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. Kombiniere Differenzbild und Gradienten
    diff = cv2.absdiff(gray_image, background_image)
    
    # 3. Gewichtete Kombination: 70% Differenz + 30% Gradient
    combined = cv2.addWeighted(diff, 0.7, gradient_magnitude, 0.3, 0)
    
    # 4. Multi-Scale Kantenerkennung
    edges = multi_scale_edge_detection(combined)
    
    # 5. Adaptive Schwellwertbildung auf kombiniertem Bild
    blur = cv2.GaussianBlur(combined, (5, 5), 0)
    _, thresh_adaptive = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 6. Kombiniere Threshold und Edges
    final_mask = cv2.bitwise_or(thresh_adaptive, edges)
    
    # 7. Morphologische Verfeinerung mit angepassten Kerneln
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
    
    return final_mask

def calculate_robust_hu_moments(contour):
    """Robuste Hu-Moment Berechnung mit Preprocessing"""
    # Kontur glätten für stabilere Momente
    epsilon = 0.02 * cv2.arcLength(contour, True)
    smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
    
    # Moments berechnen
    moments = cv2.moments(smoothed_contour)
    if moments['m00'] == 0:  # Verhindere Division durch Null
        return None
    
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log-Transform für bessere Vergleichbarkeit
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return hu_moments

def shape_matching_improved(ref_contour, test_contour, method='multiple'):
    """Verbesserte Formvergleichung mit mehreren Metriken"""
    if method == 'multiple':
        # Kombiniere verschiedene Metriken
        hu_score = cv2.matchShapes(ref_contour, test_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        
        # Zusätzliche geometrische Metriken
        ref_area = cv2.contourArea(ref_contour)
        test_area = cv2.contourArea(test_contour)
        area_ratio = abs(test_area - ref_area) / ref_area if ref_area > 0 else 1.0
        
        ref_perimeter = cv2.arcLength(ref_contour, True)
        test_perimeter = cv2.arcLength(test_contour, True)
        perimeter_ratio = abs(test_perimeter - ref_perimeter) / ref_perimeter if ref_perimeter > 0 else 1.0
        
        # Kompaktheit (Rundheit)
        ref_compact = 4 * np.pi * ref_area / (ref_perimeter ** 2) if ref_perimeter > 0 else 0
        test_compact = 4 * np.pi * test_area / (test_perimeter ** 2) if test_perimeter > 0 else 0
        compact_diff = abs(ref_compact - test_compact)
        
        # Gewichtete Kombination der Metriken
        combined_score = (0.5 * hu_score + 
                         0.2 * area_ratio + 
                         0.2 * perimeter_ratio + 
                         0.1 * compact_diff)
        
        return combined_score
    
    return cv2.matchShapes(ref_contour, test_contour, cv2.CONTOURS_MATCH_I1, 0.0)

use_robust_contours = True  # Robuste Konturerkennung standardmäßig aktiviert

# === Bildregistrierung und geometrische Ausrichtung ===
class GeometricAlignment:
    def __init__(self):
        self.reference_centroid = None
        self.reference_angle = None
        self.reference_major_axis = None
        self.reference_minor_axis = None
        
    def analyze_contour_geometry(self, contour):
        """
        Analysiert geometrische Eigenschaften einer Kontur für Ausrichtung
        Returns: (centroid, angle, major_axis, minor_axis, area)
        """
        # 1. Zentroid berechnen via Momente (präziser als Bounding Box)
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return None, None, None, None, 0
        
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        centroid = (cx, cy)
        
        # 2. Hauptachse via minimaler umschließender Rechteck
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (width, height), angle = rect
        
        # 3. Hauptachse normalisieren (längere Seite = Hauptachse)
        if width > height:
            major_axis = width
            minor_axis = height
            main_angle = angle
        else:
            major_axis = height
            minor_axis = width
            main_angle = angle + 90
        
        # 4. VERBESSERTE Winkel-Normalisierung für symmetrische Objekte
        # Normalisiere auf [0, 180] statt [-90, 90] um 180°-Ambiguität zu handhaben
        while main_angle < 0:
            main_angle += 180
        while main_angle >= 180:
            main_angle -= 180
            
        area = cv2.contourArea(contour)
        
        return centroid, main_angle, major_axis, minor_axis, area
    
    def set_reference_geometry(self, reference_contour):
        """Setzt Referenzgeometrie basierend auf Referenzkontur"""
        result = self.analyze_contour_geometry(reference_contour)
        if result[0] is not None:
            self.reference_centroid, self.reference_angle, self.reference_major_axis, self.reference_minor_axis, ref_area = result
            # WICHTIG: Speichere Referenzkontur für Alignment-Bewertung
            self.reference_contour = reference_contour.copy()
            print(f"Referenzgeometrie gesetzt:")
            print(f"  Zentroid: {self.reference_centroid}")
            print(f"  Hauptachsen-Winkel: {self.reference_angle:.1f}°")
            print(f"  Hauptachse: {self.reference_major_axis:.1f}, Nebenachse: {self.reference_minor_axis:.1f}")
            return True
        return False
    
    def _evaluate_alignment_quality(self, reference_contour, test_contour, rotation_angle):
        """
        Bewertet die Qualität einer Ausrichtung durch Überlappung der Konturen
        Returns: Überlappungs-Score (höher = besser)
        """
        # Erstelle temporäre Masken für Bewertung
        canvas_size = 500  # Feste Größe für Bewertung
        
        # Referenz-Maske
        ref_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        ref_contour_shifted = reference_contour + [canvas_size//2, canvas_size//2]
        cv2.drawContours(ref_mask, [ref_contour_shifted.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
        
        # Test-Maske mit Rotation
        test_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        
        # Rotiere Test-Kontur
        test_centroid = np.mean(test_contour, axis=0)
        rotation_matrix = cv2.getRotationMatrix2D(tuple(test_centroid), rotation_angle, 1.0)
        
        # Wende Rotation auf Kontur an
        test_contour_homogeneous = np.hstack([test_contour, np.ones((test_contour.shape[0], 1))])
        test_contour_rotated = rotation_matrix.dot(test_contour_homogeneous.T).T
        
        # Verschiebe zur Bildmitte
        test_contour_shifted = test_contour_rotated + [canvas_size//2, canvas_size//2]
        cv2.drawContours(test_mask, [test_contour_shifted.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
        
        # Berechne Überlappung
        intersection = cv2.bitwise_and(ref_mask, test_mask)
        union = cv2.bitwise_or(ref_mask, test_mask)
        
        intersection_pixels = cv2.countNonZero(intersection)
        union_pixels = cv2.countNonZero(union)
        
        # IoU (Intersection over Union) Score
        iou_score = intersection_pixels / union_pixels if union_pixels > 0 else 0
        
        return iou_score

    def align_test_image(self, test_image, test_contour):
        """
        Richtet Test-Bild geometrisch zur Referenz aus (Map-Shift)
        Verwendet nur Rotation + Translation, keine affine Verzerrung
        VERBESSERT: Intelligente Rotations-Bewertung für symmetrische Objekte
        """
        if self.reference_centroid is None:
            return test_image, None
        
        # 1. Analysiere Test-Kontur
        result = self.analyze_contour_geometry(test_contour)
        if result[0] is None:
            return test_image, None
        
        test_centroid, test_angle, test_major, test_minor, test_area = result
        
        # 2. Berechne mögliche Rotationen
        angle_diff_primary = self.reference_angle - test_angle
        angle_diff_alt = angle_diff_primary + 180
        
        # Normalisiere beide Winkel
        angle_diff_primary = ((angle_diff_primary + 180) % 360) - 180
        angle_diff_alt = ((angle_diff_alt + 180) % 360) - 180
        
        # 3. INTELLIGENTE ROTATION-BEWERTUNG
        # Bewerte beide Rotationsmöglichkeiten durch Kontur-Überlappung
        if hasattr(self, 'reference_contour') and self.reference_contour is not None:
            score_primary = self._evaluate_alignment_quality(self.reference_contour, test_contour, angle_diff_primary)
            score_alt = self._evaluate_alignment_quality(self.reference_contour, test_contour, angle_diff_alt)
            
            print(f"DEBUG: Rotations-Bewertung:")
            print(f"  Primär ({angle_diff_primary:.1f}°): Score = {score_primary:.3f}")
            print(f"  Alternativ ({angle_diff_alt:.1f}°): Score = {score_alt:.3f}")
            
            # Wähle die Rotation mit dem höheren Score
            if score_alt > score_primary:
                angle_diff = angle_diff_alt
                print(f"DEBUG: Alternative Rotation gewählt (bessere Überlappung)")
            else:
                angle_diff = angle_diff_primary
                print(f"DEBUG: Primäre Rotation gewählt")
        else:
            # Fallback: Wähle kleinere Rotation
            if abs(angle_diff_alt) < abs(angle_diff_primary):
                angle_diff = angle_diff_alt
                print(f"DEBUG: 180°-Symmetrie erkannt, verwende alternative Rotation: {angle_diff:.1f}°")
            else:
                angle_diff = angle_diff_primary
        
        # 4. Map-Shift: Reine Rotation + Translation (keine Skalierung/Verzerrung)
        img_center = (test_image.shape[1] // 2, test_image.shape[0] // 2)
        
        # Rotationsmatrix um Test-Zentroid
        M_rotate = cv2.getRotationMatrix2D(test_centroid, angle_diff, 1.0)  # Skalierung = 1.0
        
        # Translation zu Referenz-Zentroid
        M_rotate[0, 2] += self.reference_centroid[0] - test_centroid[0]
        M_rotate[1, 2] += self.reference_centroid[1] - test_centroid[1]
        
        # 5. Wende Map-Shift an mit Nearest-Neighbor für binäre Präzision
        aligned_image = cv2.warpAffine(
            test_image, 
            M_rotate, 
            (test_image.shape[1], test_image.shape[0]),
            flags=cv2.INTER_NEAREST,  # Für scharfe binäre Kanten
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # 6. Debug-Info
        alignment_info = {
            'angle_correction': angle_diff,
            'centroid_shift': (
                self.reference_centroid[0] - test_centroid[0],
                self.reference_centroid[1] - test_centroid[1]
            ),
            'test_geometry': {
                'centroid': test_centroid,
                'angle': test_angle,
                'area': test_area
            },
            'rotation_scores': {
                'primary': score_primary if 'score_primary' in locals() else None,
                'alternative': score_alt if 'score_alt' in locals() else None
            }
        }
        
        print(f"Geometrische Ausrichtung:")
        print(f"  Test-Winkel: {test_angle:.1f}°, Referenz-Winkel: {self.reference_angle:.1f}°")
        print(f"  Gewählte Winkelkorrektur: {angle_diff:.1f}°")
        print(f"  Zentroid-Verschiebung: {alignment_info['centroid_shift']}")
        
        return aligned_image, alignment_info
    
    def create_binary_mask_with_regionprops(self, gray_image, threshold_method='otsu'):
        """
        Erstellt binäre Maske mit Regionprops-Filterung
        Entfernt kleine Artefakte am Bildrand
        """
        # 1. Binarisierung
        if threshold_method == 'otsu':
            _, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_method == 'adaptive':
            binary_mask = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:  # fixed threshold
            _, binary_mask = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        
        # 2. Regionprops-Filterung
        binary_mask = self._filter_mask_with_regionprops(binary_mask)
        
        return binary_mask
    
    def _filter_mask_with_regionprops(self, binary_mask):
        """Filtert Maske mit Regionprops um kleine Artefakte zu entfernen"""
        # 1. Finde alle zusammenhängenden Komponenten
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # 2. Analysiere Komponenten (skip background label 0)
        filtered_mask = np.zeros_like(binary_mask)
        
        # Berechne Mindestgröße basierend auf Bildgröße
        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        min_component_size = max(20, int(total_pixels * 0.0005))  # Mindestens 0.05% des Bildes
        
        for i in range(1, num_labels):  # Skip background (label 0)
            # Statistiken der aktuellen Komponente
            area = stats[i, cv2.CC_STAT_AREA]
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Filter-Kriterien (wie in Sebastians Kommentar: "Regionen am Rand entfernen")
            # 1. Mindestgröße
            if area < min_component_size:
                continue
            
            # 2. Bildrand-Filter (Artefakt-Filter)
            margin = 3
            if (left < margin or top < margin or 
                left + width > binary_mask.shape[1] - margin or
                top + height > binary_mask.shape[0] - margin):
                # Nur sehr kleine Objekte am Rand filtern
                if area < min_component_size * 3:
                    continue
            
            # 3. Aspect Ratio Check (verhindert sehr schmale Streifen)
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            if aspect_ratio > 15:  # Zu schmale Objekte filtern
                continue
            
            # Komponente ist valide, füge sie zur gefilterten Maske hinzu
            filtered_mask[labels == i] = 255
        
        # Finale morphologische Bereinigung
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
        
        return filtered_mask
    
    def calculate_pixel_difference(self, reference_mask, aligned_test_mask):
        """
        Berechnet pixelweise Differenz zwischen ausgerichteten Masken
        Echte Shapedifferenz-Berechnung wie in Sebastians Kommentar beschrieben
        """
        # XOR für symmetrische Differenz (fehlendes + zusätzliches Material)
        diff_symmetric = cv2.bitwise_xor(reference_mask, aligned_test_mask)
        
        # Separate Analyse: fehlendes vs. zusätzliches Material
        missing_material = cv2.bitwise_and(reference_mask, cv2.bitwise_not(aligned_test_mask))
        extra_material = cv2.bitwise_and(aligned_test_mask, cv2.bitwise_not(reference_mask))
        
        # Pixel zählen (wie beschrieben: "Pixel zählen Bild 1 - Bild 2")
        total_diff_pixels = cv2.countNonZero(diff_symmetric)
        missing_pixels = cv2.countNonZero(missing_material)
        extra_pixels = cv2.countNonZero(extra_material)
        
        # Referenzfläche für Verhältnis
        reference_pixels = cv2.countNonZero(reference_mask)
        diff_ratio = total_diff_pixels / reference_pixels if reference_pixels > 0 else 0
        
        # Farbkodierte Differenz für Visualisierung
        diff_colored = np.zeros((diff_symmetric.shape[0], diff_symmetric.shape[1], 3), dtype=np.uint8)
        diff_colored[missing_material > 0] = [0, 0, 255]    # Rot: fehlendes Material
        diff_colored[extra_material > 0] = [255, 0, 0]      # Blau: zusätzliches Material
        
        result = {
            'total_difference_pixels': total_diff_pixels,
            'missing_pixels': missing_pixels,
            'extra_pixels': extra_pixels,
            'difference_ratio': diff_ratio,
            'difference_image': diff_colored,
            'binary_difference': diff_symmetric
        }
        
        return result

# === Einheitliche Konturerkennung für konsistente Ergebnisse ===
def consistent_contour_detection(gray_image, background_image, method="standard", threshold_type="otsu"):
    """
    Einheitliche Konturerkennung für Kalibrierung UND Analyse
    
    
    Args:
        threshold_type: "otsu" | "adaptive" | "fixed"
    """
    if method == "robust" and use_robust_contours:
        # Robuste Methode: Gradientenanalyse + Multi-Scale
        return improved_contour_detection(gray_image, background_image)
    else:
        # Standard-Methode: Background-Subtraktion + Threshold
        diff = cv2.absdiff(gray_image, background_image)
        blur = cv2.GaussianBlur(diff, (7, 7), 0)  # Einheitliche Blur-Parameter
        
        # Threshold-Methode wählen
        if threshold_type == "adaptive":
            # Adaptive Threshold für lokale Anpassung
            fg_mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        elif threshold_type == "fixed":
            # Fester Threshold
            _, fg_mask = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
        else:  # otsu (default)
            # OTSU für automatische Schwellwert-Bestimmung
            _, fg_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # EINHEITLICHE morphologische Operationen
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # Kompromiss zwischen 5 und 11
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.convertScaleAbs(fg_mask)
        
        return fg_mask

def extract_main_contour(binary_mask, min_area=100):
    """
    Extrahiert Hauptkontur aus binärer Maske
    Einheitliche Logik für Kalibrierung und Analyse
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0
    
    # Größte Kontur als Hauptkontur
    main_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    
    # Mindestgröße-Check
    if area < min_area:
        return None, area
    
    return main_contour, area

# Globaler Geometric Alignment Analyzer
geometric_aligner = GeometricAlignment()
use_geometric_alignment = True  # Aktiviere neue Bildregistrierung

# Threshold-Methode für Bildregistrierung
threshold_method_for_alignment = "adaptive"  # "otsu" | "adaptive" | "fixed"

# === ROI-Größen-Presets für verschiedene Setups ===
ROI_PRESETS = {
    "60x60 cm": (600, 600),    # Großer weißer Kasten
    "36x27 cm": (360, 270),    # Mittlerer weißer Kasten  
    "30x30 cm": (300, 300),    # Kleiner weißer Kasten
    "Benutzerdefiniert": (150, 120)  # Fallback/Custom
}

# Aktuelle ROI-Größe
current_roi_preset = "36x27 cm"  # Standard-Auswahl

# === Perspektiv-Korrektur für echte Draufsicht ===
class PerspectiveCorrection:
    def __init__(self):
        self.homography_matrix = None
        self.roi_real_size_mm = None  # Reale Größe des ROI in mm
        self.pixels_per_mm = None
        self.is_calibrated = False
        
    def detect_roi_corners(self, frame):
        """
        Erkennt die 4 Eckpunkte des ROI (weißer Kasten) für Perspektiv-Korrektur
        Returns: 4 Eckpunkte in der Reihenfolge: oben-links, oben-rechts, unten-rechts, unten-links
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold für sehr helle Flächen (wie bei ROI-Erkennung)
        _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        
        # Morphologische Operationen
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Finde Konturen
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Größte Kontur (ROI)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximiere zu Rechteck
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Brauchen 4 Eckpunkte für Rechteck
        if len(approx) != 4:
            # Fallback: Bounding Rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            approx = np.int0(box)
        
        # Sortiere Punkte: oben-links, oben-rechts, unten-rechts, unten-links
        corners = self._sort_corners(approx.reshape(-1, 2))
        
        return corners
    
    def _sort_corners(self, points):
        """Sortiert 4 Punkte in die richtige Reihenfolge für Homographie"""
        # Sortiere nach y-Koordinate
        sorted_points = points[np.argsort(points[:, 1])]
        
        # Top 2 und Bottom 2 trennen
        top_points = sorted_points[:2]
        bottom_points = sorted_points[2:]
        
        # Innerhalb von top/bottom nach x sortieren
        top_left, top_right = top_points[np.argsort(top_points[:, 0])]
        bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]
        
        return np.float32([top_left, top_right, bottom_right, bottom_left])
    
    def calibrate_perspective(self, frame, roi_width_mm=100, roi_height_mm=80):
        """
        Kalibriert Perspektiv-Korrektur basierend auf ROI-Eckpunkten
        
        Args:
            roi_width_mm: Reale Breite des ROI in mm
            roi_height_mm: Reale Höhe des ROI in mm
        """
        # Erkenne ROI-Eckpunkte
        roi_corners = self.detect_roi_corners(frame)
        
        if roi_corners is None:
            print("FEHLER: ROI-Eckpunkte konnten nicht erkannt werden!")
            return False
        
        # Definiere Ziel-Rechteck (perfekte Draufsicht)
        # Verwende gleiche Pixel-Größe wie ROI
        target_width = int(roi_width_mm * 5)   # 5 Pixel pro mm
        target_height = int(roi_height_mm * 5) # 5 Pixel pro mm
        
        target_corners = np.float32([
            [0, 0],                              # oben-links
            [target_width, 0],                   # oben-rechts  
            [target_width, target_height],       # unten-rechts
            [0, target_height]                   # unten-links
        ])
        
        # Berechne Homographie-Matrix
        self.homography_matrix = cv2.getPerspectiveTransform(roi_corners, target_corners)
        
        # Speichere Kalibrierungsdaten
        self.roi_real_size_mm = (roi_width_mm, roi_height_mm)
        self.pixels_per_mm = target_width / roi_width_mm
        self.is_calibrated = True
        
        print(f"Perspektiv-Kalibrierung erfolgreich!")
        print(f"ROI: {roi_width_mm}x{roi_height_mm} mm → {target_width}x{target_height} px")
        print(f"Auflösung: {self.pixels_per_mm:.1f} Pixel/mm")
        
        # Debug: Speichere Kalibrierungs-Bilder
        debug_frame = frame.copy()
        for i, corner in enumerate(roi_corners):
            cv2.circle(debug_frame, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
            cv2.putText(debug_frame, str(i), tuple(corner.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        save_debug_image(debug_frame, "perspective_calibration_corners.png", "Perspective_Calibration")
        
        return True
    
    def correct_perspective(self, image):
        """
        Wendet Perspektiv-Korrektur auf ein Bild an
        Returns: Korrigiertes Bild in perfekter Draufsicht
        """
        if not self.is_calibrated:
            print("WARNUNG: Perspektiv-Korrektur nicht kalibriert!")
            return image
        
        # Wende Homographie an
        target_width = int(self.roi_real_size_mm[0] * self.pixels_per_mm)
        target_height = int(self.roi_real_size_mm[1] * self.pixels_per_mm)
        
        corrected = cv2.warpPerspective(
            image, 
            self.homography_matrix, 
            (target_width, target_height),
            flags=cv2.INTER_LINEAR
        )
        
        return corrected
    
    def pixel_to_mm(self, pixel_value):
        """Konvertiert Pixel zu mm basierend auf Kalibrierung"""
        if not self.is_calibrated:
            return pixel_value
        return pixel_value / self.pixels_per_mm
    
    def mm_to_pixel(self, mm_value):
        """Konvertiert mm zu Pixel basierend auf Kalibrierung"""
        if not self.is_calibrated:
            return mm_value
        return int(mm_value * self.pixels_per_mm)

# Globale Perspektiv-Korrektur Instanz
perspective_corrector = PerspectiveCorrection()
use_perspective_correction = False  # Standardmäßig deaktiviert

# === Temporal Filter gegen Flackern (Optimiert mit EMA) ===
class TemporalFilterEMA:
    def __init__(self, alpha=0.2):
        """
        Exponential Moving Average Filter gegen Kameraflackern
        alpha: Glättungsfaktor (0 < alpha < 1)
               - Niedrig (0.1-0.3): Starke Glättung, langsame Reaktion
               - Hoch (0.5-0.8): Schwache Glättung, schnelle Reaktion
        """
        self.alpha = alpha
        self.prev_smoothed = None
        self.frame_count = 0
    
    def apply_filter(self, gray_frame):
        """Wendet EMA-Glättung gegen Flackern an"""
        gray_f = gray_frame.astype(np.float32)
        
        if self.prev_smoothed is None:
            # Erster Frame: initialisiere mit aktuellem Frame
            self.prev_smoothed = gray_f
            self.frame_count = 1
            return gray_frame
        
        # EMA: neues = α·aktuelles + (1–α)·altes
        self.prev_smoothed = self.alpha * gray_f + (1 - self.alpha) * self.prev_smoothed
        self.frame_count += 1
        
        # Zurück auf uint8
        return self.prev_smoothed.astype(np.uint8)
    
    def reset(self):
        """Reset des Filters"""
        self.prev_smoothed = None
        self.frame_count = 0
    
    def set_alpha(self, alpha):
        """Ändert Glättungsfaktor zur Laufzeit"""
        self.alpha = max(0.05, min(0.95, alpha))  # Begrenze auf sinnvolle Werte

# Globale EMA Filter Instanzen (optimiert für verschiedene Anwendungen)
temporal_filter_bg = TemporalFilterEMA(alpha=0.15)     # Hintergrund: starke Glättung
temporal_filter_live = TemporalFilterEMA(alpha=0.25)   # Live: moderate Glättung  
temporal_filter_calib = TemporalFilterEMA(alpha=0.2)   # Kalibrierung: ausgewogen
use_temporal_filter = True  # Temporal Filter standardmäßig aktiviert

# === 3. Tkinter-GUI einrichten mit Tab-System ===
import tkinter.ttk as ttk  # Import für Notebook (Tabs)

root = tk.Tk()
root.title("Qualitätskontrolle – Automatisiert")

# Erstelle Tab-System
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True, padx=10, pady=10)

# === HAUPT-TAB ===
main_tab = ttk.Frame(notebook)
notebook.add(main_tab, text="Hauptansicht")

# Video-Anzeige im Haupt-Tab
video_label = tk.Label(main_tab)
video_label.pack(pady=10)

status_label = tk.Label(main_tab, text="Bitte zuerst Hintergrund erfassen.", font=("Arial", 16))
status_label.pack(pady=5)

def update_status(text, ok=True):
    if ok:
        status_label.config(text=text, fg="green")
    else:
        status_label.config(text=text, fg="red")

# Bedien-Buttons im Haupt-Tab
control_frame = tk.Frame(main_tab)
control_frame.pack(pady=20)




# Debug-Buttons im Haupt-Tab
debug_frame = tk.Frame(main_tab)
debug_frame.pack(pady=10)

def create_new_debug_session():
    """Erstellt manuell einen neuen Debug-Ordner"""
    folder = create_debug_folder()
    update_status(f"Neuer Debug-Ordner erstellt: {os.path.basename(folder)}", ok=True)

def show_debug_info():
    """Zeigt Informationen über den aktuellen Debug-Ordner"""
    if current_debug_folder:
        folder_name = os.path.basename(current_debug_folder)
        num_files = len([f for f in os.listdir(current_debug_folder) if f.endswith('.png')]) if os.path.exists(current_debug_folder) else 0
        update_status(f"Debug-Ordner: {folder_name} ({num_files} Bilder)", ok=True)
    else:
        update_status("Kein Debug-Ordner aktiv", ok=False)

def manual_debug_analysis():
    """Führt eine manuelle Debug-Analyse durch"""
    if current_frame is not None:
        create_debug_folder()  # Neuer Ordner für diese Debug-Session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Aktuelles Bild speichern
        save_debug_image(current_frame, f"current_frame_{timestamp}.png", "Manual_Debug")
        
        # Falls Hintergrund verfügbar
        if background_image is not None:
            save_debug_image(background_image, f"background_{timestamp}.png", "Manual_Debug")
        
        # Falls Referenz verfügbar
        if reference_image is not None:
            save_debug_image(reference_image, f"reference_{timestamp}.png", "Manual_Debug")
        
        update_status("Manuelle Debug-Bilder gespeichert", ok=True)
    else:
        update_status("Kein aktuelles Bild für Debug verfügbar", ok=False)

btn_new_debug = tk.Button(debug_frame, text="Neuer Debug-Ordner", command=create_new_debug_session, font=("Arial", 10))
btn_new_debug.pack(side=tk.LEFT, padx=5)

btn_debug_info = tk.Button(debug_frame, text="Debug-Info", command=show_debug_info, font=("Arial", 10))
btn_debug_info.pack(side=tk.LEFT, padx=5)

btn_manual_debug = tk.Button(debug_frame, text="Manuelle Debug-Analyse", command=manual_debug_analysis, font=("Arial", 10))
btn_manual_debug.pack(side=tk.LEFT, padx=5)

#btn_debug_detailed = tk.Button(debug_frame, text="Detaillierte Masken-Analyse", command=debug_masks_detailed, font=("Arial", 10))
#btn_debug_detailed.pack(side=tk.LEFT, padx=5)

# === EINSTELLUNGEN-TAB ===
settings_tab = ttk.Frame(notebook)
notebook.add(settings_tab, text="Einstellungen")

# Scrollbarer Bereich für Einstellungen
canvas = tk.Canvas(settings_tab)
scrollbar = ttk.Scrollbar(settings_tab, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# === 4. Hintergrund erfassen (nur ROI) ===
def capture_background():
    global background_image, background_image_raw
    ret, frame = cap.read()
    if not ret:
        update_status("Fehler: Kein Kamerabild.", ok=False)
        return
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Temporal Filter anwenden gegen Flackern
    if use_temporal_filter:
        gray = temporal_filter_bg.apply_filter(gray)
    
    # Ungefiltertes Bild für spätere homomorphic Verarbeitung speichern
    background_image_raw = gray.copy()
    
    # Homomorphic Filter für bessere Beleuchtungskorrektur (optional)
    if use_homomorphic_filter:
        gray = homomorphic_filter(gray)
    
    # CLAHE für lokale Kontrastverbesserung (empfohlen für ungleichmäßige Beleuchtung)
    if use_clahe_filter:
        gray = apply_clahe_filter(gray)
    else:
        gray = cv2.equalizeHist(gray)  # Fallback auf normales Histogram Equalization
    
    background_image = gray.copy()
    update_status("Hintergrund erfasst. Lege Referenzteil ein und klicke Kalibrieren.", ok=True)
    print("Hintergrund gespeichert.")

btn_capture_bg = tk.Button(control_frame, text="Hintergrund erfassen", command=capture_background, font=("Arial", 14), width=20)
btn_capture_bg.pack(pady=5)

# === 5. Kalibrierung: Referenzkontur im ROI ermitteln ===
def calibrate():
    global reference_image, reference_mask, reference_contour, part_present_state, error_state
    if background_image is None:
        update_status("Bitte zuerst den Hintergrund erfassen.", ok=False)
        return
    ret, frame = cap.read()
    if not ret:
        update_status("Fehler: Kein Kamerabild.", ok=False)
        return
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Temporal Filter gegen Flackern
    if use_temporal_filter:
        gray = temporal_filter_live.apply_filter(gray)
    
    # Konsistente Homomorphic-Filter-Anwendung
    if use_homomorphic_filter:
        # Beide Bilder mit identischen Parametern filtern
        gray = homomorphic_filter(gray)
        # Hintergrund neu filtern für konsistente Verarbeitung
        background_filtered = homomorphic_filter(background_image_raw)
        if use_clahe_filter:
            background_for_diff = apply_clahe_filter(background_filtered)
        else:
            background_for_diff = cv2.equalizeHist(background_filtered)
    else:
        background_for_diff = background_image
    
    # CLAHE für lokale Kontrastverbesserung
    if use_clahe_filter:
        gray = apply_clahe_filter(gray)
    else:
        gray = cv2.equalizeHist(gray)

    # === EINHEITLICHE Konturerkennung ===
    fg_mask = consistent_contour_detection(
        gray, 
        background_for_diff, 
        method="robust" if use_robust_contours else "standard",
        threshold_type=threshold_method_for_alignment
    )

    # === EINHEITLICHE Konturextraktion ===
    reference_contour, area_ref = extract_main_contour(fg_mask, min_area=100)
    
    print(f"Debug: Kalibrierung - Konturfläche: {area_ref:.0f}")
    
    if reference_contour is None:
        update_status("Keine Kontur im ROI erkannt! Referenz neu positionieren.", ok=False)
        reference_contour = None
        reference_mask = None
        reference_image = None
        return
    
    if area_ref < 100:
        update_status("Referenzteil zu klein oder verrauscht!", ok=False)
        reference_contour = None
        reference_mask = None
        reference_image = None
        return

    reference_mask = np.zeros_like(gray)
    cv2.drawContours(reference_mask, [reference_contour], -1, 255, thickness=cv2.FILLED)
    reference_image = cv2.bitwise_and(gray, gray, mask=reference_mask)

    # === Geometrische Referenzanalyse ===
    if use_geometric_alignment:
        success = geometric_aligner.set_reference_geometry(reference_contour)
        if success:
            print("Referenzgeometrie für Alignment gesetzt")
        else:
            print("Warnung: Referenzgeometrie konnte nicht analysiert werden")

    part_present_state = False
    error_state = False
    update_status("Referenzkontur gespeichert. Entferne jetzt das Referenzteil.", ok=True)
    print(f"Kalibrierung abgeschlossen (Konturfläche: {area_ref:.0f} Pixel).")

# Kalibrierungs-Button
btn_calibrate = tk.Button(control_frame, text="Kalibrieren (Kontur erfassen)", command=calibrate, font=("Arial", 14), width=20) 
btn_calibrate.pack(pady=5)

# === 6. Analyse bei ruhendem Teil im ROI mit Bildregistrierung ===
def analyze_current_frame_contour():
    global reference_contour, current_frame
    if reference_contour is None:
        update_status("Keine Referenzkontur vorhanden.", ok=False)
        return
    if current_frame is None:
        update_status("Kein aktuelles Kamerabild.", ok=False)
        return

    gray_test = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # PERSPEKTIV-KORREKTUR: Wende vor der Analyse an (wenn kalibriert)
    if use_perspective_correction and perspective_corrector.is_calibrated:
        gray_test = perspective_corrector.correct_perspective(gray_test)
        print(f"DEBUG: Perspektiv-Korrektur angewendet")
    
    # Konsistente Homomorphic-Filter-Anwendung
    if use_homomorphic_filter:
        gray_test = homomorphic_filter(gray_test)
        # Verwende den konsistent gefilterten Hintergrund
        background_filtered = homomorphic_filter(background_image_raw)
        if use_clahe_filter:
            background_for_diff = apply_clahe_filter(background_filtered)
        else:
            background_for_diff = cv2.equalizeHist(background_filtered)
    else:
        background_for_diff = background_image
    
    # CLAHE für lokale Kontrastverbesserung
    if use_clahe_filter:
        gray_test = apply_clahe_filter(gray_test)
    else:
        gray_test = cv2.equalizeHist(gray_test)

    # === EINHEITLICHE Konturerkennung (identisch zu Kalibrierung) ===
    thresh = consistent_contour_detection(
        gray_test, 
        background_for_diff, 
        method="robust" if use_robust_contours else "standard",
        threshold_type=threshold_method_for_alignment
    )

    # === EINHEITLICHE Konturextraktion ===
    test_contour, area_test = extract_main_contour(thresh, min_area=100)
    
    print(f"Debug: Analyse - Konturfläche: {area_test:.0f}")
    
    if test_contour is None:
        update_status("Keine Testkontur im ROI erkannt!", ok=False)
        return

    area_ref = cv2.contourArea(reference_contour)
    
    # ERWEITERTE Flächen-Toleranz für verschiedene Objekte (Stift vs. Zange)
    # Alte Toleranz: ±50% war zu restriktiv für völlig verschiedene Objekte
    min_area_factor = 0.1  # 10% der Referenzfläche (sehr permissiv)
    max_area_factor = 10.0  # 1000% der Referenzfläche (sehr permissiv)
    
    if area_test < area_ref * min_area_factor or area_test > area_ref * max_area_factor:
        print(f"DEBUG: Flächen-Check SEHR permissiv - Test: {area_test:.0f}, Ref: {area_ref:.0f}")
        print(f"DEBUG: Erlaubter Bereich: {area_ref * min_area_factor:.0f} bis {area_ref * max_area_factor:.0f}")
        update_status(f"Objekt zu klein/groß (Test {area_test:.0f} vs. Ref {area_ref:.0f})", ok=False)
        return
    else:
        print(f"DEBUG: Flächen-Check BESTANDEN - Test: {area_test:.0f}, Ref: {area_ref:.0f} (Faktor: {area_test/area_ref:.2f})")

    # === NEUE BILDREGISTRIERUNG STATT HU-MOMENTE ===
    if use_geometric_alignment:
        # 1. Geometrische Ausrichtung des Test-Bildes
        aligned_image, alignment_info = geometric_aligner.align_test_image(gray_test, test_contour)
        
        if alignment_info is None:
            update_status("Geometrische Ausrichtung fehlgeschlagen!", ok=False)
            return
        
        # 2. Binäre Masken mit Regionprops-Filterung erstellen
        # KRITISCHER BUGFIX: Beide Masken müssen in derselben geometrischen Ausrichtung sein!
        
        # Test-Maske: Aus dem ausgerichteten Bild
        aligned_test_mask = geometric_aligner.create_binary_mask_with_regionprops(aligned_image, threshold_method='adaptive')
        
        # REFERENZ-MASKE: Muss AUCH geometrisch ausgerichtet werden!
        # Option 1: Direkt aus der Referenzkontur eine Maske in der Standard-Position erstellen
        reference_mask_direct = np.zeros_like(aligned_image)
        
        # Verwende die REFERENZ-KONTUR in der Standard-Ausrichtung
        # Verschiebe sie zum Referenz-Zentroid (wo auch das Test-Objekt ausgerichtet wird)
        ref_contour_aligned = reference_contour.copy()
        
        # Berechne Offset zur Standard-Position
        ref_moments = cv2.moments(reference_contour)
        if ref_moments['m00'] > 0:
            ref_cx_orig = int(ref_moments['m10'] / ref_moments['m00'])
            ref_cy_orig = int(ref_moments['m01'] / ref_moments['m00'])
            
            # Verschiebe Referenz-Kontur zur Standard-Position (wo Test-Objekt auch liegt)
            offset_x = geometric_aligner.reference_centroid[0] - ref_cx_orig
            offset_y = geometric_aligner.reference_centroid[1] - ref_cy_orig
            
            ref_contour_aligned = ref_contour_aligned + [offset_x, offset_y]
            
            # Zeichne Referenz-Maske an der GLEICHEN Position wie Test-Maske
            cv2.drawContours(reference_mask_direct, [ref_contour_aligned], -1, 255, thickness=cv2.FILLED)
        
        # Regionprops-Filterung auf beide Masken anwenden
        reference_mask_clean = geometric_aligner._filter_mask_with_regionprops(reference_mask_direct)
        
        print(f"Debug: Masken erstellt mit ADAPTIVE Threshold")
        print(f"Debug: KORREKTUR - Beide Masken geometrisch ausgerichtet!")
        
        # 5. Debug-Bilder speichern - Erstelle neuen Debug-Ordner für diese Analyse
        create_debug_folder()  # Neuer Ordner für jede Analyse
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # DEBUG: Speichere beide Masken für Vergleich
        save_debug_image(reference_mask_clean, f"debug_ref_mask_{timestamp}.png", "Masks")
        save_debug_image(aligned_test_mask, f"debug_test_mask_{timestamp}.png", "Masks")
        
        # DEBUG: Prüfe Masken-Statistiken
        ref_pixels = cv2.countNonZero(reference_mask_clean)
        test_pixels = cv2.countNonZero(aligned_test_mask)
        print(f"Debug: Referenz-Pixel: {ref_pixels}, Test-Pixel: {test_pixels}")
        print(f"Debug: Größenverhältnis: {test_pixels/ref_pixels:.3f}" if ref_pixels > 0 else "Debug: Referenz hat 0 Pixel!")
        
        # DEBUG: Erweiterte Bilder speichern
        save_debug_image(aligned_image, f"debug_aligned_image_{timestamp}.png", "Aligned_Images")
        save_debug_image(gray_test, f"debug_raw_test_{timestamp}.png", "Raw_Images")
        save_debug_image(reference_image, f"debug_raw_ref_{timestamp}.png", "Raw_Images")
        
        # DEBUG: Masken-Eigenschaften prüfen
        if ref_pixels == 0:
            print("KRITISCHER FEHLER: Referenzmaske ist komplett schwarz!")
            update_status("Referenzmaske fehlerhaft - Kalibrierung wiederholen!", ok=False)
            return
        
        if test_pixels == 0:
            print("KRITISCHER FEHLER: Testmaske ist komplett schwarz!")
            update_status("Testmaske fehlerhaft - Objekt nicht erkannt!", ok=False)
            return
        
        # 3. Pixelweise Differenzberechnung (echte Shapedifferenz)
        diff_result = geometric_aligner.calculate_pixel_difference(reference_mask_clean, aligned_test_mask)
        
        # DEBUG: Zusätzliche XOR-Visualisierung für bessere Diagnose
        xor_diff = cv2.bitwise_xor(reference_mask_clean, aligned_test_mask)
        save_debug_image(xor_diff, f"debug_xor_diff_{timestamp}.png", "Difference_Analysis")
        
        # DEBUG: Overlay-Bild für visuellen Vergleich (wie bei detaillierter Analyse)
        overlay_debug = np.zeros((reference_mask_clean.shape[0], reference_mask_clean.shape[1], 3), dtype=np.uint8)
        overlay_debug[reference_mask_clean > 0] = [0, 255, 0]      # Grün: Referenz
        overlay_debug[aligned_test_mask > 0] = [0, 0, 255]         # Blau: Test
        overlap_debug = cv2.bitwise_and(reference_mask_clean, aligned_test_mask)
        overlay_debug[overlap_debug > 0] = [255, 255, 0]           # Gelb: Überlappung
        save_debug_image(overlay_debug, f"debug_overlay_{timestamp}.png", "Difference_Analysis")
        
        # DEBUG: Überlappungsrate berechnen für bessere Diagnose
        overlap_pixels = cv2.countNonZero(overlap_debug)  # Korrekte Zählung der gelben Pixel
        overlap_ratio = overlap_pixels / ref_pixels if ref_pixels > 0 else 0
        print(f"DEBUG: Überlappungsrate: {overlap_ratio:.3f} ({overlap_pixels} von {ref_pixels} Pixel)")
        print(f"DEBUG: Differenz-Ratio von calculate_pixel_difference: {diff_result['difference_ratio']:.3f}")
        
        # KRITISCHE DIAGNOSE: Schaue ob überhaupt ein Unterschied erkannt wird
        if overlap_ratio > 0.95:  # Mehr als 95% Überlappung
            print("🚨 KRITISCH: Fast 100% Überlappung - Bildregistrierung funktioniert möglicherweise nicht!")
            print("   Mögliche Ursachen: 1) Identisches Objekt 2) Masken-Fehler 3) Alignment-Fehler")
        elif overlap_ratio < 0.05:  # Weniger als 5% Überlappung
            print("🚨 KRITISCH: Fast 0% Überlappung - Alignment komplett fehlgeschlagen!")
            print("   Mögliche Ursachen: 1) Völlig verschiedene Objekte 2) Rotation-Fehler 3) Translation-Fehler")
        
        if overlap_ratio < 0.1:  # Weniger als 10% Überlappung
            print("WARNUNG: Sehr geringe Überlappung - möglicherweise völlig verschiedene Objekte!")
        elif overlap_ratio > 0.8:  # Mehr als 80% Überlappung
            print("INFO: Hohe Überlappung - sehr ähnliche Objekte!")
        else:
            print(f"INFO: Moderate Überlappung - unterschiedliche aber verwandte Objekte")
        
        # 4. Toleranz-Check basierend auf Pixeldifferenz
        # ANGEPASST: Höhere Toleranz für Test verschiedener Objekte (Stift vs. Zange)
        pixel_tolerance = 0.30  # 30% Pixeldifferenz-Toleranz (war 5% - viel zu streng!)
        difference_ratio = diff_result['difference_ratio']
        is_defect_free = difference_ratio <= pixel_tolerance
        
        print(f"DEBUG: Pixeldifferenz-Check - Ratio: {difference_ratio:.3f}, Toleranz: {pixel_tolerance:.3f}")
        print(f"DEBUG: Fehlerfrei? {is_defect_free}")
        
        # Debug-Bilder waren bereits gespeichert
        
        # 6. Ergebnis ausgeben
        if is_defect_free:
            # Zeige sowohl Pixel- als auch mm-Messungen
            if use_perspective_correction and perspective_corrector.is_calibrated:
                diff_pixels_mm = perspective_corrector.pixel_to_mm(diff_result['total_difference_pixels'])
                update_status(f"Teil fehlerfrei - Pixeldiff: {difference_ratio:.3f} ({diff_result['total_difference_pixels']} px = {diff_pixels_mm:.1f} mm²)", ok=True)
                print(f"Pixelanalyse: {diff_result['missing_pixels']} fehlend, {diff_result['extra_pixels']} zusätzlich")
                print(f"mm-Analyse: {perspective_corrector.pixel_to_mm(diff_result['missing_pixels']):.1f} mm² fehlend, {perspective_corrector.pixel_to_mm(diff_result['extra_pixels']):.1f} mm² zusätzlich")
            else:
                update_status(f"Teil fehlerfrei - Pixeldiff: {difference_ratio:.3f} ({diff_result['total_difference_pixels']} Pixel)", ok=True)
                print(f"Pixelanalyse: {diff_result['missing_pixels']} fehlend, {diff_result['extra_pixels']} zusätzlich")
        else:
            # Zeige sowohl Pixel- als auch mm-Messungen
            if use_perspective_correction and perspective_corrector.is_calibrated:
                diff_pixels_mm = perspective_corrector.pixel_to_mm(diff_result['total_difference_pixels'])
                update_status(f"FEHLERHAFT - Pixeldiff: {difference_ratio:.3f} ({diff_result['total_difference_pixels']} px = {diff_pixels_mm:.1f} mm²)", ok=False)
                print(f"Fehler: {diff_result['total_difference_pixels']} Pixeldifferenz = {diff_pixels_mm:.1f} mm²")
                print(f"Details: {diff_result['missing_pixels']} px ({perspective_corrector.pixel_to_mm(diff_result['missing_pixels']):.1f} mm²) fehlend")
                print(f"         {diff_result['extra_pixels']} px ({perspective_corrector.pixel_to_mm(diff_result['extra_pixels']):.1f} mm²) zusätzlich")
            else:
                update_status(f"FEHLERHAFT - Pixeldiff: {difference_ratio:.3f} (>{pixel_tolerance:.3f})", ok=False)
                print(f"Fehler: {diff_result['total_difference_pixels']} Pixeldifferenz")
                print(f"Details: {diff_result['missing_pixels']} fehlend, {diff_result['extra_pixels']} zusätzlich")
            
            # Fehlerbild mit Kontur markieren
            result_frame = current_frame.copy()
            cv2.drawContours(result_frame, [test_contour], -1, (0, 0, 255), thickness=3)
            
            # Erweiterte Fehlerprotokollierung
            img_name = f"defect_geometric_{timestamp}.png"
            img_path = save_debug_image(result_frame, img_name, "Defect_Images")
            
            with open("pruefung_protokoll.csv", "a") as log:
                log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},Fehler,{img_path},Pixel-Diff:{difference_ratio:.3f},Fehlend:{diff_result['missing_pixels']},Zusätzlich:{diff_result['extra_pixels']}\n")
            
            print(f"Fehlerbild gespeichert: {img_name}")
            
    else:
        # Fallback: Verwende verbesserte Hu-Moment-Methode
        if use_robust_contours:
            score = shape_matching_improved(reference_contour, test_contour, method='multiple')
        else:
            score = cv2.matchShapes(reference_contour, test_contour, cv2.CONTOURS_MATCH_I1, 0.0)
        
        result_frame = current_frame.copy()
        if score <= shape_tolerance:
            update_status(f"Teil ist fehlerfrei (Score={score:.3f})", ok=True)
        else:
            update_status(f"Teil ist FEHLERHAFT (Score={score:.3f})", ok=False)
            cv2.drawContours(result_frame, [test_contour], -1, (0, 0, 255), thickness=3)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"defect_{timestamp}.png"
            img_path = save_debug_image(result_frame, img_name, "Defect_Images")
            with open("pruefung_protokoll.csv", "a") as log:
                log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},Fehler,{img_path}\n")
            print(f"Fehlerbild gespeichert: {img_path} (Score={score:.3f})")

# === 7. Live-Anzeige + Bewegungserkennung im ROI ===
def process_frame():
    global current_frame, previous_frame, stable_counter, part_present_state, error_state

    ret, frame = cap.read()
    if not ret:
        root.after(50, process_frame)
        return

    # Vollbild für Anzeige behalten
    full_frame = frame.copy()
    
    # ROI für Verarbeitung ausschneiden
    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    current_frame = roi.copy()
    frame_rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # ROI-Rechteck auf Vollbild zeichnen
    cv2.rectangle(full_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    cv2.putText(full_frame, "ROI", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Status-Text auf Vollbild anzeigen
    status_text = status_label.cget("text")
    status_color = (0, 255, 0) if status_label.cget("fg") == "green" else (0, 0, 255)
    cv2.putText(full_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    if previous_frame is None:
        previous_frame = frame_rgb_roi.copy()
    else:
        diff = cv2.absdiff(previous_frame, frame_rgb_roi)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        blurred_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        _, diff_thresh = cv2.threshold(blurred_diff, motion_threshold, 255, cv2.THRESH_BINARY)
        motion_pixels = cv2.countNonZero(diff_thresh)
        movement_detected = (motion_pixels > motion_pixels_threshold)

        if movement_detected:
            stable_counter = 0
            if part_present_state or error_state:
                part_present_state = False
                error_state = False
                update_status("Bewegung erkannt. Warte auf ruhiges Bild …", ok=True)
        else:
            if stable_counter < motion_frames_stable:
                stable_counter += 1

        if stable_counter >= motion_frames_stable and reference_contour is not None and not part_present_state and not error_state:
            analyze_current_frame_contour()
            stable_counter = 0

        previous_frame = frame_rgb_roi.copy()

    # Vollbild für Tkinter-Anzeige konvertieren
    full_frame_rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(full_frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)

    root.after(20, process_frame)

# === EINSTELLUNGEN IM EINSTELLUNGEN-TAB ===

# 1. PERSPEKTIV-KORREKTUR SEKTION
perspective_section = ttk.LabelFrame(scrollable_frame, text="Perspektiv-Korrektur", padding=10)
perspective_section.pack(fill="x", padx=10, pady=5)

# Perspektiv-Kalibrierung Button
def calibrate_perspective():
    """Kalibriert Perspektiv-Korrektur basierend auf aktuellem ROI"""
    ret, frame = cap.read()
    if not ret:
        update_status("Fehler: Kein Kamerabild für Perspektiv-Kalibrierung.", ok=False)
        return
    
    # Verwende die ausgewählte ROI-Größe
    roi_width_mm, roi_height_mm = ROI_PRESETS[current_roi_preset]
    
    success = perspective_corrector.calibrate_perspective(frame, roi_width_mm, roi_height_mm)
    
    if success:
        update_status(f"Perspektiv-Kalibrierung erfolgreich! ({current_roi_preset}: {roi_width_mm}x{roi_height_mm} mm)", ok=True)
        print(f"Auflösung: {perspective_corrector.pixels_per_mm:.1f} Pixel/mm")
        print(f"ROI-Preset: {current_roi_preset}")
    else:
        update_status("Perspektiv-Kalibrierung fehlgeschlagen!", ok=False)

btn_calibrate_perspective = tk.Button(perspective_section, text="Perspektiv-Kalibrierung", command=calibrate_perspective, font=("Arial", 12))
btn_calibrate_perspective.pack(pady=5)

# Toggle-Button für Perspektiv-Korrektur
def toggle_perspective_correction():
    global use_perspective_correction
    use_perspective_correction = not use_perspective_correction
    btn_perspective.config(text=f"Perspektiv-Korrektur: {'AN' if use_perspective_correction else 'AUS'}")
    if use_perspective_correction:
        print("Perspektiv-Korrektur aktiviert: Echte Draufsicht + mm-Messungen")
    else:
        print("Perspektiv-Korrektur deaktiviert: Normale Kamera-Perspektive")

btn_perspective = tk.Button(perspective_section, text=f"Perspektiv-Korrektur: {'AN' if use_perspective_correction else 'AUS'}", 
                           command=toggle_perspective_correction, font=("Arial", 10))
btn_perspective.pack(pady=5)

# ROI-Größen-Auswahl
tk.Label(perspective_section, text="ROI-Größe für Perspektiv-Korrektur:", font=("Arial", 12, "bold")).pack(pady=(10,5))

def update_roi_preset():
    """Callback für ROI-Größen-Auswahl"""
    global current_roi_preset
    current_roi_preset = roi_var.get()
    width_mm, height_mm = ROI_PRESETS[current_roi_preset]
    print(f"ROI-Größe gewechselt auf: {current_roi_preset} ({width_mm}x{height_mm} mm)")
    update_roi_info_label()
    
    # Zeige aktuelle Auswahl in Status
    if perspective_corrector.is_calibrated:
        update_status(f"ROI-Größe geändert auf {current_roi_preset}. Neu kalibrieren!", ok=False)

# Dropdown für ROI-Größen
roi_var = tk.StringVar(value=current_roi_preset)
roi_options = list(ROI_PRESETS.keys())

roi_dropdown = tk.OptionMenu(perspective_section, roi_var, *roi_options, command=lambda x: update_roi_preset())
roi_dropdown.config(font=("Arial", 10))
roi_dropdown.pack(pady=5)

# Zeige aktuelle Größe
def update_roi_info_label():
    width_mm, height_mm = ROI_PRESETS[current_roi_preset]
    roi_info_label.config(text=f"Aktuelle Größe: {width_mm}x{height_mm} mm")

roi_info_label = tk.Label(perspective_section, text="", font=("Arial", 10))
roi_info_label.pack()
update_roi_info_label()

# Benutzerdefinierte ROI-Größe Eingabe
custom_frame = tk.Frame(perspective_section)
custom_frame.pack(pady=10)

tk.Label(custom_frame, text="Benutzerdefiniert (mm):", font=("Arial", 10)).grid(row=0, column=0, columnspan=2)

tk.Label(custom_frame, text="Breite:", font=("Arial", 9)).grid(row=1, column=0, sticky="e")
custom_width_entry = tk.Entry(custom_frame, width=8, font=("Arial", 9))
custom_width_entry.grid(row=1, column=1, padx=5)
custom_width_entry.insert(0, "150")

tk.Label(custom_frame, text="Höhe:", font=("Arial", 9)).grid(row=2, column=0, sticky="e")
custom_height_entry = tk.Entry(custom_frame, width=8, font=("Arial", 9))
custom_height_entry.grid(row=2, column=1, padx=5)
custom_height_entry.insert(0, "120")

def apply_custom_roi():
    """Wendet benutzerdefinierte ROI-Größe an"""
    try:
        width = float(custom_width_entry.get())
        height = float(custom_height_entry.get())
        
        if width <= 0 or height <= 0:
            update_status("Ungültige ROI-Größe!", ok=False)
            return
        
        # Aktualisiere benutzerdefinierte Größe
        ROI_PRESETS["Benutzerdefiniert"] = (width, height)
        
        # Setze auf benutzerdefiniert
        roi_var.set("Benutzerdefiniert")
        global current_roi_preset
        current_roi_preset = "Benutzerdefiniert"
        
        update_roi_info_label()
        print(f"Benutzerdefinierte ROI-Größe: {width}x{height} mm")
        
        if perspective_corrector.is_calibrated:
            update_status(f"ROI-Größe auf {width}x{height} mm geändert. Neu kalibrieren!", ok=False)
        
    except ValueError:
        update_status("Ungültige Zahlenwerte für ROI-Größe!", ok=False)

tk.Button(custom_frame, text="Anwenden", command=apply_custom_roi, font=("Arial", 9)).grid(row=3, column=0, columnspan=2, pady=5)

# 2. FILTER-EINSTELLUNGEN SEKTION
filter_section = ttk.LabelFrame(scrollable_frame, text="Filter-Einstellungen", padding=10)
filter_section.pack(fill="x", padx=10, pady=5)

# Filter-Buttons in Grid anordnen
filter_grid = tk.Frame(filter_section)
filter_grid.pack(pady=10)

# Toggle-Button für Homomorphic Filter
def toggle_homomorphic():
    global use_homomorphic_filter
    use_homomorphic_filter = not use_homomorphic_filter
    btn_homo.config(text=f"Homomorphic Filter: {'AN' if use_homomorphic_filter else 'AUS'}")
    print(f"Homomorphic Filter {'aktiviert' if use_homomorphic_filter else 'deaktiviert'}")

btn_homo = tk.Button(filter_grid, text=f"Homomorphic Filter: {'AN' if use_homomorphic_filter else 'AUS'}", 
                     command=toggle_homomorphic, font=("Arial", 10))
btn_homo.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

# Toggle-Button für Temporal Filter
def toggle_temporal():
    global use_temporal_filter
    use_temporal_filter = not use_temporal_filter
    btn_temporal.config(text=f"Temporal Filter: {'AN' if use_temporal_filter else 'AUS'}")
    print(f"Temporal Filter {'aktiviert' if use_temporal_filter else 'deaktiviert'}")
    # Reset der Filter wenn deaktiviert
    if not use_temporal_filter:
        temporal_filter_bg.reset()
        temporal_filter_live.reset()
        temporal_filter_calib.reset()

btn_temporal = tk.Button(filter_grid, text=f"Temporal Filter: {'AN' if use_temporal_filter else 'AUS'}", 
                        command=toggle_temporal, font=("Arial", 10))
btn_temporal.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

# Toggle-Button für CLAHE Filter
def toggle_clahe():
    global use_clahe_filter
    use_clahe_filter = not use_clahe_filter
    btn_clahe.config(text=f"CLAHE Filter: {'AN' if use_clahe_filter else 'AUS'}")
    print(f"CLAHE Filter {'aktiviert' if use_clahe_filter else 'deaktiviert'}")

btn_clahe = tk.Button(filter_grid, text=f"CLAHE Filter: {'AN' if use_clahe_filter else 'AUS'}", 
                     command=toggle_clahe, font=("Arial", 10))
btn_clahe.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

# Toggle-Button für Robuste Konturerkennung
def toggle_robust_contours():
    global use_robust_contours
    use_robust_contours = not use_robust_contours
    btn_robust.config(text=f"Robuste Konturen: {'AN' if use_robust_contours else 'AUS'}")
    print(f"Robuste Konturerkennung {'aktiviert' if use_robust_contours else 'deaktiviert'}")

btn_robust = tk.Button(filter_grid, text=f"Robuste Konturen: {'AN' if use_robust_contours else 'AUS'}", 
                      command=toggle_robust_contours, font=("Arial", 10))
btn_robust.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

# Toggle-Button für Geometrische Ausrichtung  
def toggle_geometric_alignment():
    global use_geometric_alignment
    use_geometric_alignment = not use_geometric_alignment
    btn_geometric.config(text=f"Geometric Align: {'AN' if use_geometric_alignment else 'AUS'}")
    if use_geometric_alignment:
        print("Bildregistrierung aktiviert: Pixelgenaue Ausrichtung + Differenzberechnung")
    else:
        print("Bildregistrierung deaktiviert: Fallback auf Hu-Moment-Matching")

btn_geometric = tk.Button(filter_grid, text=f"Geometric Align: {'AN' if use_geometric_alignment else 'AUS'}", 
                         command=toggle_geometric_alignment, font=("Arial", 10))
btn_geometric.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

# Toggle-Button für Threshold-Methode  
def toggle_threshold_method():
    global threshold_method_for_alignment
    methods = ["otsu", "adaptive", "fixed"]
    current_index = methods.index(threshold_method_for_alignment)
    threshold_method_for_alignment = methods[(current_index + 1) % len(methods)]
    btn_threshold.config(text=f"Threshold: {threshold_method_for_alignment.upper()}")
    print(f"Threshold-Methode gewechselt auf: {threshold_method_for_alignment}")

btn_threshold = tk.Button(filter_grid, text=f"Threshold: {threshold_method_for_alignment.upper()}", 
                         command=toggle_threshold_method, font=("Arial", 10))
btn_threshold.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

# Grid-Spalten gleichmäßig verteilen
filter_grid.grid_columnconfigure(0, weight=1)
filter_grid.grid_columnconfigure(1, weight=1)

# 3. KAMERA-EINSTELLUNGEN SEKTION
camera_section = ttk.LabelFrame(scrollable_frame, text="Kamera-Einstellungen", padding=10)
camera_section.pack(fill="x", padx=10, pady=5)

# Belichtungsregler mit Schieberegler
def update_exposure(value):
    """Callback für Belichtungs-Schieberegler"""
    exposure_value = float(value)
    if cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value):
        exposure_label.config(text=f"Belichtung: {exposure_value:.1f}")
    else:
        print("Belichtung konnte nicht geändert werden")

def reset_camera_settings():
    """Setzt alle Kamera-Einstellungen zurück"""
    configure_camera_anti_flicker()
    # Aktualisiere Schieberegler auf neuen Wert
    current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    exposure_slider.set(current_exposure)

# Belichtungsregler
exposure_frame = tk.Frame(camera_section)
exposure_frame.pack(pady=5)

current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
exposure_label = tk.Label(exposure_frame, text=f"Belichtung: {current_exposure:.1f}", font=("Arial", 10))
exposure_label.pack()

exposure_slider = tk.Scale(
    exposure_frame, 
    from_=-13, 
    to=0, 
    resolution=0.5,
    orient=tk.HORIZONTAL, 
    length=300,
    command=update_exposure,
    font=("Arial", 9)
)
exposure_slider.set(current_exposure)
exposure_slider.pack()

# Reset Button
tk.Button(camera_section, text="Kamera Reset", command=reset_camera_settings, font=("Arial", 10)).pack(pady=5)

# === 9. Live-Loop starten ===
process_frame()

# === 10. Tkinter-Event-Loop ===
root.mainloop()

# === 11. Kamera freigeben ===
cap.release()
