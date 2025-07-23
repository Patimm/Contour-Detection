Work in Progress



Qualitätskontrolle aktueller Stand/Ergänzungen


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
