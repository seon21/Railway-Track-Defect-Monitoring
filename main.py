import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import csv
from datetime import datetime

def log_inspection_to_csv(image_name, detected, area, length, width, severity, unsafe_zone):
    outputs_folder = "outputs"
    os.makedirs(outputs_folder, exist_ok=True)
    csv_path = os.path.join(outputs_folder, "inspection_log.csv")
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header if file is newly created
        if not file_exists:
            writer.writerow(["Timestamp", "Image_Name", "Defect_Detected", "Area_Pixels", "Length_Pixels", "Width_Pixels", "Severity", "Unsafe_Zone"])
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, image_name, "YES" if detected else "NO", f"{area:.2f}", length, width, severity, unsafe_zone])

def detect_rail_crack(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    image_filename = os.path.basename(image_path)
    scale_percent = 60
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    
    original = img.copy()
    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, "input.png"), original)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite(os.path.join(output_folder, "edge_detection.png"), edges)

    dark_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=15, C=5
    )

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, vertical_kernel)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, small_kernel)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = original.copy()
    for contour in contours:
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_folder, "contour_detection.png"), contour_image)

    best_contour = None
    best_score = -1

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if w == 0:
            continue

        aspect_ratio = h / float(w)
        if area < 100 or aspect_ratio < 1.5 or h < 30:
            continue

        contour_mask = np.zeros_like(gray)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        inside_pixels = gray[contour_mask > 0]
        
        if len(inside_pixels) == 0:
            continue

        mean_inside = np.mean(inside_pixels)
        dilated = cv2.dilate(contour_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        ring = (dilated > 0) & (contour_mask == 0)
        
        if np.sum(ring) == 0:
            continue

        mean_surrounding = np.mean(gray[ring])
        contrast = mean_surrounding - mean_inside

        if contrast < 10:
            continue

        score = (area * 0.4) + (h * 3) + (contrast * 5)
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        print("\n--------------------------------")
        print("RAILWAY TRACK INSPECTION RESULT")
        print("--------------------------------")
        print("No crack or structural defect detected.")
        print("--------------------------------")
        
        log_inspection_to_csv(image_filename, False, 0.0, 0, 0, "NONE", "NO")
        
        cv2.imwrite(os.path.join(output_folder, "final_output.png"), img)
        cv2.imshow("Rail Crack Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    area = cv2.contourArea(best_contour)
    x, y, w, h = cv2.boundingRect(best_contour)

    if area >= 500 or h >= 100:
        severity = "HIGH"
        unsafe_zone = "YES"
    elif area >= 200 or h >= 60:
        severity = "MEDIUM"
        unsafe_zone = "YES"
    else:
        severity = "LOW"
        unsafe_zone = "NO"

    cv2.rectangle(img, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), 3)
    label = f"DEFECT - {severity}"
    label_y = max(25, y - 12)
    cv2.putText(img, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    print("\n--------------------------------")
    print("RAILWAY TRACK INSPECTION RESULT")
    print("--------------------------------")
    print("Crack/Defect detected : YES")
    print(f"Defect area          : {area:.2f} pixels")
    print(f"Defect length        : {h} pixels")
    print(f"Defect width         : {w} pixels")
    print(f"Severity             : {severity}")
    print(f"Unsafe zone          : {unsafe_zone}")
    print("--------------------------------")

    log_inspection_to_csv(image_filename, True, area, h, w, severity, unsafe_zone)

    final_output_path = os.path.join(output_folder, "final_output.png")
    cv2.imwrite(final_output_path, img)
    print(f"Final output saved to: {final_output_path}")
    print(f"Inspection logged to : outputs/inspection_log.csv\n")

    cv2.imshow("Rail Crack Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("\n==============================================")
    print("FULL-FRAME RAILWAY TRACK DEFECT DETECTION")
    print("==============================================")

    while True:
        selected_file_path = filedialog.askopenfilename(
            title="Select Railway Track Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not selected_file_path:
            print("Exiting program.")
            break

        print(f"\nProcessing: {selected_file_path}")
        detect_rail_crack(selected_file_path)