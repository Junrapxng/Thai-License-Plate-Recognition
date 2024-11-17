from flask import Flask, request, render_template, redirect, url_for
import cv2 
import easyocr
import numpy as np
import os
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def enhance_image(image):
    """ปรับปรุงคุณภาพของภาพ"""
    # แปลงเป็นภาพสีเทา
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ปรับความสว่างและคอนทราสต์
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # ลดสัญญาณรบกวน
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, searchWindowSize=21, templateWindowSize=7)
    
    # ปรับความคมชัด
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # ปรับขนาดภาพให้ใหญ่ขึ้น
    scale_factor = 2.0
    enlarged = cv2.resize(sharpened, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # ปรับ threshold แบบ adaptive
    binary = cv2.adaptiveThreshold(enlarged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # ทำ morphological operations เพื่อลดสัญญาณรบกวนและเชื่อมตัวอักษร
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    return morph

def detect_plate_area(image):
    """ตรวจจับพื้นที่ป้ายทะเบียน"""
    # แปลงเป็นภาพสีเทา
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ปรับ blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # หาขอบด้วย Canny
    edged = cv2.Canny(blur, 50, 200)
    
    # ทำ dilation เพื่อเชื่อมเส้นขอบ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(edged, kernel, iterations=1)
    
    # หา contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for contour in contours:
        # หาสี่เหลี่ยมที่มีอัตราส่วนใกล้เคียงป้ายทะเบียน
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        
        if len(approx) == 4:
            # ตรวจสอบอัตราส่วน
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            
            # อัตราส่วนป้ายทะเบียนไทยประมาณ 2:1
            if 1.5 <= aspect_ratio <= 3.0:
                return approx
    
    return None

def format_license_plate(text):
    """จัดรูปแบบข้อความป้ายทะเบียน"""
    provinces = [
        'กรุงเทพมหานคร', 'กระบี่', 'กาญจนบุรี', 'กาฬสินธุ์', 'กำแพงเพชร', 'ขอนแก่น', 'จันทบุรี', 
        'ฉะเชิงเทรา', 'ชลบุรี', 'ชัยนาท', 'ชัยภูมิ', 'ชุมพร', 'เชียงราย', 'เชียงใหม่', 'ตรัง', 
        'ตราด', 'ตาก', 'นครนายก', 'นครปฐม', 'นครพนม', 'นครราชสีมา', 'นครศรีธรรมราช', 'นครสวรรค์', 
        'นนทบุรี', 'นราธิวาส', 'น่าน', 'บึงกาฬ', 'บุรีรัมย์', 'ปทุมธานี', 'ประจวบคีรีขันธ์', 
        'ปราจีนบุรี', 'ปัตตานี', 'พระนครศรีอยุธยา', 'พังงา', 'พัทลุง', 'พิจิตร', 'พิษณุโลก', 
        'เพชรบุรี', 'เพชรบูรณ์', 'แพร่', 'พะเยา', 'ภูเก็ต', 'มหาสารคาม', 'มุกดาหาร', 'แม่ฮ่องสอน', 
        'ยะลา', 'ยโสธร', 'ร้อยเอ็ด', 'ระนอง', 'ระยอง', 'ราชบุรี', 'ลพบุรี', 'ลำปาง', 'ลำพูน', 
        'เลย', 'ศรีสะเกษ', 'สกลนคร', 'สงขลา', 'สตูล', 'สมุทรปราการ', 'สมุทรสงคราม', 'สมุทรสาคร', 
        'สระแก้ว', 'สระบุรี', 'สิงห์บุรี', 'สุโขทัย', 'สุพรรณบุรี', 'สุราษฎร์ธานี', 'สุรินทร์', 
        'หนองคาย', 'หนองบัวลำภู', 'อ่างทอง', 'อุดรธานี', 'อุทัยธานี', 'อุตรดิตถ์', 'อุบลราชธานี', 
        'อำนาจเจริญ'
    ]
    
    text = text.replace('\n', '').strip()
    words = text.split()
    
    province = None
    for word in words:
        if word in provinces:
            province = word
            break
    
    # อัปเดตรูปแบบทะเบียนให้รองรับตัวเลขนำหน้า
    plate_pattern = r'\d?[ก-ฮ]{1,3}\s*\d{1,4}'
    plate_match = re.search(plate_pattern, text)
    
    if plate_match:
        plate_number = plate_match.group().strip()
        letters = re.findall(r'[ก-ฮ]+', plate_number)[0]
        numbers = re.findall(r'\d+', plate_number)
        
        # จัดรูปแบบใหม่โดยรวมเลขนำหน้า (ถ้ามี)
        if len(numbers) > 1:
            formatted_plate = f"{numbers[0]}{letters} {numbers[1]}"
        else:
            formatted_plate = f"{letters} {numbers[0]}"
        
        if province:
            return f"{formatted_plate} {province}"
        return f"{formatted_plate}"
    
    return "ไม่สามารถระบุเลขทะเบียนได้"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            text, plate_image_path = process_image(file_path)
            return render_template("index.html", text=text, plate_image_path=plate_image_path)
    
    return render_template("index.html")

def process_image(image_path):
    # อ่านภาพ
    image = cv2.imread(image_path)
    
    # ตรวจจับพื้นที่ป้ายทะเบียน
    plate_contour = detect_plate_area(image)
    
    if plate_contour is None:
        return "ไม่พบแผ่นป้ายทะเบียน", None

    # สร้าง mask จาก contour
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    
    # ตัดเฉพาะส่วนป้ายทะเบียน
    x, y, w, h = cv2.boundingRect(plate_contour)
    plate_region = image[y:y+h, x:x+w]
    
    # ปรับปรุงคุณภาพของภาพป้ายทะเบียน
    enhanced_plate = enhance_image(plate_region)
    
    # บันทึกภาพที่ปรับปรุงแล้ว
    plate_image_path = "uploads/enhanced_plate.jpg"
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "enhanced_plate.jpg"), enhanced_plate)
    
    # อ่านตัวอักษรด้วย EasyOCR
    reader = easyocr.Reader(['th', 'en'])
    
    # ลองอ่านทั้งภาพปกติและภาพที่ปรับปรุงแล้ว
    result_original = reader.readtext(plate_region)
    result_enhanced = reader.readtext(enhanced_plate)
    
    # เลือกผลลัพธ์ที่น่าจะถูกต้องที่สุด
    text_original = " ".join([res[1] for res in result_original])
    text_enhanced = " ".join([res[1] for res in result_enhanced])
    
    # เลือกข้อความที่ยาวกว่า เพราะน่าจะมีข้อมูลครบถ้วนกว่า
    text = text_enhanced if len(text_enhanced) > len(text_original) else text_original
    
    # จัดรูปแบบข้อความ
    formatted_text = format_license_plate(text)
    
    return formatted_text, plate_image_path

if __name__ == "__main__":
    app.run(debug=True)