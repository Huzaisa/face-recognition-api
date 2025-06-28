import cv2
import requests

url = "http://127.0.0.1:8000/verify/"  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    cv2.imshow("Tekan 's' untuk verifikasi wajah", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):  
        
        cv2.imwrite("temp.jpg", frame)

        
        with open("temp.jpg", "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, files=files)

        print("Respons dari API:", response.status_code, response.text)

    elif key & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
