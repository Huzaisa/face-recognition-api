from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
import uuid
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import pytz
TIME_ZONE = pytz.timezone("Asia/Jakarta")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FACE_DIR = "../public/uploads/employee_faces"
DATABASE_URL = "postgresql://postgres:pinter12@localhost:5432/absensi"
engine = create_engine(DATABASE_URL)


KNOWN_ENCODINGS = {}


def load_known_faces():
    if not os.path.exists(FACE_DIR):
        os.makedirs(FACE_DIR)

    with engine.connect() as conn:
        result = conn.execute(text('SELECT "userId", "faceEncoding" FROM "FaceRegistration"'))
        for row in result.mappings():
            user_id = row["userId"]
            encoding_bytes = row["faceEncoding"]
            if encoding_bytes:
                try:
                    
                    encoding_array = np.frombuffer(encoding_bytes, dtype=np.float64)
                    KNOWN_ENCODINGS[user_id] = encoding_array
                except Exception as e:
                    print(f"Gagal decode encoding untuk user {user_id}: {e}")

    print(f"{len(KNOWN_ENCODINGS)} wajah berhasil dimuat ke memori.")

load_known_faces()


def encode_face(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        return encodings[0] if encodings else None
    except Exception as e:
        print(f"Error saat encode wajah: {e}")
        return None



def save_attendance(user_id: str):
    now_local = datetime.now(TIME_ZONE)           
    midnight_utc = datetime.strptime(
        f"{now_local.strftime('%Y-%m-%d')}T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"
    )                                             

    attendance_id = str(uuid.uuid4())

    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT s."startTime"
            FROM "ShiftMapping" sm
            JOIN "Shift" s ON sm."shiftId" = s."id"
            WHERE sm."userId" = :userId AND sm."date" = :date
        """), {"userId": user_id, "date": midnight_utc}).mappings().fetchone()

        if not row:
            raise HTTPException(400, "User belum punya shift hari ini.")

        
        shift_start_local = row["startTime"].astimezone(TIME_ZONE)

        is_late = now_local.time() > shift_start_local.time()
        status  = "LATE" if is_late else "ONTIME"

        conn.execute(text("""
            INSERT INTO "Attendance"
              ("id","userId","date","clockIn","status","isLate","createdAt","updatedAt")
            VALUES
              (:id,:userId,:date,:clockIn,:status,:isLate,:createdAt,:updatedAt)
            ON CONFLICT ("userId","date") DO NOTHING;
        """), {
            "id": attendance_id,
            "userId": user_id,
            "date": midnight_utc,
            "clockIn": now_local,
            "status": status,
            "isLate": is_late,
            "createdAt": now_local,
            "updatedAt": now_local
        })




@app.post("/register/")
async def register_face(file: UploadFile = File(...), userId: str = Form(...)):
    image_bytes = await file.read()
    encoding = encode_face(image_bytes)
    if encoding is None:
        return JSONResponse(content={"detail": "Tidak ada wajah terdeteksi."}, status_code=400)

    
    filename = f"{userId}_{int(datetime.now().timestamp())}.jpg"
    save_path = os.path.join(FACE_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    new_id = str(uuid.uuid4())
    
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO "FaceRegistration" ("id", "userId", "imagePath", "faceEncoding", "createdAt")
                VALUES (:id, :userId, :imagePath, :faceEncoding, NOW())
                ON CONFLICT ("userId") DO UPDATE SET 
                    "imagePath" = :imagePath,
                    "faceEncoding" = :faceEncoding,
                    "createdAt" = NOW();
            """),
            {
                "id": new_id,
                "userId": userId,
                "imagePath": filename,
                "faceEncoding": encoding.tobytes()
            }
        )

    
    KNOWN_ENCODINGS[userId] = encoding

    return {"detail": f"Wajah untuk {userId} berhasil diregistrasi."}


@app.post("/verify/")
async def verify_face(file: UploadFile = File(None)):
    try:
        rgb_frame = None

        # 🔁 Prioritaskan gambar dari file
        if file is not None:
            image_bytes = await file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(400, "Gambar tidak valid.")
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ✅ Jika tidak ada file, baru fallback ke kamera
        else:
            cam = cv2.VideoCapture(0)
            if cam.isOpened():
                for _ in range(5): cam.read()
                ret, frame = cam.read()
                cam.release()

                if not ret or frame is None or frame.size == 0:
                    raise HTTPException(status_code=400, detail="Gagal mengambil gambar dari kamera.")

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raise HTTPException(400, "Kamera tidak tersedia dan file tidak dikirim.")

        # Debug simpan gambar terakhir
        cv2.imwrite("last_frame.jpg", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        if not face_locations:
            raise HTTPException(404, "Wajah tidak terdeteksi. Pastikan wajah terlihat jelas di kamera.")

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not face_encodings:
            raise HTTPException(404, "Gagal mengenali encoding wajah.")

        for encoding in face_encodings:
            best_match_user = None
            best_match_dist = 1.0
            for user_id, known_encoding in KNOWN_ENCODINGS.items():
                dist = face_recognition.face_distance([known_encoding], encoding)[0]
                if dist < 0.7 and dist < best_match_dist:
                    best_match_dist = dist
                    best_match_user = user_id

            if best_match_user:
                save_attendance(best_match_user)
                return {"userId": best_match_user, "detail": f"Wajah user {best_match_user} terverifikasi dengan jarak {best_match_dist:.3f}."}

        raise HTTPException(status_code=403, detail="Wajah tidak dikenali atau belum diregistrasi.")

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error di endpoint /verify/: {e}")
        raise HTTPException(500, detail="Terjadi kesalahan pada sistem verifikasi.")


@app.get("/faces/")
def get_faces():
    return {"registered": list(KNOWN_ENCODINGS.keys())}
