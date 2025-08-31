import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

def detect_faces_from_image():
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )

    if not image_path:
        messagebox.showwarning("No file", "No image selected!")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", f"Could not load image from {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Detection (Press any key to close)", image)
    cv2.imwrite("detected_faces.jpg", image)
    print(f"Found {len(faces)} face(s). Output saved as detected_faces.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_faces_from_webcam():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Webcam Face Detection (Press 'q' or ESC to quit)", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- GUI ----------------
def main():
    root = tk.Tk()
    root.title("Simple Image Recognition System")
    root.geometry("400x250")
    root.resizable(False, False)

    title = tk.Label(root, text="Face Detection System", font=("Arial", 16, "bold"))
    title.pack(pady=20)

    btn_image = tk.Button(root, text="Detect Faces from Image", font=("Arial", 12),
                          command=detect_faces_from_image, width=25, bg="lightblue")
    btn_image.pack(pady=10)

    btn_webcam = tk.Button(root, text="Detect Faces from Webcam", font=("Arial", 12),
                           command=detect_faces_from_webcam, width=25, bg="lightgreen")
    btn_webcam.pack(pady=10)

    btn_exit = tk.Button(root, text="Exit", font=("Arial", 12),
                         command=root.quit, width=25, bg="salmon")
    btn_exit.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
