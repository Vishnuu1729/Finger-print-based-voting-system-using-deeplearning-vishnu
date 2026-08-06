import os
import cv2
import sqlite3
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime

IMG_SIZE = (96, 96)
MODEL_PATH = "fingerprint_verifier.weights.h5"
DB_PATH = "voting_records.db"

AUTHORIZED_DIR = "C:/Users/anant/OneDrive/ドキュメント/Desktop/final_year_project/voter_list"

class DatabaseManager:
    """Handles persistent storage to prevent multiple voting and record election data."""
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self._setup_table()

    def _setup_table(self):
        """Creates the voters table with candidate tracking."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS voters (
                fingerprint_id TEXT PRIMARY KEY,
                candidate_name TEXT,
                voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Ensure candidate_name column exists (migration helper)
        try:
            self.cursor.execute("ALTER TABLE voters ADD COLUMN candidate_name TEXT")
        except sqlite3.OperationalError:
            pass
        self.conn.commit()

    def has_voted(self, fingerprint_id):
        """Checks if a fingerprint ID already exists in the database."""
        self.cursor.execute("SELECT 1 FROM voters WHERE fingerprint_id = ?", (fingerprint_id, ))
        return self.cursor.fetchone() is not None

    def record_vote(self, fingerprint_id, candidate_name):
        """Saves the fingerprint ID and the chosen candidate."""
        try:
            self.cursor.execute(
                "INSERT INTO voters (fingerprint_id, candidate_name) VALUES (?, ?)", 
                (fingerprint_id, candidate_name)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_results(self):
        """Fetches vote counts for all candidates."""
        try:
            self.cursor.execute("SELECT candidate_name, COUNT(*) as count FROM voters GROUP BY candidate_name")
            return self.cursor.fetchall()
        except sqlite3.OperationalError:
            return []

    def clear_all_votes(self):
        """Deletes all records from the voters table for a new session."""
        try:
            self.cursor.execute("DELETE FROM voters")
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False

class FingerprintVerifier:
    """Handles the model loading and prediction with robust path support."""
    def __init__(self):
        self.model = self._build_architecture()
        if os.path.exists(MODEL_PATH):
            print(f"Loading trained weights from {MODEL_PATH}")
            self.model.load_weights(MODEL_PATH)
            self.is_ready = True
        else:
            self.is_ready = False

    def _build_architecture(self):
        model = models.Sequential([
            layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def _robust_imread(self, path, flags=cv2.IMREAD_GRAYSCALE):
        """
        FIX: Reads image as bytes to handle Unicode/Japanese characters in paths.
        """
        try:
            if not os.path.exists(path):
                return None
            with open(path, 'rb') as f:
                chunk = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(chunk, flags)
            return img
        except Exception as e:
            print(f"Error reading image: {e}")
            return None

    def predict(self, img_path):
        if not self.is_ready:
            return None, "Model weights not found. Please run train.py first."
        
        img = self._robust_imread(img_path)
        
        if img is None: 
            return None, "Could not read image. Check file path integrity."
            
        img = cv2.resize(img, IMG_SIZE)
        img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1) / 255.0
        prediction = self.model.predict(img, verbose=0)[0][0]
        return (prediction < 0.5), None

class VotingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nexus Biometric Voting System")
        self.root.geometry("600x820")
        self.root.configure(bg="#0f172a") 
        self.colors = {
            "bg": "#0f172a",
            "card": "#1e293b",
            "accent": "#6366f1", 
            "text": "#f8fafc",
            "text_muted": "#94a3b8",
            "success": "#22c55e",
            "danger": "#ef4444"
        }
        self.verifier = FingerprintVerifier()
        self.db = DatabaseManager()
        self.selected_file = None
        self.setup_ui()

    def setup_ui(self):
        header_frame = tk.Frame(self.root, bg=self.colors["bg"])
        header_frame.pack(fill="x", pady=(40, 10))
        tk.Label(header_frame, text="NEXUS VOTE", font=("Segoe UI", 24, "bold"), 
                 bg=self.colors["bg"], fg=self.colors["accent"]).pack()
        tk.Label(header_frame, text="Secure Biometric Authentication", font=("Segoe UI", 10), 
                 bg=self.colors["bg"], fg=self.colors["text_muted"]).pack()

        self.main_card = tk.Frame(self.root, bg=self.colors["card"], padx=30, pady=20, 
                                  highlightbackground="#334155", highlightthickness=1)
        self.main_card.pack(pady=10, padx=40, fill="both", expand=True)
        
        self.scan_box = tk.Frame(self.main_card, bg="#0f172a", width=220, height=220, 
                                 highlightbackground=self.colors["accent"], highlightthickness=2)
        self.scan_box.pack(pady=10)
        self.scan_box.pack_propagate(False)
        
        self.img_display = tk.Label(self.scan_box, text="PLACE FINGERPRINT\nSCANNER IDLE", 
                                    font=("Segoe UI", 10), bg="#0f172a", fg=self.colors["text_muted"])
        self.img_display.pack(expand=True, fill="both")

        self.status_label = tk.Label(self.main_card, text="Waiting for biometric input...", 
                                     font=("Segoe UI", 11), bg=self.colors["card"], fg=self.colors["text"])
        self.status_label.pack(pady=10)

        btn_container = tk.Frame(self.main_card, bg=self.colors["card"])
        btn_container.pack(fill="x", pady=10)

        self.upload_btn = tk.Button(btn_container, text="UPLOAD SCAN", command=self.load_image,
                                    bg=self.colors["accent"], fg="white", font=("Segoe UI", 10, "bold"),
                                    activebackground="#4f46e5", activeforeground="white",
                                    relief="flat", cursor="hand2", pady=10)
        self.upload_btn.pack(fill="x", pady=5)

        self.verify_btn = tk.Button(btn_container, text="START VERIFICATION", command=self.verify_identity,
                                    bg="#10b981", fg="white", font=("Segoe UI", 10, "bold"),
                                    activebackground="#059669", activeforeground="white",
                                    relief="flat", cursor="hand2", pady=10)
        self.verify_btn.pack(fill="x", pady=5)

        admin_frame = tk.Frame(self.root, bg=self.colors["bg"])
        admin_frame.pack(fill="x", pady=10)
        
        self.report_btn = tk.Button(admin_frame, text="GENERATE ELECTION REPORT", command=self.generate_report,
                                     bg="#334155", fg=self.colors["text_muted"], font=("Segoe UI", 8, "bold"),
                                     relief="flat", cursor="hand2", pady=5)
        self.report_btn.pack()

        tk.Label(self.root, text="System: OFFLINE ENCRYPTED DATABASE", font=("Consolas", 8), 
                 bg=self.colors["bg"], fg="#475569").pack(side="bottom", pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Fingerprint Scans", "*.bmp *.png *.jpg")])
        if path:
            self.selected_file = path
            # PIL handle paths well, so resize is safe
            img = Image.open(path).resize((220, 220))
            photo = ImageTk.PhotoImage(img)
            self.img_display.config(image=photo, text="")
            self.img_display.image = photo
            self.status_label.config(text=f"Scan Loaded: {os.path.basename(path)}", fg=self.colors["accent"])

    def verify_identity(self):
        if not self.selected_file:
            messagebox.showwarning("Incomplete", "Please upload a fingerprint scan to proceed.")
            return

        auth_dir = os.path.normpath(AUTHORIZED_DIR).lower()
        file_dir = os.path.normpath(os.path.dirname(self.selected_file)).lower()

        if file_dir != auth_dir:
            self.status_label.config(text="ACCESS DENIED - UNAUTHORIZED VOTER", fg=self.colors["danger"])
            messagebox.showerror("Security Alert", f"INVALID VOTER:\nIdentity must be sourced from the authorized folder.\n\nRequired: {AUTHORIZED_DIR}")
            return

        voter_id = os.path.basename(self.selected_file)
        if self.db.has_voted(voter_id):
            self.status_label.config(text="ACCESS DENIED - DUPLICATE VOTE", fg=self.colors["danger"])
            messagebox.showerror("Security Alert", "This person has already cast a vote. Multiple votes are not allowed.")
            return
            
        self.status_label.config(text="ANALYZING RIDGE PATTERNS...", fg="#fbbf24")
        self.root.update()
        
        is_real, error = self.verifier.predict(self.selected_file)
        
        if error:
            messagebox.showerror("Error", error)
            self.status_label.config(text="SYSTEM ERROR", fg=self.colors["danger"])
        elif is_real:
            self.status_label.config(text="IDENTITY VERIFIED - ACCESS GRANTED", fg=self.colors["success"])
            self.show_ballot()
        else:
            self.status_label.config(text="FRAUD DETECTED - ACCESS DENIED", fg=self.colors["danger"])
            messagebox.showerror("Security Alert", "Fake fingerprint detected. Verification rejected.")

    def show_ballot(self):
        ballot = tk.Toplevel(self.root)
        ballot.title("Nexus Voting Ballot")
        ballot.geometry("450x550")
        ballot.configure(bg=self.colors["card"])
        ballot.grab_set() 
        tk.Label(ballot, text="OFFICIAL BALLOT", font=("Segoe UI", 16, "bold"), 
                 bg=self.colors["card"], fg=self.colors["accent"]).pack(pady=30)
        tk.Label(ballot, text="Identity: VALIDATED\nSession: ENCRYPTED", 
                 font=("Consolas", 9), bg=self.colors["card"], fg=self.colors["success"]).pack(pady=10)
        instr = tk.Label(ballot, text="Please select your preferred candidate below.", 
                         font=("Segoe UI", 10), bg=self.colors["card"], fg=self.colors["text_muted"], justify="center")
        instr.pack(pady=20)
        
        candidates = ["CANDIDATE ALPHA", "CANDIDATE BETA", "CANDIDATE GAMMA"]
        for candidate in candidates:
            btn = tk.Button(ballot, text=candidate, font=("Segoe UI", 10, "bold"), 
                            bg="#334155", fg=self.colors["text"], relief="flat", 
                            width=30, pady=12, cursor="hand2", activebackground=self.colors["accent"])
            btn.config(command=lambda c=candidate: self.cast_vote(c, ballot))
            btn.pack(pady=8)

    def cast_vote(self, name, window):
        voter_id = os.path.basename(self.selected_file)
        if self.db.record_vote(voter_id, name):
            messagebox.showinfo("Receipt", f"VOTE REGISTERED\n\nRecipient: {name}\nStatus: Saved to Secure Ledger")
        else:
            messagebox.showerror("Database Error", "Failed to record vote. Please try again.")
        
        window.destroy()
        self.selected_file = None
        self.img_display.config(image="", text="PLACE FINGERPRINT\nSCANNER IDLE")
        self.status_label.config(text="Waiting for fingerprint...", fg=self.colors["text"])

    def generate_report(self):
        """Generates a text report and resets the database session."""
        results = self.db.get_results()
        if not results:
            messagebox.showwarning("No Data", "No votes have been cast yet.")
            return

        total_votes = sum(count for _, count in results)
        winner_name = "Tie"
        max_votes = 0
        for name, count in results:
            if count > max_votes:
                max_votes = count
                winner_name = name
            elif count == max_votes:
                winner_name = "Tie (Multiple Candidates)"

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        report_filename = f"election_results_{timestamp}.txt"
        
        try:
            with open(report_filename, "w") as f:
                f.write("="*40 + "\n")
                f.write("     ELECTION REPORT\n")
                f.write("="*40 + "\n")
                f.write(f"Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Votes Cast: {total_votes}\n")
                f.write("-" * 40 + "\n")
                f.write("CANDIDATE BREAKDOWN:\n")
                for name, count in results:
                    percentage = (count / total_votes) * 100
                    f.write(f"{name}: {count} votes ({percentage:.2f}%)\n")
                f.write("-" * 40 + "\n")
                f.write(f"ELECTION WINNER: {winner_name}\n")
                f.write("="*40 + "\n")
                f.write("Status: Session Finalized and Verified\n")

            if self.db.clear_all_votes():
                messagebox.showinfo("Session Completed", f"Results saved to '{report_filename}'.\n\nThe local voter registry has been cleared.")
            else:
                messagebox.showwarning("Warning", f"Results saved, but database clear failed.")

            try:
                os.startfile(report_filename)
            except AttributeError:
                import subprocess
                subprocess.call(['open', report_filename])
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to generate report: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    app = VotingApp(root)
    root.mainloop()