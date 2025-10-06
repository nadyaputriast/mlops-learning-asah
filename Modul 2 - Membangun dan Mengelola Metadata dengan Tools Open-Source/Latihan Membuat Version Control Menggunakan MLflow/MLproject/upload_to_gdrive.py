import os
import json
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Load environment variables from .env file
load_dotenv()

# 1. Load credentials from JSON file
json_file_path = r"C:\Kuliah\Semester 5\Coding\ML-System\totemic-inquiry-473805-f0-bd899ef66142.json"

try:
    with open(json_file_path, 'r') as f:
        creds = json.load(f)
    print("JSON credentials loaded successfully!")
except FileNotFoundError:
    print(f"JSON file not found: {json_file_path}")
    print("Please check the file path!")
    exit(1)

# 2. Build credentials
credentials = Credentials.from_service_account_info(
    creds,
    scopes=["https://www.googleapis.com/auth/drive"]
)

# 3. Build Drive API
service = build('drive', 'v3', credentials=credentials)

# 4. Get Shared Drive ID from .env
SHARED_DRIVE_ID = os.getenv("GDRIVE_FOLDER_ID")
if not SHARED_DRIVE_ID:
    print("GDRIVE_FOLDER_ID not found in .env file!")
    exit(1)

print(f"Using Shared Drive ID: {SHARED_DRIVE_ID}")

def upload_directory(local_dir_path, parent_drive_id):
    """
    Rekursif:
     - Jika item folder, buat folder di Drive, lalu panggil upload_directory lagi.
     - Jika item file, langsung upload ke parent_drive_id.
    """
    for item_name in os.listdir(local_dir_path):
        item_path = os.path.join(local_dir_path, item_name)
        if os.path.isdir(item_path):
            folder_meta = {
                'name': item_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_drive_id]
            }
            created_folder = service.files().create(
                body=folder_meta,
                fields='id',
                supportsAllDrives=True
            ).execute()
            new_folder_id = created_folder["id"]
            print(f"Created folder: {item_name} (ID: {new_folder_id})")

            # Rekursif ke subfolder
            upload_directory(item_path, new_folder_id)
        else:
            print(f"Uploading file: {item_name}")
            file_meta = {
                'name': item_name,
                'parents': [parent_drive_id]
            }
            media = MediaFileUpload(item_path, resumable=True)
            service.files().create(
                body=file_meta,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()


# 4. Baca semua subfolder (run_id) di "./mlruns/0"
#    Kemudian buat folder sesuai run_id di Shared Drive (tanpa folder "mlruns" agar tidad redundant).
local_mlruns_0 = r"C:\Kuliah\Semester 5\Coding\ML-System\mlruns\0"

for run_id in os.listdir(local_mlruns_0):
    run_id_local_path = os.path.join(local_mlruns_0, run_id)
    # Pastikan hanya folder (bukan file)
    if os.path.isdir(run_id_local_path):
        # Buat folder dengan nama run_id di root Shared Drive (SHARED_DRIVE_ID)
        run_id_folder_meta = {
            'name': run_id,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [SHARED_DRIVE_ID]
        }
        run_id_folder = service.files().create(
            body=run_id_folder_meta,
            fields='id',
            supportsAllDrives=True
        ).execute()
        run_id_folder_id = run_id_folder["id"]
        print(f"=== Created run_id folder: {run_id} (ID: {run_id_folder_id}) ===")

        # Upload isinya (subfolder, file) secara rekursif
        upload_directory(run_id_local_path, run_id_folder_id)

print("=== All run_id folders and files have been uploaded directly to Shared Drive! ===")