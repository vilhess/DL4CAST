import gdown

# Dictionnaire des fichiers à télécharger : {nom_fichier: id_google_drive}
file_ids = {
    "data/etth1.csv": "1vOClm_t4RgUf8nqherpTfNnB8rrYq14Q",
    "data/etth2.csv": "1bOcmp9VAv03d3kUYSrttOFvLZ0keXDC5",
    "data/ettm1.csv": "1B7VcTWdIfPl3g17zKXATKF9XQJtNHTtl",
    "data/ettm2.csv": "1JweODeVxt6YTIRFA0ivAgZQkR3rldtbi",
    "data/exchange_rate.csv": "1EBLfP2Dx2K7LsSZybX4JJes-wEcTJbz-",
    "data/weather.csv": "1Tc7GeVN7DLEl-RAs-JVwG9yFMf--S8dy",
    "data/national_illness.csv": "1n4kDxT38rQEGmczIaYaaUWqRPWs-rtcH",
}

for filename, file_id in file_ids.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {filename}...")
    gdown.download(url, filename, quiet=False)
