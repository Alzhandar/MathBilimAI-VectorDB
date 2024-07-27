import requests

url = "http://127.0.0.1:8000/upload_book"
file_path = "/Users/lzandaribaev/Desktop/englishman-around-the-world_/server/uploads/Approaches_to_Teaching_English_in_the_Elementary_School.pdf"

# Проверьте, существует ли файл и имеет ли он правильное расширение
if not file_path.endswith(".pdf"):
    print("Error: File is not a PDF")
else:
    try:
        with open(file_path, "rb") as f:
            files = {"file": ("Approaches_to_Teaching_English_in_the_Elementary_School.pdf", f, "application/pdf")}
            response = requests.post(url, files=files)

        print(response.status_code)
        print(response.json())
    except FileNotFoundError:
        print("Error: File not found")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
