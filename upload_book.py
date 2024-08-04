import requests

url = "http://127.0.0.1:8000/upload_book"
file_path = "/Users/lzandaribaev/Desktop/MathTeachAI update front/fast_api/uploads/978-5-7996-2028-8_2017.pdf"

# Проверьте, существует ли файл и имеет ли он правильное расширение
if not file_path.endswith(".pdf"):
    print("Error: File is not a PDF")
else:
    try:
        print("Opening the file...")
        with open(file_path, "rb") as f:
            files = {"file": ("978-5-7996-2028-8_2017.pdf", f, "application/pdf")}
            print("Sending the request...")
            response = requests.post(url, files=files)
            print("Request sent, awaiting response...")

        print("Response received")
        print(response.status_code)
        print(response.json())
    except FileNotFoundError:
        print("Error: File not found")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
