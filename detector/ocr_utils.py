import pytesseract
from PIL import Image, ImageOps

def extract_text_from_image(image_path: str) -> str:
    img = Image.open(image_path)

    # Basic preprocessing for better OCR
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")  # grayscale

    text = pytesseract.image_to_string(img)
    return (text or "").strip()
