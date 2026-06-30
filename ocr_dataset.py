import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import requests
from tqdm import tqdm
from paddleocr import PaddleOCR
from requests.exceptions import RequestsDependencyWarning

def info_images(csv_path, output_dir, max_valid_images=20):

	# Create folder for images
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
		print(f"Created {output_dir}")

	print("Initializing PaddleOCR...")
	ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False, use_gpu=False)

	print(f"Loading file {csv_path}")
	df = pd.read_csv(csv_path)
	url_column = 'url'

	if url_column not in df.columns:
		raise ValueError("Did not find columns with URL images.")

	urls = df[url_column].dropna().unique() # skip missing values and filter out duplicate URLs

	print(f"Start downloading until {max_valid_images} functional text images")

	success_count = 0
	for idx, url in enumerate(urls):
		if success_count >= max_valid_images:
			break

		# Skip corrupted on non-HTTP text entries
		if not str(url).startswith(('http://', 'https://')):
			continue

		file_path = None

		try:
			response = requests.get(url, timeout=10, stream=True)
			if response.status_code == 200:

				# Correct file extensions
				ext = ".jpg"
				if ".png" in url.lower(): ext = ".png"
				elif ".jpeg" in url.lower(): ext = ".jpeg"
				elif ".webp" in url.lower(): ext = ".webp"

				filename = f"image_{idx}{ext}"
				file_path = os.path.join(output_dir, filename)

				with open(file_path, 'wb') as f:
					for chunk in response.iter_content(chunk_size=8192):
						f.write(chunk)

				result = ocr.ocr(file_path, cls=False) # layout analysis and text extraction

				# Delete file if no text
				if not result or result[0] is None:
					os.remove(file_path)
				else:
					success_count += 1
					print(f"[{success_count}/{max_valid_images}]. Saved valid image: {filename}")
					print("Extracted text:")
				words = []
				# iterate through the main layout blocks detected on image
				for block in result:
					# process each text line within the current block
					for line in block:
						words.append(line[1][0]) # extract only the raw text
				print(" ".join(words)) # merge and print extracted text

		except Exception as e:
			if file_path is not None and os.path.exists(file_path):
				os.remove(file_path)
			continue

	print(f"\nSaved {success_count} photos in {output_dir}")

if __name__ == "__main__":
	CSV_FILE = "archive/image_metadata.csv"
	OUTPUT_FOLDER = "archive/ocr_images"

	info_images(CSV_FILE, OUTPUT_FOLDER, max_valid_images=15)
