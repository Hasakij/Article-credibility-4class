import argparse
import os
import torch
import pdfplumber
import trafilatura
from transformers import AutoTokenizer
from paddleocr import PaddleOCR

from model import Classifier
from utils import clean_text

def extract_text_from_image(image_path):

	# Extract text from PNG/JPEG/JPG/WEBP images using PaddleOCR

	ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False, use_gpu=False)
	
	result = ocr.ocr(image_path, cls=False)

	if not result or result[0] is None:
		raise ValueError("PaddleOCR did not extract any text from this image")

	words = []
	# iterate through the main layout blocks detected on image
	for block in result:
		# process each text line within the current block
		for line in block:
			words.append(line[1][0]) # extract only the raw text
	text = " ".join(words) # merge extracted text

	if len(text.strip()) == 0:
		raise ValueError("Could not extract text from image")

	return text

def extract_text_from_url(url):
	downloaded = trafilatura.fetch_url(url) # download web page

	if downloaded is None:
		raise ValueError("Could not download URL content")

	text = trafilatura.extract(downloaded)

	if text is None or len(text.strip()) == 0:
		raise ValueError("Could not extract text from URL")

	return text

def extract_text_from_pdf(pdf_path):
	pages_text = [] # store text from all PDF pages

	with pdfplumber.open(pdf_path) as pdf:
		for page in pdf.pages:
			page_text = page.extract_text() # extract text from current page

			if page_text:
				pages_text.append(page_text)

		text = "\n".join(pages_text) # merge all pages into one document

		if len(text.strip()) == 0:
			raise ValueError("Could not extract text from PDF")

		return text
def predict_text(text, model, tokenizer, device, max_length=128):
	text = clean_text(text) # apply the same preprocessing as during training

	# Convert text into token ids
	encodings = tokenizer(
		text,
		padding="max_length",
		truncation=True,
		max_length=max_length,
		return_tensors="pt"
	)

	input_ids = encodings["input_ids"].to(device)

	model.eval() # inference mode

	with torch.no_grad():
		logits = model(input_ids) # raw model outputs
		probs = torch.softmax(logits, dim=1) # convert logits to class probabilities

	pred_label = torch.argmax(probs, dim=1).item() # select class with highest probability

	return pred_label, probs.cpu().numpy().ravel()

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--text", type=str)
	parser.add_argument("--url", type=str)
	parser.add_argument("--pdf", type=str)
	parser.add_argument("--image", type=str)

	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

	model = Classifier(
		vocab_size=250002,
		embedding_dim=512,
		lstm_hidden=128,
		num_classes=4
	).to(device)

	# Load trained model weights
	model.load_state_dict(torch.load("best_model.pt", map_location=device, weights_only=True))

	# Read input from text, URL, PDF or image
	if args.text:
		raw_text = args.text
	elif args.url:
		raw_text = extract_text_from_url(args.url)
	elif args.pdf:
		raw_text = extract_text_from_pdf(args.pdf)
	elif args.image:
		raw_text = extract_text_from_image(args.image)
	else:
		raise ValueError("Provide --text, --url, --pdf or --image")

	label, probs = predict_text(
		raw_text,
		model,
		tokenizer,
		device
	)
	
	# Class names
	classes = {
	0: "REAL",
	1: "FAKE",
	2: "SATIRE",
	3: "BIAS"
	}

	print("Extracted text:")
	print(raw_text)
	print(f"Prediction: {classes[label]}")
	for idx, prob in enumerate(probs):
		print(f"{classes[idx]} probability: {prob:.2f}")

if __name__ == "__main__":
	main()
