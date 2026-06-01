import pandas as pd
import re
import spacy
from langdetect import detect

MODELS = {}

def get_spacy_doc(text):

	# Detect language and select appropriate spacy model
	try:
		lang = detect(text)
	except:
		lang = "en"

	if lang == "pl":
		model_name = "pl_core_news_sm"
	else:
		model_name = "en_core_web_sm" 

	# Load model once and keep it in memory 
	if model_name not in MODELS:
		try:
			MODELS[model_name] = spacy.load(model_name)
		except OSError:
			MODELS[model_name] = spacy.load("en_core_web_sm")

	nlp = MODELS[model_name]
	return nlp(text)
	
def clean_text(text):

	if not isinstance(text, str):
		return ""

	# Find "Reuters" with dashes and remove everything before that
	text = re.sub(re.compile(r'^[^--\-]*\(Reuters\)\s*[--\-]\s*'), '', text)

	# Remove repetitive messages
	text = re.sub(r'Reuters has not edited the statements or confirmed their accuracy\.', '', text)

	# Remove URL links and usernames
	text = re.sub(r'https?://\S+|www\.\S+|bit\.ly/\S+', '', text, flags=re.IGNORECASE)
	text = re.sub(r'@\w+', '', text)

	# Remove timestamps in square brackets
	text = re.sub(r'\[d+\s*EST\]', '', text)

	# Remove "Source link:" and "---"
	text = re.sub(r'Source link:', '', text, flags=re.IGNORECASE)
	text = re.sub(r'--{2,}', '', text)
	
	# Lower letters and remove excess spaces
	text = text.lower()
	text = re.sub(r'\s+', ' ', text).strip()

	return text

def combine_text(row):

	title = str(row["title"])
	content = str(row["content"])

	# Avoid duplicating title when title and content are the same
	if title == content:
		return content

	return title + " " + content

def load_prepare_data():
	
	print("Loading data...")
	df = pd.read_csv("dataset/Dataset_Clean.csv", low_memory=False) # load dataset

	# Replace missing values with empty string
	df["title"] = df["title"].fillna("").astype(str)
	df["content"] = df["content"].fillna("").astype(str)

	# Final text for training
	df["text"] = df.apply(combine_text, axis=1)
	df = df[["title", "text", "label", "label_text"]]

	print("Cleaning text...")

	# Apply clean function only on text
	df["text"] = df["text"].apply(clean_text)

	# Remove blank lines
	df = df[df["text"].str.strip() != ""]

	len_before = len(df)
	duplicates_before = df["text"].duplicated().sum()

	print(f"Samples before deduplication: {len_before}")
	print(f"Duplicate texts before deduplication: {duplicates_before}")

	# Remove duplicates
	df = df.drop_duplicates(subset=["text"])
	len_after = len(df)
	duplicates_after = df["text"].duplicated().sum()

	print(f"Samples after deduplication: {len_after}")
	print(f"Duplicate texts after deduplication: {duplicates_after}")
	
	class_counts = df["label"].value_counts()
	print(f"\nClass distribution:\n- Real (0): {class_counts.get(0, 0)} \n- Fake (1): {class_counts.get(1, 0)} \n- Satire (2): {class_counts.get(2, 0)} \n- Bias (3): {class_counts.get(3, 0)}")
	return df

if __name__ == "__main__":
	df = load_prepare_data()
	print("\nFew samples:")
	print(df.head())
	for label in sorted(df["label"].unique()):
		label_name = df[df["label"] == label]["label_text"].iloc[0]
		print(f"{label}: {label_name}")
		print(df[df["label"] == label]["text"].iloc[0])
	print("\nText length statistics:")
	df["length"] = df["text"].str.len()
	print(df["length"].describe().to_string())
