import os
import time
import random
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import comet_ml
from comet_ml.integration.pytorch import log_model
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score, classification_report, ConfusionMatrixDisplay
from transformers import AutoTokenizer
from dotenv import load_dotenv

from utils import load_prepare_data
from model import Classifier

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

load_dotenv()

experiment = comet_ml.start(
	project_name="fake-news",
	workspace="hasakij"
)

hyper_params = {
	"vocab_size": 250002,
	"embedding_dim": 512,
	"learning_rate": 0.0002,
	"epochs": 20,
	"batch_size": 64,
	"lstm_hidden": 128,
	"max_length": 128,
	"num_classes": 4
}
experiment.log_parameters(hyper_params)

# Load cached tokenized tensor if available
CACHE_X = "dataset/X_tensor.pt"
CACHE_Y = "dataset/y_tensor.pt"
if os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
	print("Loading data...")
	X = torch.load(CACHE_X, weights_only=True)
	y = torch.load(CACHE_Y, weights_only=True)
else:
	df = load_prepare_data()
	print("Tokenizing text...")
	tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
	texts = df["text"].tolist()

	# Convert texts into token ids
	encodings = tokenizer(
		texts,
		padding="max_length",
		truncation=True,
		max_length=hyper_params["max_length"],
		return_tensors="pt"
	)

	X = encodings["input_ids"]
	y = torch.tensor(df["label"].values, dtype=torch.long)

	# Save for future runs
	torch.save(X, CACHE_X)
	torch.save(y, CACHE_Y)

# 70 % train, 15 % val, 15 % test
X_train, X_temp, y_train, y_temp = train_test_split(
	X, y, test_size=0.3, random_state=42, stratify=y.numpy()
)
X_val, X_test, y_val, y_test = train_test_split(
	X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp.numpy()
)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=hyper_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hyper_params["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=hyper_params["batch_size"], shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = Classifier(
	vocab_size=hyper_params["vocab_size"],
	embedding_dim=hyper_params["embedding_dim"],
	lstm_hidden=hyper_params["lstm_hidden"],
	num_classes=hyper_params["num_classes"]
).to(device)

# Compute class weights for imbalanced dataset
class_counts = torch.bincount(y_train, minlength=hyper_params["num_classes"]).float()
class_weights = class_counts.sum() / (hyper_params["num_classes"] * class_counts)
print(f"class counts: {class_counts} \nclass weights: {class_weights}")
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

# AdamW optimizer with weight decay regularization
optimizer = torch.optim.AdamW(
	model.parameters(),
	lr=hyper_params["learning_rate"],
	weight_decay=1e-4
)

# Reduce learning rate when validation loss stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
	optimizer,
	mode="min",
	factor=0.5,
	patience=2,
	min_lr=1e-5
)

best_val_loss = float("inf")
early_stop_patience = 3
counter = 0
train_start = time.time()

train_losses = []
val_losses = []

# Training and validation
for epoch in range(hyper_params["epochs"]):

	# Training phase
	model.train()
	running_train_loss = 0.0

	for batch_idx, (text_tokens, labels) in enumerate(train_loader):
		text_tokens, labels = text_tokens.to(device), labels.to(device)

		optimizer.zero_grad()
		outputs = model(text_tokens)
		loss = criterion(outputs, labels)
		loss.backward()

		nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # prevent exploding gradients
		optimizer.step()

		running_train_loss += loss.item()

	epoch_train_loss = running_train_loss / len(train_loader)
	train_losses.append(epoch_train_loss)
	print(f"Epoch [{epoch+1}/{hyper_params['epochs']}] \ntrain loss: {epoch_train_loss:.4f}")
	experiment.log_metric("train_loss", epoch_train_loss, epoch=epoch)

	# Validation phase
	model.eval()
	running_val_loss = 0.0
	all_preds = []
	all_labels = []

	with torch.no_grad():
		for batch_idx, (text_tokens, labels) in enumerate(val_loader):
			text_tokens, labels = text_tokens.to(device), labels.to(device)

			outputs = model(text_tokens)
			loss = criterion(outputs, labels)
			running_val_loss += loss.item()

			probs = torch.softmax(outputs, dim=1) # convert logits to class probabilities
			preds = torch.argmax(probs, dim=1) # select class with highest probability

			all_preds.extend(preds.cpu().numpy().ravel())
			all_labels.extend(labels.cpu().numpy().ravel())

	epoch_val_loss = running_val_loss / len(val_loader)
	val_losses.append(epoch_val_loss)

	# Validation metrics
	val_accuracy = accuracy_score(all_labels, all_preds)
	val_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
	val_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
	val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
	val_conf_matrix = confusion_matrix(all_labels, all_preds)
	val_report = classification_report(all_labels, all_preds, target_names=["TRUE", "FAKE", "SATIRE", "BIAS"], digits=2, zero_division=0)

	
	val_metrics = {"val_loss": epoch_val_loss, "val_accuracy": val_accuracy, "val_precision": val_precision, "val_recall": val_recall, "val_f1": val_f1}
	for name, value in val_metrics.items():
		print(f"{name}: {value:.4f}")
		experiment.log_metric(name, value, epoch=epoch)
	print(f"Val confusion matrix:\n{val_conf_matrix}")
	experiment.log_confusion_matrix(matrix=val_conf_matrix, labels=["TRUE", "FAKE", "SATIRE", "BIAS"], title="Validation confusion matrix",epoch=epoch)
	print(f"Val classification report: \n{val_report}")

	scheduler.step(epoch_val_loss) # update learning rate scheduler

	# Save best model
	if epoch_val_loss < best_val_loss:
		best_val_loss = epoch_val_loss
		counter = 0

		torch.save(
			model.state_dict(),
			"best_model.pt"
		)
		print("Best model saved")
	else:
		counter += 1

	if counter >= early_stop_patience:
		print("Early stopping")
		break

	current_lr = optimizer.param_groups[0]["lr"]
	print(f"Learning rate: {current_lr}")
	experiment.log_metric("learning_rate_x1e4", current_lr * 10000, epoch=epoch)
train_end = time.time()


# Test phase
print("\nLoading best model for test...")
model.load_state_dict(torch.load("best_model.pt", map_location=device, weights_only=True))

model.eval()

running_test_loss = 0.0

all_preds = []
all_labels = []

with torch.no_grad():
	for batch_idx, (text_tokens, labels) in enumerate(test_loader):
		text_tokens, labels = text_tokens.to(device), labels.to(device)

		outputs = model(text_tokens)
		loss = criterion(outputs, labels)
		running_test_loss += loss.item()

		probs = torch.softmax(outputs, dim=1)
		preds = torch.argmax(probs,dim=1)

		all_preds.extend(preds.cpu().numpy().ravel())
		all_labels.extend(labels.cpu().numpy().ravel())

test_loss = running_test_loss / len(test_loader)

# Test metrics
test_accuracy = accuracy_score(all_labels, all_preds)
test_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
test_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
test_conf_matrix = confusion_matrix(all_labels, all_preds)
test_report = classification_report(all_labels, all_preds, target_names=["TRUE","FAKE","SATIRE","BIAS"], digits=2, zero_division=0)

print("\nFinal results:")
test_metrics = {"test_loss": test_loss, "test_accuracy": test_accuracy, "test_precision": test_precision, "test_recall": test_recall, "test_f1": test_f1}
for name, value in test_metrics.items():
	print(f"{name}: {value:.4f}")
	experiment.log_metric(name, value)
print(f"Test confusion matrix: \n{test_conf_matrix}")
experiment.log_confusion_matrix(matrix=test_conf_matrix, labels=["TRUE", "FAKE", "SATIRE", "BIAS"], title="Test confusion matrix")
print(f"Test classification report: \n{test_report}")

print(f"Training time: {(train_end - train_start) / 60:.2f} min")

os.makedirs("images", exist_ok=True)

# Save and log confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=test_conf_matrix, display_labels=["TRUE", "FAKE", "SATIRE", "BIAS"])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Test Confusion Matrix")
plt.tight_layout()

conf_matrix_path = "images/confusion_matrix.png"
plt.savefig(conf_matrix_path, dpi=300)

experiment.log_image(image_data=conf_matrix_path, name="confusion_matrix.png")
plt.close(fig)

# Save and log loss curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_losses, marker="o", label="Train loss")
ax.plot(val_losses, marker="o", label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training and Validation Loss")
ax.legend()
ax.grid(True)
plt.tight_layout()

loss_curve_path = "images/loss_curve.png"
plt.savefig(loss_curve_path, dpi=300)

experiment.log_image(image_data=loss_curve_path, name="loss_curve.png")
plt.close(fig)

print("Saving model to Comet ML...")
log_model(experiment, model=model, model_name="fake-news")

experiment.end()
print("Training complete")
