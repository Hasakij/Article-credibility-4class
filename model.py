import torch
import torch.nn as nn

class Classifier(nn.Module):
	def __init__(self, vocab_size, embedding_dim, num_classes, pretrained_weights=None,
				 num_filters=128, kernel_size=3, lstm_hidden=128):
		super(Classifier, self).__init__()

		# Use pretrained embeddings if provided, otherwise initialize from scratch
		if pretrained_weights is not None:
			self.embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=True)
		else:
			self.embedding = nn.Embedding(vocab_size, embedding_dim)

		# CNN layer (1D convolution)
		self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
							kernel_size=kernel_size, padding=1)
		self.relu = nn.ReLU()
		self.pool = nn.MaxPool1d(kernel_size=2)

		# LSTM layer
		self.lstm = nn.LSTM(input_size=num_filters, hidden_size=lstm_hidden,
							num_layers=1, batch_first=True, bidirectional=True)
		self.dropout = nn.Dropout(0.5)

		self.fc = nn.Linear(lstm_hidden * 2, num_classes) # output: true (0), fake (1), satire (2), bias (3)

		self._init_weights()

	def _init_weights(self):

		# He(Kaiming) for CNN(ReLU)
		nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
		if self.conv.bias is not None:
			nn.init.constant_(self.conv.bias, 0)

		# Xavier for output logits
		nn.init.xavier_normal_(self.fc.weight)
		nn.init.constant_(self.fc.bias, 0)

	def forward(self, text_tokens):

		embedded = self.embedding(text_tokens)	# [batch, seq_len, embed_dim]
		embedded = embedded.permute(0, 2, 1)	# [batch, embed_dim, seq_len]
		
		features = self.conv(embedded)			# [batch, num_filters, seq_len]
		features = self.relu(features)
		features = self.pool(features)			# [batch, num_filters, seq_len // 2]

		# back to LSTM format
		features = features.permute(0,2,1)
		lstm_out, (h_n, c_n) = self.lstm(features)

		# h_n shape: [num_layers * num_directions, batch, hidden_dim]
		last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # [batch, hidden_dim * 2]

		out = self.dropout(last_hidden)
		logits = self.fc(out)
		return logits