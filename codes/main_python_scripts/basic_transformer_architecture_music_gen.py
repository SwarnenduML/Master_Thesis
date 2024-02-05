import time
import os
from transformers import BertTokenizer, BertModel
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torch
import torch.nn as nn
import numpy as np

print("Imports done")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a simple model using PyTorch's TransformerDecoder
class SimpleTransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, batch_size=32):
        super(SimpleTransformerDecoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model).to(device)
        self.pos_encoder = PositionalEncoding(d_model).to(device)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True), 
            num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.batch_size = batch_size


    def forward(self, tgt, memory, train = True):
        self.training = train
        # print(type(tgt))
        # print(type(self.d_model))
        # Assume tgt shape: (batch_size, sequence_length)
        # tgt = self.embedding(tgt.to(torch.int)) * np.sqrt(self.d_model.to(torch.int))
        tgt = self.embedding(tgt) * np.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        output = self.fc_out(output)
        # During training, return the output for loss computation
        if self.training:
            return output
        else:
            # During inference, return probabilities using softmax
            return torch.nn.functional.softmax(output, dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to(device)
        self.position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(self.position * self.div_term)
        self.encoding[:, 1::2] = torch.cos(self.position * self.div_term)
        # self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        # Assume x shape: (batch_size, sequence_length, d_model)
        batch_size, sequence_length, _ = x.size()
        self.encoding = self.encoding[:sequence_length, :].expand(batch_size, -1, -1)
        return x + self.encoding

class BERTSentenceEncoder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)

    def encode_sentences(self, input_sentences, atn_mask):
        # tokenized_input = self.tokenizer(input_sentences, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(input_sentences, attention_mask = atn_mask)
        encoded_sentences = outputs.last_hidden_state

        # take only the CLS mode
        context_vector = encoded_sentences[:, 0,:]
        return encoded_sentences, context_vector

print("Classes define")


note_mapping = {'0': 1, 'C0': 2, 'C0#': 3, 'D0': 4, 'D0#': 5, 'E0': 6, 'F0': 7, 'F0#': 8, 'G0': 9, 'G0#': 10, 'A0': 11, 'A0#': 12, 'B0': 13,
                'C1': 14, 'C1#': 15, 'D1': 16, 'D1#': 17, 'E1': 18, 'F1': 19, 'F1#': 20, 'G1': 21, 'G1#': 22, 'A1': 23, 'A1#': 24, 'B1': 25,
                'C2': 26, 'C2#': 27, 'D2': 28, 'D2#': 29, 'E2': 30, 'F2': 31, 'F2#': 32, 'G2': 33, 'G2#': 34, 'A2': 35, 'A2#': 36, 'B2': 37,
                'C3': 38, 'C3#': 39, 'D3': 40, 'D3#': 41, 'E3': 42, 'F3': 43, 'F3#': 44, 'G3': 45, 'G3#': 46, 'A3': 47, 'A3#': 48, 'B3': 49,
                'C4': 50, 'C4#': 51, 'D4': 52, 'D4#': 53, 'E4': 54, 'F4': 55, 'F4#': 56, 'G4': 57, 'G4#': 58, 'A4': 59, 'A4#': 60, 'B4': 61,
                'C5': 62, 'C5#': 63, 'D5': 64, 'D5#': 65, 'E5': 66, 'F5': 67, 'F5#': 68, 'G5': 69, 'G5#': 70, 'A5': 71, 'A5#': 72, 'B5': 73,
                'C6': 74, 'C6#': 75, 'D6': 76, 'D6#': 77, 'E6': 78, 'F6': 79, 'F6#': 80, 'G6': 81, 'G6#': 82, 'A6': 83, 'A6#': 84, 'B6': 85,
                'C7': 86, 'C7#': 87, 'D7': 88, 'D7#': 89, 'E7': 90, 'F7': 91, 'F7#': 92, 'G7': 93, 'G7#': 94, 'A7': 95, 'A7#': 96, 'B7': 97,
                'C8': 98, 'C8#': 99, 'D8': 100, 'D8#': 101, 'E8': 102, 'F8': 103, 'F8#': 104, 'G8': 105, 'G8#': 106, 'A8': 107, 'A8#': 108, 'B8': 109,'-1':100 }
reverse_note_mapping = {v: k for k, v in note_mapping.items()}

# create a function to read all the data in a given folder
def read_all_files(folder_path):
    if not os.path.exists(folder_path):
        raise Exception("Folder doesnot exist")
    
    # Get a list of all Excel files in the folder
    excel_files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx') or file.endswith('.xls')]

    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # Loop through each Excel file and read it into a DataFrame
    for file in excel_files:
        # Assuming that all sheets in each Excel file need to be concatenated
        xls = pd.ExcelFile(os.path.join(folder_path, file))
        sheet_names = xls.sheet_names
        for sheet_name in sheet_names:
            df = pd.read_excel(xls, sheet_name)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    if combined_df.columns[0]=='Unnamed: 0':
        combined_df = combined_df.drop('Unnamed: 0', axis=1)
    
    for c in combined_df.columns:
        combined_df[c] = combined_df[c].replace('',' ', regex=True)

    training_words = combined_df['words']
    training_words = [sentence.replace(';', ' ') for sentence in training_words]
    training_words[0] = '<BOS> ' + training_words[0]
    training_words[-1] = training_words[-1] + ' <EOS>'
    training_labels = [[note_mapping[note] for note in d.split(' ; ')] for d in combined_df['mean_note_crepe']]

    return training_labels,training_words

training_labels,training_words = read_all_files("/speech/dbwork/mul/spielwiese4/students/desengus/dry_crepe_pesto/excels/train/")


model_name = 'bert-base-uncased'
max_length = 200
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

for item_no,i in enumerate(training_labels):
    # print((training_labels[item_no]))
    # print(type((training_labels[item_no])))
    if len(i)<max_length:
        if len(training_labels[item_no]) < max_length:
            training_labels[item_no].extend([100] * (max_length - len(training_labels[item_no]))) # adding EOS or -1 to end of song


df = pd.DataFrame({'word':training_words, 'label':training_labels})

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=64, train=True):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['word']
        if self.train == True:
            label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)

        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')

        if self.train == True:
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'label': label
            }
        else:
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze()
            }


sample_text = ['hello, its me. I love you for thee','i love you for a thousand years. I love you for a thousand more.']
input_text = ['I laugh, sometimes cry, do both and don\'t  know why Touching it all And that\'s just the way things are You me could give a whirl, but I\'m wanting you, boy, an emotional girl.',
                'girl. I\'m an emotional girl I can\'t help myself Sometimes laugh,']
val_text = sample_text+input_text


train_ds = CustomDataset(df, tokenizer, max_length)

# Batch size
batch_size = 16

# Create data loaders
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last = True)
print("Dataset done")

# Parameters
vocab_size = len(note_mapping)  # As there are 108 notes, 1 silence
d_model = 768  # has to be the BERT encoder output size
nhead = 8  # ensure d_model is divisible by nhead
num_layers = 64
dim_feedforward = 768

# Model
encoder = BERTSentenceEncoder()
decoder = SimpleTransformerDecoderModel(vocab_size, d_model, nhead, num_layers, dim_feedforward).to(device)

# Set up optimizer and loss function
optimizer = optim.AdamW(decoder.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

MAX_SEQ_LENGTH = 200

print("Training start")


num_epochs = 100
cal_per_epoch = 5
# img_batch = [np.zeros((int(num_epochs/cal_per_epoch),3,len(note_mapping),MAX_SEQ_LENGTH+1))]
# image_df['images'] = [img_batch[0]] * len(image_df)

# writer = SummaryWriter(log_dir='logs')
for epoch in range(num_epochs):
    start_time = time.time()
    decoder.train()
    total_loss = 0

    for batch in train_loader:
        encoded, melody, attention_mask = batch['input_ids'].to(device), batch['label'].to(device),batch['attention_mask'].to(device)
        # print(type(encoded))
        # print(type(melody))
        # print(type(attention_mask))
        encoded, _ = encoder.encode_sentences(encoded, attention_mask)
        encoded = encoded.to(device)
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = decoder(melody, encoded)

        # Reshape predictions to match the shape of targets
        predictions = outputs.view(-1, vocab_size)
        targets = melody.view(-1)

        # Define the CrossEntropyLoss criterion
        criterion = nn.CrossEntropyLoss()

        # Compute the loss
        loss = criterion(predictions, targets)
        # breakpoint()

        # print("Categorical CrossEntropy Loss:", loss.item())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch + 1}, Time taken: {epoch_time:.2f} seconds")



    decoder.eval()  # Set the model to evaluation mode
    val_loss = 0

    if epoch % cal_per_epoch == 0:
        with torch.no_grad():
            for i,val_txt in enumerate(val_text):
                val_batch = tokenizer(val_txt, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
                val_encoded, val_attention_mask = val_batch['input_ids'].to(device), val_batch['attention_mask'].to(device)

                # Encode sentences
                val_encoded, _ = encoder.encode_sentences(val_encoded, val_attention_mask).to(device)

                # Initialize the starting token for the decoder input
                start_token = torch.tensor([[1]], dtype=torch.long)

                # Store the predicted sequence for each example in the batch
                batch_predictions = []

                # Iterate over the maximum sequence length or a predefined maximum length
                for step in range(MAX_SEQ_LENGTH):
                    #print("Input Size:", val_encoded.size())
                    # Predict the next token
                    with torch.no_grad():
                        output = decoder(start_token, val_encoded)
                    
                    # Get the predicted token (greedy decoding)
                    pred_token = output.argmax(2)[:, -1].unsqueeze(1)
                    
                    # Append the predicted token to the output sequence
                    start_token = torch.cat((start_token, pred_token), dim=1)
                    
                
                # Add the batch predictions to the overall predictions list
                print(val_txt)
                print(start_token)

                # img_tensor = start_token.numpy()[0]
                # image_df['images'][i][epoch//cal_per_epoch,0] = img_tensor
                # image_df['images'][i][epoch//cal_per_epoch,1] = img_tensor
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation done")

        torch.save(decoder.state_dict(), '/speech/dbwork/mul/spielwiese4/students/desengus/codes/main_python_scripts/checkpoints/epoch'+str(epoch))

    
    
 
