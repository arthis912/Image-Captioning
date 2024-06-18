#Arthi Seetharaman - ECSE 6850

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor
from torchtext.data.utils import get_tokenizer
import time
from bleu import get_bleu
from MyTransformer import MyTransformer
import matplotlib.pyplot as plt

#Class to create dataset for training and val
class FlickrDataset(Dataset):

    def __init__(self, file_path, img_dir, vocab, transform = None):

        with open(file_path, 'r') as file:

            self.data = json.load(file)['images']

        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        self.images = {}

        #Preprocess and store all images
        self.preprocess_images()

    def preprocess_images(self):

        for item in self.data:

            img_path = os.path.join(self.img_dir, item['filename'])

            img = Image.open(img_path)

            if self.transform:

                img = self.transform(img)

            self.images[item['filename']] = img

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]
        img = self.images[item['filename']]  # Retrieve preprocessed image from the dictionary

        #Process captions and select the longest one for training purposes
        captions = [self.parse_tokens(caption['tokens']) for caption in item['sentences']]
        longest_caption = max(captions, key=len)

        return img, longest_caption

    def parse_tokens(self, tokens):

        #Initialize the sequence with the start-of-sentence token
        indices = [self.vocab['<bos>']]

        #Convert list of tokens into list of indices from the vocabulary
        for token in tokens:

            if token in self.vocab:

                indices.append(self.vocab[token])

            else:

                indices.append(self.vocab['<unk>'])

        #Append end-of-sentence token
        indices.append(self.vocab['<eos>'])

        return indices

#Class to create dataset for testing
class FlickrTestDataset(Dataset):

    def __init__(self, file_path, img_dir, vocab, transform = None):

        with open(file_path, 'r') as file:

            self.data = json.load(file)['images']

        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        self.images = {}

        #Preprocess and store all images
        self.preprocess_images()

    def preprocess_images(self):

        for item in self.data:

            img_path = os.path.join(self.img_dir, item['filename'])
            img = Image.open(img_path)

            if self.transform:

                img = self.transform(img)

            self.images[item['filename']] = img

    def __len__(self):

        return len(self.data) * 5

    def __getitem__(self, idx):

        #Index for which image and which caption this index corresponds to
        image_num = idx // 5
        caption_num = idx % 5

        item = self.data[image_num]
        img = self.images[item['filename']]

        #Get the specific caption for this index
        caption_data = item['sentences'][caption_num]
        tokens = self.parse_tokens(caption_data['tokens'])
        caption = tokens

        return img, caption

    def parse_tokens(self, tokens):

        #Initialize the sequence with the start-of-sentence token
        indices = [self.vocab['<bos>']]

        #Convert list of tokens into list of indices from the vocabulary
        for token in tokens:

            if token in self.vocab:

                indices.append(self.vocab[token])

            else:

                indices.append(self.vocab['<unk>'])

        #Append end-of-sentence token
        indices.append(self.vocab['<eos>'])

        return indices


#Create batch function
def create_batch(data_batch, PAD_IDX = 1):

    #seperating images and captions
    imgs, captions = zip(*data_batch)

    #Stacking images
    imgs = torch.stack(imgs)

    #getting max length of captions
    max_len = max(len(c) for c in captions)

    #Padding captions
    padded_captions = torch.full((len(captions), max_len), PAD_IDX, dtype=torch.long)

    for i, c in enumerate(captions):
        padded_captions[i, :len(c)] = torch.tensor(c, dtype=torch.long)

    return imgs, padded_captions

#Create batch function for test dataset which has all 5 captions
def create_batch_test(data_batch, PAD_IDX=1):

    #seperating images and captions
    imgs, captions = zip(*data_batch)

    #Stacking images
    imgs = torch.stack(imgs)

    #getting max length of captions
    max_len = max(len(c) for caps in captions for c in caps)

    #Padding captions
    padded_captions = torch.full((len(captions), max_len), PAD_IDX, dtype=torch.long)

    for i, cap_set in enumerate(captions):

        all_caps = [token for cap in cap_set for token in cap]
        all_caps = all_caps[:max_len]

        padded_captions[i, :len(all_caps)] = torch.tensor(all_caps, dtype=torch.long)

    return imgs, padded_captions

#Testing function for each epoch
def epoch_testing(model, test_iter, vocab, t_len):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    model.eval()

    #Initializing vars
    start_symbol = vocab['<bos>']
    score = 0
    PAD_IDX = torch.tensor(1.0)

    #Looping through the testing dataloader
    for imgs, captions in test_iter:

        imgs = imgs.to(device)

        #Intializing sentence with size of 1, 50 since the batch size is 50
        sentence = torch.full((1, 50), start_symbol, dtype=torch.int64)

        for i in range(captions.size()[1]):

            #Getting predicted sentence
            pred_sent = model(imgs, sentence, PAD_IDX)
            pred_sent = torch.argmax(pred_sent, dim=2)
            sentence = torch.cat((sentence, pred_sent[-1:]), dim=0)

        pred = torch.transpose(sentence, 0, 1)[:, 1:]
        actual = captions[:, 1:]

        p_len = pred.size()[1]

        #Untokenizing the predicted and actual senteces
        for i in range(p_len):

            h = untokenize(pred[i], vocab)
            r = untokenize(actual[i], vocab)

            #Calculating bleu score
            score += get_bleu(hypotheses = h.split(), reference = r.split())

        bleu_score = score/p_len
        #Returning average bleu score
        return (bleu_score)

#Function to untokenize the sentences
def untokenize(indices, vocab):

    #Mapping words to indices
    all_words = {index: word for word, index in vocab.items()}

    eos_index = vocab.get('<eos>', None)
    pad_index = vocab.get('<pad>', None)

    words = []

    for i in indices:

        i = int(i)
        #If there is an eos or pad index, break
        if i == eos_index or i == pad_index:
            break

        word = all_words.get(i, '<unk>')

        if i in all_words:

            words.append(word)

    result = ' '.join(words)

    return result

#Training function
def training():

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_default_device(device)

  #Transform for images
  transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
  ])

  #Intializing spacy vocab
  vocab_path = "English_vocab.pth"
  tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
  vocab = torch.load(vocab_path, tokenizer).stoi

  #All file paths
  train_path = 'training_data.json'
  val_path = 'val_data.json'
  test_path = 'test_data.json'
  images_path = 'Images'
  
  #Creating datasets
  train_dataset = FlickrDataset(train_path, images_path, vocab, transform)
  val_dataset = FlickrDataset(val_path, images_path, vocab, transform)
  test_dataset = FlickrTestDataset(test_path, images_path, vocab, transform)
  
  #Creating data loaders
  batch_size = 50
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=create_batch, generator=torch.Generator(device=device))
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=create_batch, generator=torch.Generator(device=device))
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=create_batch_test, generator=torch.Generator(device=device))

  #Initializing params for transformer
  NUM_ENCODER_LAYERS = 4
  NUM_DECODER_LAYERS = 4
  EMB_DIM = 512
  NHEAD = 8
  VOCAB_SIZE = len(vocab)
  FFN_HID_DIM = 512
  DROPOUT = 0.1
  NUM_CLASSES = 10000
  PATCH_SIZE = 16

  #Intializing model
  model = MyTransformer(PATCH_SIZE, NUM_CLASSES, EMB_DIM, NUM_ENCODER_LAYERS,
                        NUM_DECODER_LAYERS, NHEAD, FFN_HID_DIM, DROPOUT, VOCAB_SIZE)
  model.to(device)

  #Optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
  PAD_IDX = vocab['<pad>']

  #Loss function
  loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

  #Initializing variables
  num_epochs = 15
  train_losses = []
  val_losses = []
  best_val_loss = float('inf')
  bleu_scores = []

  #Starting training
  for epoch in range(num_epochs):

    start_time = time.time()

    model.train()
    training_loss = 0

    for imgs, caption in train_loader:

        imgs = imgs.to(device)

        trans_cap = caption.transpose(0, 1)
        tgt_input = trans_cap[:-1, :]
        tgt_out = trans_cap[1:, :]

        optimizer.zero_grad()

        logits = model(imgs, tgt_input, PAD_IDX=PAD_IDX)

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    #Saving training loss
    epoch_training_loss = training_loss/len(train_loader)
    train_losses.append(epoch_training_loss)

    #Starting validation
    model.eval()
    val_loss = 0

    with torch.no_grad():

      for imgs, captions in val_loader:

        imgs = imgs.to(device)

        trans_cap = captions.transpose(0, 1)
        tgt_input = trans_cap[:-1, :]
        tgt_out = trans_cap[1:, :]

        logits = model(imgs, tgt_input, PAD_IDX=PAD_IDX)

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        val_loss += loss.item()

      #Saving validation loss
      epoch_val_loss = val_loss/len(val_loader)
      val_losses.append(epoch_val_loss)

      if val_loss < best_val_loss:
          best_val_loss = val_loss
          torch.save(model.state_dict(), 'saved_model.pt')

    #Testing starts
    epoch_bleu_score = epoch_testing(model, test_loader, vocab, len(test_dataset))
    bleu_scores.append(epoch_bleu_score)

    end_time = time.time()
    elapsed_time = end_time - start_time

    #Print values for the epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_training_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Time: {elapsed_time:.2f} sec')
    print(f'Average BLEU score: {epoch_bleu_score}')

  plt.plot(range(num_epochs), train_losses, label='Training Loss')
  plt.plot(range(num_epochs), val_losses, label='Validation Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training and Validation Losses')
  plt.savefig('loss.png')
  plt.legend()
  plt.show()

  plt.figure(figsize=(10, 5))
  plt.plot(range(1, len(bleu_scores) + 1), bleu_scores, label='BLEU Score')
  plt.title('BLEU Scores vs. Epochs')
  plt.xlabel('Epoch')
  plt.ylabel('BLEU Score')
  plt.legend()
  plt.grid(True)
  plt.savefig('bleu.png')
  plt.show()

#Testing Function
def testing():

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.set_default_device(device)

  #Intializing spacy vocab
  vocab_path = "English_vocab.pth"
  tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
  vocab = torch.load(vocab_path, tokenizer).stoi

  #Transform for images
  transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
  ])

  #Intializing spacy vocab
  vocab_path = "English_vocab.pth"
  tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
  vocab = torch.load(vocab_path, tokenizer).stoi

  #All file paths
  test_path = 'test_data.json'
  images_path = 'Images'

  #Creating datasets
  test_dataset = FlickrTestDataset(test_path, images_path, vocab, transform)

  batch_size = 50
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=create_batch_test, generator=torch.Generator(device=device))

  #Initializing params for transformer
  NUM_ENCODER_LAYERS = 4
  NUM_DECODER_LAYERS = 4
  EMB_DIM = 512
  NHEAD = 8
  VOCAB_SIZE = len(vocab)
  FFN_HID_DIM = 512
  DROPOUT = 0.1
  NUM_CLASSES = 10000
  PATCH_SIZE = 16

  #Intializing model
  model = MyTransformer(PATCH_SIZE, NUM_CLASSES, EMB_DIM, NUM_ENCODER_LAYERS,
                        NUM_DECODER_LAYERS, NHEAD, FFN_HID_DIM, DROPOUT, VOCAB_SIZE)
  model.to(device)

  model.load_state_dict(torch.load('saved_model.pt'))

  bleu_score = epoch_testing(model, test_loader, vocab, len(test_dataset))
  print(bleu_score)

if __name__ == '__main__':
    training()
    testing()