# FineTuningPipeline (ElaLoRA)

from loralib.elalora import SVDLinear
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup)


class FineTuningPipeline:
    def __init__(self, dataset, tokenizer, model, optimizer,
                 loss_function=nn.CrossEntropyLoss(), val_size=0.1,
                 epochs=4, seed=42 , allocator=None):
        
        self.allocator = allocator
        self.df_dataset = dataset
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.val_size = val_size
        self.epochs = epochs
        self.seed = seed

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.set_seeds()
        self.freeze_base_weights()  # add freeze gradient
        
        # tokenization + dataloaders
        self.token_ids, self.attention_masks = self.tokenize_dataset()
        self.train_dataloader, self.val_dataloader = self.create_dataloaders()

        #
        self.configure_allocator()

        self.scheduler = self.create_scheduler()
        self.fine_tune()

    # freeze fradient update weight
    def freeze_base_weights(self):
        print("Freezing base model weights (non-SVDLinear layers)...")

        from loralib.elalora import SVDLinear
        for name, module in self.model.named_modules():
            if isinstance(module, SVDLinear):
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
            else:
                for param in module.parameters(recurse=False):
                    param.requires_grad = False

        print("🔎 Checking which parameters are trainable...")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"✅ TRAINING: {name}")
            else:
                print(f"❌ FROZEN:   {name}")

    '''
    def tokenize(self, text):
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=256, # before 128 , 512 , 256
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']
    '''

    def tokenize(self, text, max_len=256, ratio=0.72):
        budget = max_len - 2
        head = int(budget * ratio)
        tail = budget - head

        toks = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(toks) > budget:
            toks = toks[:head] + toks[-tail:]

        # return_tensors=None 
        enc = self.tokenizer.prepare_for_model(
            toks, max_length=max_len, truncation=True, padding='max_length', return_tensors=None
        )
        ids  = torch.tensor(enc["input_ids"], dtype=torch.long).unsqueeze(0)      # (1, L)
        mask = torch.tensor(enc["attention_mask"], dtype=torch.long).unsqueeze(0) # (1, L)
        return ids, mask

    '''
    def tokenize_dataset(self):
        token_ids, attention_masks = [], []
        for review in self.df_dataset['review_cleaned']:
            ids, mask = self.tokenize(review)
            token_ids.append(ids)
            attention_masks.append(mask)
        return torch.cat(token_ids, dim=0), torch.cat(attention_masks, dim=0)
    '''

    def tokenize_dataset(self):
        token_ids, attention_masks = [], []
        for review in self.df_dataset['review_cleaned']:
            ids, mask = self.tokenize(review)        # ids, mask shape = (1, L)
            token_ids.append(ids)
            attention_masks.append(mask)
        token_ids = torch.cat(token_ids, dim=0)          # (N, L)
        attention_masks = torch.cat(attention_masks, 0)  # (N, L)

        # debug safety
        print("shapes:", token_ids.shape, attention_masks.shape)
        return token_ids, attention_masks



    def create_dataloaders(self):
        from sklearn.model_selection import train_test_split
        labels = torch.tensor(self.df_dataset['sentiment_encoded'].values)
        train_ids, val_ids, train_masks, val_masks, train_labels, val_labels = train_test_split(
            self.token_ids, 
            self.attention_masks, 
            labels, 
            test_size=self.val_size, 
            shuffle=True,
            stratify=labels,
            random_state=self.seed
            )

        train_data = TensorDataset(train_ids, train_masks, train_labels)
        val_data = TensorDataset(val_ids, val_masks, val_labels)

        #return DataLoader(train_data, shuffle=True, batch_size=16), DataLoader(val_data, batch_size=16)  # before bacth_size = 32

        train_loader = DataLoader(train_data, shuffle=True,  batch_size=16, num_workers=2, pin_memory=True, drop_last=False)
        val_loader   = DataLoader(val_data,   shuffle=False, batch_size=16, num_workers=2, pin_memory=True)
        return train_loader, val_loader
    
    def configure_allocator(self):
        if self.allocator is None:
            return
        steps_per_epoch = len(self.train_dataloader)
        total_steps = steps_per_epoch * self.epochs
        # 10% / 60% / 10%
        self.allocator.total_step    = total_steps
        self.allocator.init_warmup   = int(0.10 * total_steps)
        self.allocator.final_warmup  = int(0.10 * total_steps)
        self.allocator.mask_interval = max(50, int(0.10 * total_steps))
        print("Allocator:", self.allocator.init_warmup, self.allocator.final_warmup,
          self.allocator.mask_interval, self.allocator.total_step)

    #def create_scheduler(self):
    #    total_steps = self.epochs * len(self.train_dataloader)
    #    return get_linear_schedule_with_warmup(self.optimizer, 0, total_steps)
    
    def create_scheduler(self):
        total_steps = self.epochs * len(self.train_dataloader)
        warmup_steps = int(0.10 * total_steps)  # 6% warmup
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def set_seeds(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def fine_tune(self):
        from datetime import datetime
        print(f"🔍 Model type: {type(self.model)}")
        t0_train = datetime.now()
        global_step = 0  # 🔁 Step counter for ElaLoRA
        
        for epoch in range(self.epochs):
            print(f"\n===== Epoch {epoch+1}/{self.epochs} =====")

            # Training
            self.model.train()
            train_loss = 0
            for batch in self.train_dataloader:
                ids, mask, labels = [x.to(self.device) for x in batch]
                self.model.zero_grad()
                outputs = self.model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                loss.backward()
                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()



                global_step += 1

                # ✅ NEW: Adapt rank if model supports it
                if hasattr(self.model, "maybe_adapt_rank"):
                    self.model.maybe_adapt_rank(global_step=global_step)



            print(f"✅ Avg Train Loss: {train_loss / len(self.train_dataloader):.4f}")

            # Validation
            self.model.eval()
            val_loss, val_accuracy = 0, 0
            t0_val = datetime.now()
            for batch in self.val_dataloader:
                ids, mask, labels = [x.to(self.device) for x in batch]
                with torch.no_grad():
                    outputs = self.model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                val_loss += loss.item()
                val_accuracy += self.calculate_accuracy(logits.cpu().numpy(), labels.cpu().numpy())

            val_time = datetime.now() - t0_val
            print(f"🧪 Avg Val Loss:  {val_loss / len(self.val_dataloader):.4f}")
            print(f"🎯 Val Accuracy: {val_accuracy / len(self.val_dataloader):.4f}")
            print(f"🕒 Val Time:      {val_time}")

        print(f"\n✅ Total training time: {datetime.now() - t0_train}")


    def calculate_accuracy(self, preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        return np.sum(preds_flat == labels.flatten()) / len(labels)
    
    def predict(self, dataloader):
        """Return the predicted probabilities of each class for input text.
        
        Parameters:
            dataloader (torch.utils.data.DataLoader): A DataLoader containing
                the token IDs and attention masks for the text to perform
                inference on.
        
        Returns:
            probs (PyTorch.Tensor): A tensor containing the probability values
                for each class as predicted by the model.

        """

        self.model.eval()
        all_logits = []

        for batch in dataloader:

            batch_token_ids, batch_attention_mask = tuple(t.to(self.device) \
                for t in batch)[:2]

            with torch.no_grad():
                outputs = self.model(batch_token_ids, attention_mask=batch_attention_mask)
                logits = outputs.logits

                #logits = self.model(batch_token_ids, batch_attention_mask)

            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)

        probs = F.softmax(all_logits, dim=1).cpu().numpy()
        return probs
    
