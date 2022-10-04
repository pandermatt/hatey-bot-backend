import torch
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from tqdm import trange
from torch.nn import BCEWithLogitsLoss

class ToxicCommentClassifier:
    def __init__(self, model, tokenizer, label_columns, threshold=0.50):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.idx2label = dict(zip(range(len(label_columns)),label_columns))
        self.num_labels = len(label_columns)
        self.label_columns = label_columns
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'gamma', 'beta']
        self.optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01}, {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
                                             ]

    def train(self, optimizer, train_dataloader, val_dataloader=None, epochs=4, lr=2e-5, eps=1e-8):
        train_loss_set = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer =  optimizer(self.optimizer_grouped_parameters, lr=lr, eps=eps)

        for _ in trange(epochs, desc="Epoch"):
            self.model.train()

            tr_loss = 0 #running loss
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                optimizer.zero_grad()

                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs[0]
                loss_func = BCEWithLogitsLoss()
                loss = loss_func(logits.view(-1,self.num_labels),b_labels.type_as(logits).view(-1,self.num_labels)) #convert labels to float for calculation
                train_loss_set.append(loss.item())

                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss/nb_tr_steps))

        if val_dataloader:
            self.validation(val_dataloader)


    def validation(self, val_dataloader):

        self.model.eval()

        logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

        for i, batch in enumerate(val_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.sigmoid(b_logit_pred)

                b_logit_pred = b_logit_pred.detach().cpu().numpy()
                pred_label = pred_label.cpu().numpy()
                b_labels = b_labels.cpu().numpy()

            tokenized_texts.append(b_input_ids)
            logit_preds.append(b_logit_pred)
            true_labels.append(b_labels)
            pred_labels.append(pred_label)

        pred_labels = [item for sublist in pred_labels for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]

        threshold = 0.50
        pred_bools = [pl>threshold for pl in pred_labels]
        true_bools = [tl==1 for tl in true_labels]
        val_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')*100
        val_flat_accuracy = accuracy_score(true_bools, pred_bools)*100

        print('F1 Validation Accuracy: ', val_f1_accuracy)
        print('Flat Validation Accuracy: ', val_flat_accuracy)

    def test(self, test_dataloader):
        self.model.eval()

        logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

        for i, batch in enumerate(test_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.sigmoid(b_logit_pred)

                b_logit_pred = b_logit_pred.detach().cpu().numpy()
                pred_label = pred_label.cpu().numpy()
                b_labels = b_labels.cpu().numpy()

            tokenized_texts.append(b_input_ids)
            logit_preds.append(b_logit_pred)
            true_labels.append(b_labels)
            pred_labels.append(pred_label)

        pred_labels = [item for sublist in pred_labels for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]
        true_bools = [tl==1 for tl in true_labels]
        pred_bools = [pl>0.50 for pl in pred_labels]

        print('Test F1 Accuracy: ', f1_score(true_bools, pred_bools,average='micro'))
        print('Test Flat Accuracy: ', accuracy_score(true_bools, pred_bools),'\n')
        clf_report = classification_report(true_bools,pred_bools,target_names=self.label_columns)
        pickle.dump(clf_report, open('classification_report.txt','wb'))
        print(clf_report)

    def save_modle(self, path, name):
        torch.save(self.model, path+name)

    def load_model(self, path, name):
        self.model = torch.load(path+name)

    def predict(self, sentence):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        self.model.eval()
        with torch.no_grad():
            outs = self.model(tokens_tensor, token_type_ids=None, attention_mask=None)
            logit_pred = outs[0]
            pred_label = torch.sigmoid(logit_pred)
            pred_label = pred_label.cpu().numpy()
            pred_label = pred_label.flatten().tolist()
            pred_label = [pl>self.threshold for pl in pred_label]
            pred_label = [self.idx2label[idx] for idx, val in enumerate(pred_label) if val]
        return pred_label