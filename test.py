import torch
from pytorch_transformers import BertModel
from sklearn import metrics
from torch.utils.data import DataLoader

from data_utils import Tokenizer4Bert, ABSADataset
from models.aen import AEN_BERT
from utils import get_options

opt = get_options()
bert = BertModel.from_pretrained(opt.pretrained_bert_name)

model_path = 'state_dict/aen_bert_laptop_val_acc0.7821'
model = AEN_BERT(bert, opt)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
test_set = ABSADataset(opt.dataset_file['test'], tokenizer)
data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

n_correct, n_total = 0, 0
t_targets_all, t_outputs_all = None, None

with torch.no_grad():
    for t_batch, t_sample_batched in enumerate(data_loader):
        t_inputs = [t_sample_batched[col].to(opt.device) for col in opt.inputs_cols]
        print("input: ", t_inputs)
        t_targets = t_sample_batched['polarity'].to(opt.device)
        print("targets: ", t_targets)
        t_outputs = model(t_inputs)
        print("outputs: ", t_outputs)

        n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
        n_total += len(t_outputs)

        if t_targets_all is None:
            t_targets_all = t_targets
            t_outputs_all = t_outputs
        else:
            t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
            t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
acc = n_correct / n_total
f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                      average='macro')
