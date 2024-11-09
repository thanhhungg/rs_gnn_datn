import torch
import pickle
import numpy as np
from model import SessionGraph, trans_to_cuda
from utils import Data

def load_model(model_path, opt, n_node):
    model = trans_to_cuda(SessionGraph(opt, n_node))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def predict_next_item(model, session):
    alias_inputs, A, items, mask, n_node = prepare_input(session)
    
    hidden = model.embedding(torch.arange(n_node).to(items.device)).unsqueeze(0)  # [1, n_node, hidden_size]
    hidden = model.gnn(A, hidden)
    
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    scores = model.compute_scores(seq_hidden, mask)
    
    _, indices = torch.topk(scores, k=10)
    return indices.cpu().numpy()[0]  # Trả về kết quả cho session đầu tiên

def prepare_input(session):
    inputs = np.array([session])
    items = np.unique(inputs)
    n_node = max(items.max() + 1, len(items))  # Đảm bảo n_node đủ lớn
    
    alias_inputs = [np.array([np.where(items == i)[0][0] for i in seq]) for seq in inputs]
    
    A = np.zeros((len(inputs), n_node, 2*n_node))
    for i, seq in enumerate(inputs):
        for j in range(len(seq) - 1):
            u = np.where(items == seq[j])[0][0]
            v = np.where(items == seq[j + 1])[0][0]
            A[i, u, v] = 1
            A[i, u, v + n_node] = 1  # Thêm kết nối ngược
    
    alias_inputs = trans_to_cuda(torch.LongTensor(np.array(alias_inputs)))
    A = trans_to_cuda(torch.FloatTensor(A))
    items = trans_to_cuda(torch.LongTensor(items))
    mask = trans_to_cuda(torch.BoolTensor([[True] * len(session)]))
    
    return alias_inputs, A, items, mask, n_node

# Tải tham số và mô hình
try:
    opt = pickle.load(open('opt.pkl', 'rb'))
    if hasattr(opt, 'n_node'):
        n_node = opt.n_node
    else:
        n_node = 310  # Giá trị mặc định cho dataset 'sample'
except FileNotFoundError:
    import argparse
    opt = argparse.Namespace()
    opt.hiddenSize = 100
    opt.batchSize = 100
    opt.step = 1
    opt.nonhybrid = False
    n_node = 310  # Giá trị mặc định cho dataset 'sample'

model = load_model('sr_gnn_model.pth', opt, n_node)

# Ví dụ về một phiên mua sắm
example_session = [13, ]  # ID của các item trong phiên

# Dự đoán item tiếp theo
next_items = predict_next_item(model, example_session)
print("Các item được đề xuất tiếp theo:", next_items)