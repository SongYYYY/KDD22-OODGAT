import sys
import argparse
import torch
from scipy.stats import entropy
import torch.nn.functional as F
import torch.nn as nn
# pyg imports
from torch_geometric.datasets import Planetoid, WikiCS, Amazon, Coauthor, LastFMAsia, NELL, CitationFull, GNNBenchmarkDataset
from torch_geometric.transforms import NormalizeFeatures, Compose
from torch_geometric.utils import homophily
sys.path.append("../..")
from metrics import get_acc, get_ood_performance, get_f1_score
from utils import EntropyLoss, get_consistent_loss_new, cosine_similarity, CE_uniform, seed_torch
from data_process import generate_masks
from model import OODGAT



def train(args):

    dataset_str = args.dataset.split('_')[0]
    # transforms
    trans = []
    continuous = args.continuous
    if not continuous:
        trans.append(NormalizeFeatures())

    trans = Compose(trans)

    if dataset_str == 'wiki-CS':
        dataset = WikiCS(root='pyg_data/wiki-CS', transform=trans)
    elif dataset_str == 'LastFMAsia':
        dataset = LastFMAsia(root='pyg_data/LastFMAsia', transform=trans)
    elif dataset_str == 'NELL':
        dataset = NELL(root='pyg_data/NELL', transform=trans)
    elif dataset_str == 'Amazon':
        dataset_name = args.dataset.split('_')[1]
        dataset = Amazon(root='pyg_data/Amazon', name=dataset_name, transform=trans)
    elif dataset_str == 'Coauthor':
        dataset_name = args.dataset.split('_')[1]
        dataset = Coauthor(root='pyg_data/Coauthor', name=dataset_name, transform=trans)
    elif dataset_str == 'CitationFull':
        dataset_name = args.dataset.split('_')[1]
        dataset = CitationFull(root='pyg_data/CitationFull', name=dataset_name, transform=trans)
    elif dataset_str == 'GNNBenchmark':
        dataset_name = args.dataset.split('_')[1]
        dataset = GNNBenchmarkDataset(root='pyg_data/GNNBenchmark', name=dataset_name, transform=trans)
    elif dataset_str == 'Planetoid':
        dataset_name = args.dataset.split('_')[1]
        dataset = Planetoid(root='pyg_data/Planetoid', name=dataset_name, transform=trans)
    else:
        raise Exception('unknown dataset.')

    data = dataset[0]
    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print(f'homophily: {homophily(data.edge_index, data.y)}')

    # dataset parameters
    ID_classes = args.ID_classes
    splits = args.splits
    n_samples_per_class = splits[0]
    val_size_per_class = splits[1]
    test_size = splits[2]
    ID_mask, OOD_mask, train_mask_ID, val_mask_ID, test_mask_ID, train_mask_OOD, val_mask_OOD, test_mask_OOD, \
    detection_mask_val, detection_y_val, joint_y_val, detection_mask_test, detection_y_test, joint_y_test, ID_y = \
        generate_masks(data, ID_classes, n_samples_per_class, val_size_per_class, test_size, args.random_seed_data)

    assert (ID_mask | OOD_mask).sum() == data.num_nodes
    assert (train_mask_ID | val_mask_ID | test_mask_ID | train_mask_OOD | val_mask_OOD | test_mask_OOD).sum() == \
           train_mask_ID.sum() + val_mask_ID.sum() + test_mask_ID.sum() + train_mask_OOD.sum() + val_mask_OOD.sum() + test_mask_OOD.sum()
    assert detection_mask_val.sum() == len(detection_y_val) == len(joint_y_val)
    assert detection_mask_test.sum() == len(detection_y_test) == len(joint_y_test)
    assert train_mask_ID.sum() == train_mask_OOD.sum() == n_samples_per_class * len(ID_classes)
    assert val_mask_ID.sum() == val_mask_OOD.sum() == val_size_per_class * len(ID_classes)
    assert test_mask_ID.sum() == test_mask_OOD.sum() == test_size

    data.train_mask, data.val_mask, data.test_mask = train_mask_ID, val_mask_ID, test_mask_ID
    data.y = ID_y

    print('ID size: {}, OOD size: {}, total size: {}.'.format(ID_mask.sum(), OOD_mask.sum(), data.num_nodes))
    print('train%ID: {:.2%}, val%ID: {:.2%}, test%ID: {:.2%}.'.format(train_mask_ID.sum() / ID_mask.sum(),
                                                                      val_mask_ID.sum() / ID_mask.sum(),
                                                                      test_mask_ID.sum() / ID_mask.sum()))
    print('train%OOD: {:.2%}, val%OOD: {:.2%}, test%OOD: {:.2%}.'.format(train_mask_OOD.sum() / OOD_mask.sum(),
                                                                         val_mask_OOD.sum() / OOD_mask.sum(),
                                                                         test_mask_OOD.sum() / OOD_mask.sum()))
    device = torch.device('cuda')
    data = data.to(device)




    # inline help functions
    def init_oodgat():
        # fitlog.set_rng_seed(args.random_seed_model)
        seed_torch(args.random_seed_model)
        # init model
        in_dim = data.x.shape[1]
        out_dim = len(ID_classes)
        if args.model_name == 'OODGAT':
            model = OODGAT(in_dim, args.hidden_dim, out_dim, args.heads, False, args.drop_edge, True,
                           args.drop_prob, True, args.drop_input).to(device)
        else:
            print('unknown model: {}.'.format(args.model_name))
            raise Exception('unknown model')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        return model, optimizer

    def train_oodgat(model, optimizer):
        a = torch.tensor(0.9).to(device)
        b = torch.tensor(0.01).to(device)
        # train
        best_t = args.epochs - 1
        best_metric = 0
        patience = 0
        xent = nn.CrossEntropyLoss()
        ent_loss_func = EntropyLoss(reduction=False)
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = torch.zeros(1).to(device)
            logits, att = model(data, return_attention_weights=True)
            if args.w_consistent is not None and args.w_consistent > 0:
                ent_loss = ent_loss_func(logits)  # ent_loss: N-dim tensor
                cos_loss_1 = get_consistent_loss_new(att[0].T, (ent_loss - ent_loss.mean()) / ent_loss.std(),
                                                        f1=F.sigmoid, f2=F.sigmoid)
                cos_loss_2 = get_consistent_loss_new(att[1].T, (ent_loss - ent_loss.mean()) / ent_loss.std(),
                                                        f1=F.sigmoid, f2=F.sigmoid)
                consistent_loss = 0.5 * (cos_loss_1 + cos_loss_2)
                loss += torch.pow(a, b * epoch) * args.w_consistent * consistent_loss
            if args.w_discrepancy is not None and args.w_discrepancy > 0:
                loss -= torch.pow(a, b * epoch) * args.w_discrepancy * cosine_similarity(att[0].mean(axis=1), att[1].mean(axis=1))
            if args.w_ent is not None and args.w_ent > 0:
                loss += torch.pow(a, b * epoch) * args.w_ent * local_ent_loss(logits, att, len(ID_classes), args.margin)

            sup_loss = xent(logits[data.train_mask], data.y[data.train_mask])
            loss += sup_loss
            loss.backward()
            optimizer.step()
            # validate
            if epoch % 10 == 0:
                model.eval()
                logits, att = model(data, return_attention_weights=True)
                preds = logits.argmax(axis=1).detach()
                val_acc = get_acc(data.y, preds, data.val_mask)

                ATT = F.sigmoid(torch.hstack([att[0].detach(), att[1].detach()]).mean(axis=1)).cpu()
                auroc, aupr_0, aupr_1, fprs = get_ood_performance(detection_y_val, ATT,
                                                                          detection_mask_val)

                print('epoch: {}, loss: {}, val_acc: {}, auroc:{}.'.format(epoch + 1, loss.item(), val_acc.item(),
                                                                               auroc))

                current_metric = val_acc + auroc
                if current_metric > best_metric:
                    best_t = epoch
                    patience = 0
                    best_metric = current_metric
                    torch.save(model.state_dict(), 'best_oodgat.pkl')
                else:
                    patience += 1
                    if patience > 20:
                        break

        return best_metric, best_t

    def evaluate_oodgat(model):
        # evaluate
        model.load_state_dict(torch.load('best_oodgat.pkl'))
        model.eval()
        # classification
        logits, a = model(data, return_attention_weights=True)
        preds = logits.argmax(axis=1).detach()
        a = [a[0].detach(), a[1].detach()]
        test_acc = get_acc(data.y, preds, data.test_mask)
        print('test_acc:{}'.format(test_acc.item()))

        # OOD detection
        pred_dist = F.softmax(logits, dim=1).detach().cpu()
        ENT = entropy(pred_dist, axis=1)
        ATT = F.sigmoid(torch.hstack([a[0], a[1]]).mean(axis=1)).cpu()

        auroc_ENT, aupr_0_ENT, aupr_1_ENT, fprs_ENT = get_ood_performance(detection_y_test, ENT,
                                                                          detection_mask_test)
        auroc_ATT, aupr_0_ATT, aupr_1_ATT, fprs_ATT = get_ood_performance(detection_y_test, ATT,
                                                                          detection_mask_test)

        print('detection via ENT: auroc:{}, aupr_0:{}, aupr_1:{}, fpr95:{}'.format(auroc_ENT, aupr_0_ENT, aupr_1_ENT,
                                                                                 fprs_ENT[2]))
        print('detection via ATT: auroc:{}, aupr_0:{}, aupr_1:{}, fpr95:{}'.format(auroc_ATT, aupr_0_ATT, aupr_1_ATT,
                                                                                 fprs_ATT[2]))

        # N+1 clf
        f1_list_ent = get_f1_score(torch.tensor(joint_y_test), preds.detach().cpu(), ENT, 'ent',
                                   detection_mask_test, len(ID_classes))
        f1_list_att = get_f1_score(torch.tensor(joint_y_test), preds.detach().cpu(), ATT, 'att',
                                   detection_mask_test, len(ID_classes))

        micro_f1_best_ent = max([f['micro'] for f in f1_list_ent])
        macro_f1_best_ent = max([f['macro'] for f in f1_list_ent])
        weighted_f1_best_ent = max([f['weighted'] for f in f1_list_ent])
        print('N+1 clf (best_thres f1) via ENT: micro: {}, macro: {}, weighted: {}.'.format(micro_f1_best_ent,
                                                                                            macro_f1_best_ent,
                                                                                            weighted_f1_best_ent))

        micro_f1_best_att = max([f['micro'] for f in f1_list_att])
        macro_f1_best_att = max([f['macro'] for f in f1_list_att])
        weighted_f1_best_att = max([f['weighted'] for f in f1_list_att])
        print('N+1 clf (best_thres f1) via ATT: micro: {}, macro: {}, weighted: {}.'.format(micro_f1_best_att,
                                                                                            macro_f1_best_att,
                                                                                            weighted_f1_best_att))

        return


    model, opt = init_oodgat()
    best_metric, best_t = train_oodgat(model, opt)
    evaluate_oodgat(model)
    print('best_metric: {}, best_t:{}.'.format(best_metric, best_t))

def local_ent_loss(logits, att, n_id_classes, m=0.5):
    att_norm = F.sigmoid(torch.hstack([att[0], att[1]]).mean(axis=1)).detach()  # n-dim
    mask = torch.ge(att_norm - m, 0)
    ce_uni = CE_uniform(n_id_classes, reduction=False)
    ce = ce_uni(logits)  # N-dim
    if mask.sum() > 0:
        loss = ce[mask].mean()
    else:
        loss = 0

    return loss







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KDD22: Learning on Graphs with Out-of-Distribution nodes')
    parser.add_argument('--dataset', default="Planetoid_Cora", type=str)
    parser.add_argument('--ID_classes', default=[4, 2, 5, 6], type=list)
    parser.add_argument('--splits', default=[20, 10, 1000], type=list)
    parser.add_argument('--continuous', default=False, type=bool)
    parser.add_argument('--model_name', default='OODGAT', type=str)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--drop_prob', default=0.5, type=float)
    parser.add_argument('--drop_input', default=0.5, type=float)
    parser.add_argument('--drop_edge', default=0.6, type=float)
    parser.add_argument('--heads', default=4, type=int)
    parser.add_argument('--w_consistent', default=2.0, type=float)
    parser.add_argument('--w_ent', default=0.05, type=float)
    parser.add_argument('--w_discrepancy', default=5e-3, type=float)
    parser.add_argument('--margin', default=0.6, type=float)
    parser.add_argument('--random_seed_data', default=123, type=int)
    parser.add_argument('--random_seed_model', default=456, type=int)


    args = parser.parse_args()
    train(args)

