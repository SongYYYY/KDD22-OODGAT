import torch
import numpy as np

# split the graph into ID and OOD nodes. generate masks for clf&ood tasks.
# this version allows for the following tasks:
# 1) ID clf
# 2) OOD detection
# 3) joint classification (N+1 classes)
def generate_masks(data, ID_classes, n_samples_per_class, val_size_per_class, test_size, random_seed=123):
    '''
    generate masks for node classification and OOD detection tasks.
    specifically, we view all nodes as 2 disjoint parts, i.e., ID mask and OOD mask.
    masks for classification are all sampled within ID part, i.e., train_mask, val_mask and test_mask.
    masks for OOD detection are constructed by combining ID masks and OOD masks.
    besides, we need to re-encode ID classes such that all ID labels are within {0, 1, ..., #ID_classes-1}.
    :param data: graph pyg_data.
    :return:
    ID_mask: ID mask.
    OOD_mask: OOD mask.
    train_mask_ID: training ID samples.
    val_mask_ID: validation ID samples.
    test_mask_ID: test ID samples.
    train_mask_OOD:
    val_mask_OOD:
    test_mask_ODO:
    detection_mask_val:
    detection_y_val: list
    joint_y_val: list
    detection_mask_test:
    detection_y_test: list
    joint_y_test: list
    ID_y: tensor. ground truth labels mapped for ID classification.

    '''
    seed = np.random.get_state()
    np.random.seed(random_seed)
    classes = set(data.y.tolist())
    n_classes = len(classes)
    ID_classes = set(ID_classes)
    assert len(ID_classes) < n_classes
    OOD_classes = classes - ID_classes

    n_nodes = len(data.y)
    ID_mask, OOD_mask = torch.zeros(n_nodes), torch.zeros(n_nodes)
    for i in range(n_nodes):
        if data.y[i].item() in ID_classes:
            ID_mask[i] = 1
        else:
            OOD_mask[i] = 1
    ID_mask, OOD_mask = ID_mask.bool(), OOD_mask.bool()
    assert n_samples_per_class*len(ID_classes) + val_size_per_class*len(ID_classes) + test_size <= ID_mask.sum()
    assert n_samples_per_class*len(ID_classes) + val_size_per_class*len(ID_classes) + test_size <= OOD_mask.sum()

    # ID part
    ID_idx = torch.nonzero(ID_mask).squeeze().tolist()
    train_idx = []
    val_idx = []
    for k in ID_classes:
        k_idxs = torch.nonzero(data.y==k).squeeze().tolist()
        samples_to_take = n_samples_per_class + val_size_per_class
        samples = np.random.choice(k_idxs, samples_to_take, False)
        train_idx.extend(samples[:n_samples_per_class])
        val_idx.extend(samples[n_samples_per_class:])
    left_idx = list(set(ID_idx)-set(train_idx)-set(val_idx))
    left_idx = np.random.permutation(left_idx)
    test_idx = left_idx[:test_size]

    train_mask, val_mask, test_mask = torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes)
    for i in range(n_nodes):
        if i in train_idx:
            train_mask[i] = 1
        elif i in val_idx:
            val_mask[i] = 1
        elif i in test_idx:
            test_mask[i] = 1
    train_mask_ID, val_mask_ID, test_mask_ID = train_mask.bool(), val_mask.bool(), test_mask.bool()

    # OOD part
    OOD_idx = torch.nonzero(OOD_mask).squeeze().tolist()
    OOD_idx = np.random.permutation(OOD_idx)
    train_idx = OOD_idx[: train_mask_ID.sum()]
    val_idx = OOD_idx[train_mask_ID.sum(): train_mask_ID.sum()+val_mask_ID.sum()]
    test_idx = OOD_idx[train_mask_ID.sum()+val_mask_ID.sum(): train_mask_ID.sum()+val_mask_ID.sum()+test_size]
    train_mask, val_mask, test_mask = torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes)
    for i in range(n_nodes):
        if i in train_idx:
            train_mask[i] = 1
        elif i in val_idx:
            val_mask[i] = 1
        elif i in test_idx:
            test_mask[i] = 1
    train_mask_OOD, val_mask_OOD, test_mask_OOD = train_mask.bool(), val_mask.bool(), test_mask.bool()

    # detection
    detection_mask_val = val_mask_ID | val_mask_OOD
    detection_y_val = data.y[detection_mask_val]
    detection_y_val = [y.item() in OOD_classes for y in detection_y_val]
    detection_mask_test = test_mask_ID | test_mask_OOD
    detection_y_test = data.y[detection_mask_test]
    detection_y_test = [y.item() in OOD_classes for y in detection_y_test]

    # re-map labels
    label_map = {y : i for i, y in enumerate(ID_classes)}

    # ID classification y
    ID_y = data.y.clone()
    for i in range(n_nodes):
        ID_y[i] = label_map.get(data.y[i].item(), 0)

    # joint clf y
    joint_y_val = data.y[detection_mask_val]
    joint_y_val = [label_map.get(y.item(), len(ID_classes)) for y in joint_y_val]
    joint_y_test = data.y[detection_mask_test]
    joint_y_test = [label_map.get(y.item(), len(ID_classes)) for y in joint_y_test]

    # reset seed
    np.random.set_state(seed)

    return ID_mask, OOD_mask, train_mask_ID, val_mask_ID, test_mask_ID, train_mask_OOD, val_mask_OOD, test_mask_OOD, \
        detection_mask_val, detection_y_val, joint_y_val, detection_mask_test, detection_y_test, joint_y_test, \
        ID_y



#
def normalize_feature(x):
    x_new = x.clone()
    sum = x_new.sum(axis=1, keepdims=True)
    x_new = x_new / sum
    x_new[np.isinf(x_new)] = 0

    return x_new

