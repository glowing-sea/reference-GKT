import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F # contain stateless functions such as relu, softmax, embedding, dropout, one_hot, pad, etc. Not trainable.
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence # [batch_size, max_seq_len, 2 * concept_num]
from utils import build_dense_graph

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


class KTDataset(Dataset):
    def __init__(self, features, questions, answers):
        super(KTDataset, self).__init__() # Call Dataset.__init__(self), but Dataset has no __init__ function.
        self.features = features
        self.questions = questions
        self.answers = answers

    def __getitem__(self, index):
        return self.features[index], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.features)


def pad_collate(batch):
    # *: takes a list and passes it as individual arguments into zip
    # batch = [
    #     ([1, 2],   [10, 11], [0, 1]),   # student A
    #     ([3, 4, 5],[12, 13, 14], [1, 0, 1])  # student B
    # ]
    # features = ([1, 2], [3, 4, 5])
    # questions = ([10, 11], [12, 13, 14])
    # answers = ([0, 1], [1, 0, 1])
    (features, questions, answers) = zip(*batch)
    # features = [[1, 2], [3, 4, 5]]
    # questions = [[10, 11], [12, 13, 14]]
    # answers = [[0, 1], [1, 0, 1]]
    features = [torch.LongTensor(feat) for feat in features]
    questions = [torch.LongTensor(qt) for qt in questions]
    answers = [torch.LongTensor(ans) for ans in answers]
    # takes a list of tensors with different lengths and turns them into a padded batch tensor with shape [batch_size, max_seq_len]
    # feature_pad = [[1, 2, -1], [3, 4, 5]]
    # batch_first=True
    # [batch_size, seq_len, features]
    # batch_first=False
    # [seq_len, batch_size, features]
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)
    return feature_pad, question_pad, answer_pad


def load_dataset(file_path, batch_size, graph_type, dkt_graph_path=None, train_ratio=0.7, val_ratio=0.2, shuffle=True, model_type='GKT', use_binary=True, res_len=2, use_cuda=True):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        batch_size: the size of a student batch
        graph_type: the type of the concept graph
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
    Return:
        concept_num: the number of all concepts(or questions)
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """

    # Check if these columns exist
    df = pd.read_csv(file_path)
    if "skill_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {file_path}")
    if "correct" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {file_path}")
    if "user_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {file_path}")


    # if not (df['correct'].isin([0, 1])).all():
    #     raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

    # Step 1.1 - Remove questions without skill
    df.dropna(subset=['skill_id'], inplace=True)

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    # convert skill labels to integers starting from 0
    # skill_id	first time seen	label when sorted=False
    # 51444	yes	0
    # 33159	yes	1
    # 42000	yes	2
    # When sorted=True, sort first and then label
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)  # we can also use problem_id to represent exercises

    # Step 3 - Cross skill id with answer to form a synthetic feature
    # use_binary: (0,1); !use_binary: (1,2,3,4,5,6,7,8,9,10,11,12). Either way, the correct result index is guaranteed to be 1
    # correct or incorrect may be interpreted as choosing between two options and options 1 is always the correct one.
    if use_binary:
        # multiply by 2 to create space for correct(1) and incorrect(0), correct comes after incorrect
        df['skill_with_answer'] = df['skill'] * 2 + df['correct']
    else:
        df['skill_with_answer'] = df['skill'] * res_len + df['correct'] - 1


    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []

    def get_data(series):
        feature_list.append(series['skill_with_answer'].tolist())
        question_list.append(series['skill'].tolist())
        answer_list.append(series['correct'].eq(1).astype('int').tolist())
        seq_len_list.append(series['correct'].shape[0])

    df.groupby('user_id').apply(get_data)
    # For feature_list, we have a list of students, each student contains a list of features (skill_with_answer) 
    # in temporal order

    max_seq_len = np.max(seq_len_list)
    print('max seq_len: ', max_seq_len)

    student_num = len(seq_len_list)
    print('student num: ', student_num)

    feature_dim = int(df['skill_with_answer'].max() + 1)
    print('feature_dim: ', feature_dim)
    question_dim = int(df['skill'].max() + 1)
    print('question_dim: ', question_dim)
    concept_num = question_dim

    # print('feature_dim:', feature_dim, 'res_len*question_dim:', res_len*question_dim)
    # assert feature_dim == res_len * question_dim

    kt_dataset = KTDataset(feature_list, question_list, answer_list)
    train_size = int(train_ratio * student_num)
    val_size = int(val_ratio * student_num)
    test_size = student_num - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(kt_dataset, [train_size, val_size, test_size])
    print('train_size: ', train_size, 'val_size: ', val_size, 'test_size: ', test_size)

    # Why collate_fn=pad_collate?
    # Because the sequences have different lengths, we need to pad them to the same length in
    # What is collate_fn in PyTorch DataLoader?
    # A function that takes a list of samples from the dataset and converts them into a batch.
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

    graph = None
    if model_type == 'GKT':
        if graph_type == 'Dense':
            graph = build_dense_graph(concept_num)
        elif graph_type == 'Transition':
            # train_dataset.indices are not continuous because of the test-train-val split and shuffle
            graph = build_transition_graph(question_list, seq_len_list, train_dataset.indices, student_num, concept_num)
        elif graph_type == 'DKT':
            graph = build_dkt_graph(dkt_graph_path, concept_num)
        if use_cuda and graph_type in ['Dense', 'Transition', 'DKT']:
            graph = graph.cuda()
            # same as graph = graph.to(torch.device("cuda"))
    return concept_num, graph, train_data_loader, valid_data_loader, test_data_loader




def build_transition_graph(question_list, seq_len_list, indices, student_num, concept_num):
    # graph[a, b] = probability that concept b appears right after concept a in student histories.
    # indices = a list of student indices in the training set
    # np.arange(student_num) = a list of all student indices in the training set

    graph = np.zeros((concept_num, concept_num))
    student_dict = dict(zip(indices, np.arange(student_num))) # Rubbish code!!!
    # zip([7, 1, 4, 9, 2, 6],
    # [0,1,2,3,4,5,6,7,8,9])
    # 
    # (7, 0)
    # (1, 1)
    # (4, 2)
    # (9, 3)
    # (2, 4)
    # (6, 5)
    for i in range(student_num):
        if i not in student_dict:
            continue
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            next = questions[j + 1]
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))
    def inv(x): # normalise from i to j
        if x == 0:
            return x
        return 1. / x
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    # covert to tensor
    graph = torch.from_numpy(graph).float()
    return graph


def build_dkt_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    graph = torch.from_numpy(graph).float()
    return graph