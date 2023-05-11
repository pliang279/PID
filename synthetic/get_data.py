import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader


def get_dataloader(path, keys=['a','b','label'], modalities=[0,1], batch_size=32, num_workers=4, test_path=None):
    if type(path) == list:
        data_list = []
        for dat in path:
            try:
                with open(dat, "rb") as f:
                    data_list.append(pickle.load(f))
                    f.close()
            except Exception as ex:
                print("Error during unpickling object", ex)
                exit()
        data = dict()
        data['train'] = dict()
        for f in data_list[0]['train']:
            for (i, dat) in enumerate(data_list):
                d = data['train'].get(f, [])
                if i == 0:
                    d.append(dat['train'][f])
                else:
                    d.append(dat['train'][f][:int(0.1*len(dat['train'][f]))])
                data['train'][f] = d
            data['train'][f] = np.concatenate(data['train'][f])
    else:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as ex:
            print("Error during unpickling object", ex)
            exit()
    if test_path:
        try:
            with open(test_path, "rb") as f:
                test_data = pickle.load(f)
        except Exception as ex:
            print("Error during unpickling object", ex)
            exit()
        data['valid1'] = test_data['valid1']
        data['valid2'] = test_data['valid2']
        data['test'] = test_data['test']
    label = keys[-1]

    traindata = DataLoader(SyntheticDataset(data['train'], keys, modalities=modalities),
                    shuffle=True, 
                    num_workers=num_workers, 
                    batch_size=batch_size, 
                    collate_fn=process_input)
    print("Train data: {}".format(data['train'][label].shape[0]))
    if 'valid' in data:
        validdata1 = DataLoader(SyntheticDataset(data['valid'], keys, modalities=modalities),
                        shuffle=False, 
                        num_workers=num_workers, 
                        batch_size=batch_size, 
                        collate_fn=process_input)
        print("Valid data: {}".format(data['valid'][label].shape[0]))
        validdata2 = None
    else:
        validdata1 = DataLoader(SyntheticDataset(data['valid1'], keys, modalities=modalities),
                            shuffle=False, 
                            num_workers=num_workers, 
                            batch_size=batch_size, 
                            collate_fn=process_input)
        validdata2 = DataLoader(SyntheticDataset(data['valid2'], keys, modalities=modalities),
                            shuffle=False, 
                            num_workers=num_workers, 
                            batch_size=batch_size, 
                            collate_fn=process_input)
        print("Valid data 1: {}".format(data['valid1'][label].shape[0]))
        print("Valid data 2: {}".format(data['valid2'][label].shape[0]))
    testdata = DataLoader(SyntheticDataset(data['test'], keys, modalities=modalities),
                        shuffle=False, 
                        num_workers=num_workers, 
                        batch_size=batch_size, 
                        collate_fn=process_input)
    print("Test data: {}".format(data['test'][label].shape[0]))

    return traindata, validdata1, validdata2, testdata


class SyntheticDataset(Dataset):
    def __init__(self, data, keys, modalities):
        self.data = data
        self.keys = keys
        self.modalities = modalities
        
    def __len__(self):
        return len(self.data[self.keys[-1]])

    def __getitem__(self, index):
        tmp = []
        for i, modality in enumerate(self.modalities):
            if modality:
                tmp.append(torch.tensor(self.data[self.keys[i]][index]))
            else:
                tmp.append(torch.ones(self.data[self.keys[i]][index].size))
        tmp.append(torch.tensor(self.data[self.keys[-1]][index]))
        return tmp


def process_input(inputs):
    processed_input = []
    labels = []

    for i in range(len(inputs[0])-1):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input.append(torch.stack(feature))

    for sample in inputs:  
        labels.append(sample[-1])
    processed_input.append(torch.tensor(labels).view(len(inputs),))
    
    return processed_input
