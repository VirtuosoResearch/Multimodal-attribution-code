import os
import torch
import torchvision
from torchvision.datasets.folder import default_loader
from engine.datasets import dataset_classes
from engine.tools.utils import load_json


def get_few_shot_setup_name(train_shot, seed):
    """Get the name for a few-shot setup.
    """
    return f"shot_{train_shot}-seed_{seed}"

class TextTensorDatasetMultimodalNeurons(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor, eot_indices, prompts_dict, n_shots=None):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.eot_indices = eot_indices
        self.prompts_dict = prompts_dict

        from collections import defaultdict
        counters = defaultdict(int)
        self.all_prompts = []
        for lab in self.label_tensor.tolist():
            lab = int(lab)
            prompt_list = self.prompts_dict[lab]
            j = counters[lab]

            # --- safety guard ---
            if j >= len(prompt_list):
                raise IndexError(
                    f"More samples ({j+1}) for label {lab} than available prompts ({len(prompt_list)}). "
                    "Your label ordering may not match the prompt building order."
                )
            # ---------------------
            self.all_prompts.append(prompt_list[j % len(prompt_list)])
            counters[lab] += 1

    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.eot_indices[index], self.all_prompts[index]

    def __len__(self):
        if isinstance(self.input_tensor, torch.Tensor):
            return self.input_tensor.size(0)
        else:
            return len(self.input_tensor)

class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor, eot_indices, n_shots=None):
        # Optionally sample n_shots examples per class at random.
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.eot_indices = eot_indices

        if isinstance(n_shots, int):
            indices = self._select_n_shots(label_tensor, n_shots)
            if isinstance(input_tensor, list):
                indices_list = indices.tolist()                     # convert to Python ints
                self.input_tensor = [input_tensor[i] for i in indices_list]
            else:                                                   # original tensor path
                self.input_tensor = input_tensor[indices]
                
            self.label_tensor, self.eot_indices = label_tensor[indices], eot_indices[indices]
            # self.input_tensor, self.label_tensor, self.eot_indices = input_tensor[indices], label_tensor[indices], eot_indices[indices]
            print(f"=> Using {n_shots} text shots per class, with total of {len(self)} samples")
        
        # Optionally average features per class.
        elif isinstance(n_shots, str) and n_shots.lower() == "average":
            self.input_tensor, self.label_tensor, self.eot_indices = self._average_features(input_tensor, label_tensor, eot_indices)
            print(f"=> Averaging text features per class, with total of {len(self)} samples")

        elif n_shots is not None:
            raise ValueError("n_shots must be an int, None, or 'average'")
    

    def _select_n_shots(self, labels, n_shots):
        indices = []
        mini = 1000000
        for label in torch.unique(labels):
            label_inds = (labels == label).nonzero(as_tuple=True)[0]
            n = min(n_shots, label_inds.size(0))
            perm = torch.randperm(label_inds.size(0))[:n]
            indices.append(label_inds[perm])
            mini = min(mini, n)
        print(f"=> Using minumum of {mini} text shots per class, with total of {len(self)} samples")
        return torch.cat(indices)

    def _average_features(self, inputs, labels, eot_indices):
        unique_labels = torch.unique(labels)
        avg_inputs, avg_eot = [], []
        for label in unique_labels:
            mask = labels == label
            avg_inputs.append(inputs[mask].mean(dim=0))
            avg_eot.append(eot_indices[mask][0])
        avg_inputs = torch.stack(avg_inputs)
        avg_labels = unique_labels
        avg_eot = torch.stack(avg_eot) if isinstance(avg_eot[0], torch.Tensor) else torch.tensor(avg_eot)
        return avg_inputs, avg_labels, avg_eot

    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.eot_indices[index]

    def __len__(self):
        if isinstance(self.input_tensor, torch.Tensor):
            return self.input_tensor.size(0)
        else:
            return len(self.input_tensor)
    
# class TextTensorDataset(torch.utils.data.Dataset):
#     def __init__(self, input_tensor, label_tensor, eot_indices, n_shots=None):
#         self.input_tensor = input_tensor
#         self.label_tensor = label_tensor
#         self.eot_indices = eot_indices

#         # If n_shots is provided, precompute indices with at most n_shots per label 
#         self.shot_indices = self._select_n_shots(n_shots) if n_shots is not None else None
#         print(f"=> Using {n_shots} text shots per class, with total of {len(self)} samples")

#     def _select_n_shots(self, n_shots):
#         indices = []
#         # Loop over each unique label and select the first n_shots indices for that label
#         for label in torch.unique(self.label_tensor):
#             label_inds = (self.label_tensor == label).nonzero(as_tuple=True)[0]
#             indices.append(label_inds[:n_shots])
#         return torch.cat(indices)

#     def __getitem__(self, index):
#         idx = self.shot_indices[index] if self.shot_indices is not None else index
#         return self.input_tensor[idx], self.label_tensor[idx], self.eot_indices[idx]

#     def __len__(self):
#         return self.shot_indices.size(0) if self.shot_indices is not None else self.input_tensor.size(0)

    # def __getitem__(self, index):
    #     return self.input_tensor[index], self.label_tensor[index], self.eot_indices[index]

    # def __len__(self):
    #     return self.input_tensor.size(0)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index]
    
    def __len__(self):
        return self.input_tensor.size(0)


class DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, data_source, transform):
        self.data_source = data_source
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        img = self.transform(default_loader(item['impath']))

        output = {
            "img": img,
            "label": item['label'],
            "classname": item['classname'],
            "impath": item['impath'],
        }

        return output


def get_few_shot_benchmark(data_dir,
                           indices_dir,
                           dataset,
                           train_shot,
                           seed):
    # Check if the dataset is supported
    assert dataset in dataset_classes
    if train_shot!=-1:
        few_shot_index_file = os.path.join(
            indices_dir, dataset, f"{get_few_shot_setup_name(train_shot, seed)}.json")
        assert os.path.exists(few_shot_index_file), f"Few-shot data does not exist at {few_shot_index_file}."
        few_shot_dataset = load_json(few_shot_index_file)
    else:
        print("=> Using full dataset for feature extraction")
    benchmark = dataset_classes[dataset](data_dir)
    
    return {
        'train': few_shot_dataset['train']['data'] if train_shot!=-1 else benchmark.train,
        'val': few_shot_dataset['val']['data'] if train_shot!=-1 else benchmark.val,
        'test': benchmark.test,
        'lab2cname': benchmark.lab2cname,
        'classnames': benchmark.classnames,
    }


def get_testset(dataset, data_dir):
    if dataset in dataset_classes:
        benchmark = dataset_classes[dataset](data_dir)
        return benchmark.test
    else:
        raise NotImplementedError()


def get_label_map(data_dir, dataset_name):
    if dataset_name in ['imagenet_a', 'imagenet_r']:
        benchmark = dataset_classes[dataset_name](data_dir)
        return benchmark.label_map
    else:
        return None