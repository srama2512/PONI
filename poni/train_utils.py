import collections

import torch
import torch.nn as nn
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
    string_classes,
)


def get_loss_fn(loss_type):
    assert loss_type in ["bce", "l2", "l1", "xent"]
    loss_fn = None
    if loss_type == "bce":
        loss_fn = nn.BCELoss(reduction="none")
    elif loss_type == "l2":
        loss_fn = nn.MSELoss(reduction="none")
    elif loss_type == "l1":
        loss_fn = nn.L1Loss(reduction="none")
    elif loss_type == "xent":
        loss_fn = nn.CrossEntropyLoss(reduction="none")
    return loss_fn


def get_activation_fn(activation_type):
    assert activation_type in ["none", "sigmoid", "relu"]
    activation = nn.Identity()
    if activation_type == "sigmoid":
        activation = nn.Sigmoid()
    elif activation_type == "relu":
        activation = nn.ReLU()
    return activation


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size.
    Modified version of default_collate which returns the batch as it has lists
    of varying length sizes.
    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            return batch
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
