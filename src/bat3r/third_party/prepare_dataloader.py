"""
Copied from Hugging Face Accelerate v1.12.0 (accelerate.data_loader).
Modified by Jakub Zadrozny (2026).
Vendored to ensure full reproducibility and consistency between multi-GPU and single-GPU runs.

Summary of changes against upstream accelerate.data_loader:
  In __prepare_data_loader: bypassed the `num_processes != 1` check so that BatchSamplerShard
  is unconditionally created even for single-process runs (num_processes == 1).
"""

# Based on Hugging Face Accelerate (Copyright 2021 The HuggingFace Team).
# Modified by Jakub Zadrozny (2026). Licensed under Apache-2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler

from accelerate import Accelerator
from accelerate.data_loader import (
    DataLoaderShard,
    get_sampler,
    SeedableRandomSampler,
    IterableDatasetShard,
    DataLoaderDispatcher,
    BatchSamplerShard,
    _PYTORCH_DATALOADER_KWARGS,
)
from accelerate.state import PartialState, DistributedType
from accelerate.utils import is_datasets_available, RNGType


def __prepare_data_loader(
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    num_processes: Optional[int] = None,
    process_index: Optional[int] = None,
    split_batches: bool = False,
    put_on_device: bool = False,
    rng_types: Optional[list[Union[str, RNGType]]] = None,
    dispatch_batches: Optional[bool] = None,
    even_batches: bool = True,
    slice_fn_for_dispatch: Optional[Callable] = None,
    use_seedable_sampler: bool = False,
    data_seed: Optional[int] = None,
    non_blocking: bool = False,
    use_stateful_dataloader: bool = False,
    torch_device_mesh=None,
) -> DataLoader:
    """
    Wraps a PyTorch `DataLoader` to generate batches for one of the processes only.

    Depending on the value of the `drop_last` attribute of the `dataloader` passed, it will either stop the iteration
    at the first batch that would be too small / not present on all processes or loop with indices from the beginning.

    Args:
        dataloader (`torch.utils.data.dataloader.DataLoader`):
            The data loader to split across several devices.
        device (`torch.device`):
            The target device for the returned `DataLoader`.
        num_processes (`int`, *optional*):
            The number of processes running concurrently. Will default to the value given by [`~state.PartialState`].
        process_index (`int`, *optional*):
            The index of the current process. Will default to the value given by [`~state.PartialState`].
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether the resulting `DataLoader` should split the batches of the original data loader across devices or
            yield full batches (in which case it will yield batches starting at the `process_index`-th and advancing of
            `num_processes` batches at each iteration).

            Another way to see this is that the observed batch size will be the same as the initial `dataloader` if
            this option is set to `True`, the batch size of the initial `dataloader` multiplied by `num_processes`
            otherwise.

            Setting this option to `True` requires that the batch size of the `dataloader` is a round multiple of
            `batch_size`.
        put_on_device (`bool`, *optional*, defaults to `False`):
            Whether or not to put the batches on `device` (only works if the batches are nested list, tuples or
            dictionaries of tensors).
        rng_types (list of `str` or [`~utils.RNGType`]):
            The list of random number generators to synchronize at the beginning of each iteration. Should be one or
            several of:

            - `"torch"`: the base torch random number generator
            - `"cuda"`: the CUDA random number generator (GPU only)
            - `"xla"`: the XLA random number generator (TPU only)
            - `"generator"`: the `torch.Generator` of the sampler (or batch sampler if there is no sampler in your
              dataloader) or of the iterable dataset (if it exists) if the underlying dataset is of that type.

        dispatch_batches (`bool`, *optional*):
            If set to `True`, the dataloader prepared is only iterated through on the main process and then the batches
            are split and broadcast to each process. Will default to `True` when the underlying dataset is an
            `IterableDataset`, `False` otherwise.
        even_batches (`bool`, *optional*, defaults to `True`):
            If set to `True`, in cases where the total batch size across all processes does not exactly divide the
            dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
            all workers.
        slice_fn_for_dispatch (`Callable`, *optional*`):
            If passed, this function will be used to slice tensors across `num_processes`. Will default to
            [`~utils.slice_tensors`]. This argument is used only when `dispatch_batches` is set to `True` and will be
            ignored otherwise.
        use_seedable_sampler (`bool`, *optional*, defaults to `False`):
            Whether to use the [`~data_loader.SeedableRandomSampler`] instead of a `RandomSampler` for better
            reproducibility. Comes at a cost of potentially different performances due to different shuffling
            algorithms but ensures results will be the *exact* same. Should be paired with `set_seed()` at every
            `self.set_epoch`
        data_seed (`int`, *optional*, defaults to `None`):
            The seed to use for the underlying generator when using `use_seedable_sampler`. If `None`, the generator
            will use the current default seed from torch.
        non_blocking (`bool`, *optional*, defaults to `False`):
            If set to `True`, dataloader will utilize non-blocking host-to-device transfers. If the dataloader has
            `pin_memory` set to `True`, this will help to increase overlap between data transfer and computations.
        use_stateful_dataloader (`bool`, *optional*, defaults to `False`):
            "If set to true, the dataloader prepared by the Accelerator will be backed by "
            "[torchdata.StatefulDataLoader](https://github.com/pytorch/data/tree/main/torchdata/stateful_dataloader).
            This requires `torchdata` version 0.8.0 or higher that supports StatefulDataLoader to be installed."
        torch_device_mesh (`torch.distributed.DeviceMesh`, *optional*, defaults to `None`):
            PyTorch device mesh.

    Returns:
        `torch.utils.data.dataloader.DataLoader`: A new data loader that will yield the portion of the batches
    """
    if dispatch_batches is None:
        if not put_on_device:
            dispatch_batches = False
        else:
            dispatch_batches = isinstance(dataloader.dataset, IterableDataset)

    if dispatch_batches and not put_on_device:
        raise ValueError("Using `dispatch_batches=True` requires `put_on_device=True`.")
    # Grab defaults from PartialState
    state = PartialState()
    if num_processes is None:
        num_processes = state.num_processes

    if process_index is None:
        process_index = state.process_index

    if torch_device_mesh:
        if state.distributed_type == DistributedType.DEEPSPEED:
            submesh_tp_size = 1
            if "tp" in torch_device_mesh.mesh_dim_names:
                submesh_tp_size = torch_device_mesh["tp"].size()
            process_index = process_index // submesh_tp_size
            num_processes = num_processes // submesh_tp_size
        else:
            submesh_fsdp_size = 1
            submesh_dp_size = 1
            submesh_tp_size = 1
            submesh_cp_size = 1
            if "tp" in torch_device_mesh.mesh_dim_names:
                submesh_tp_size = torch_device_mesh["tp"].size()
            if "cp" in torch_device_mesh.mesh_dim_names:
                submesh_cp_size = torch_device_mesh["cp"].size()
            if "dp_replicate" in torch_device_mesh.mesh_dim_names:
                submesh_dp_size = torch_device_mesh["dp_replicate"].size()
            if "dp_shard" in torch_device_mesh.mesh_dim_names:
                submesh_fsdp_size = torch_device_mesh["dp_shard"].size()
            process_index = process_index // (submesh_tp_size * submesh_cp_size)
            num_processes = submesh_fsdp_size * submesh_dp_size

    # Sanity check
    if split_batches:
        if dataloader.batch_size is not None:
            batch_size_for_check = dataloader.batch_size
        else:
            if hasattr(dataloader.batch_sampler, "batch_size"):
                batch_size_for_check = dataloader.batch_sampler.batch_size
            else:
                raise ValueError(
                    "In order to use `split_batches==True` you must have a `batch_size` attribute either in the passed "
                    "`dataloader` or `dataloader.batch_sampler` objects, and it has to return a natural number. "
                    "Your `dataloader.batch_size` is None and `dataloader.batch_sampler` "
                    f"(`{type(dataloader.batch_sampler)}`) does not have the `batch_size` attribute set."
                )

        if batch_size_for_check > 1 and batch_size_for_check % num_processes != 0:
            raise ValueError(
                f"To use a `DataLoader` in `split_batches` mode, the batch size ({dataloader.batch_size}) "
                f"needs to be a round multiple of the number of processes ({num_processes})."
            )

    new_dataset = dataloader.dataset
    new_batch_sampler = dataloader.batch_sampler if not isinstance(new_dataset, IterableDataset) else None
    sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
    synchronized_generator = None

    sampler = get_sampler(dataloader)
    if isinstance(sampler, RandomSampler) and use_seedable_sampler:
        sampler = SeedableRandomSampler(
            data_source=sampler.data_source,
            replacement=sampler.replacement,
            num_samples=sampler._num_samples,
            generator=getattr(
                sampler,
                "generator",
                torch.Generator(device=torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"),
            ),
            data_seed=data_seed,
        )

    if isinstance(dataloader.sampler, RandomSampler) and state.distributed_type == DistributedType.XLA:
        generator = torch.Generator(
            device=torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
        )
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator.manual_seed(seed)
        dataloader.generator = generator
        dataloader.sampler.generator = generator

    # Full reproducibility and consistency between multi-GPU and single-GPU:
    # Unconditionally create BatchSamplerShard even when num_processes == 1.
    if not dispatch_batches:
        if is_datasets_available():
            from datasets import IterableDataset as DatasetsIterableDataset
        if (
            is_datasets_available()
            and isinstance(new_dataset, DatasetsIterableDataset)
            and not split_batches
            and new_dataset.n_shards > num_processes
        ):
            new_dataset = new_dataset.shard(num_shards=num_processes, index=process_index)
        elif isinstance(new_dataset, IterableDataset):
            if getattr(dataloader.dataset, "generator", None) is not None:
                synchronized_generator = dataloader.dataset.generator
            new_dataset = IterableDatasetShard(
                new_dataset,
                batch_size=dataloader.batch_size,
                drop_last=dataloader.drop_last,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
            )
        else:
            if not use_seedable_sampler and hasattr(sampler, "generator"):
                if sampler.generator is None:
                    sampler.generator = torch.Generator(
                        device=torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
                    )
                    seed = int(torch.empty((), dtype=torch.int64).random_().item())
                    sampler.generator.manual_seed(seed)
                synchronized_generator = sampler.generator
            batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
            new_batch_sampler = BatchSamplerShard(
                batch_sampler,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
                even_batches=even_batches,
            )

    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    ]

    if rng_types is not None and synchronized_generator is None and "generator" in rng_types:
        rng_types.remove("generator")

    kwargs = {
        k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k])
        for k in _PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    if new_batch_sampler is None:
        kwargs["drop_last"] = dataloader.drop_last
        kwargs["batch_size"] = (
            dataloader.batch_size // num_processes if split_batches and not dispatch_batches else dataloader.batch_size
        )
    if dispatch_batches:
        kwargs.pop("generator")
        dataloader = DataLoaderDispatcher(
            new_dataset,
            split_batches=split_batches,
            batch_sampler=new_batch_sampler,
            _drop_last=dataloader.drop_last,
            _non_blocking=non_blocking,
            slice_fn=slice_fn_for_dispatch,
            use_stateful_dataloader=use_stateful_dataloader,
            torch_device_mesh=torch_device_mesh,
            **kwargs,
        )
    elif sampler_is_batch_sampler:
        dataloader = DataLoaderShard(
            new_dataset,
            device=device if put_on_device and state.distributed_type != DistributedType.XLA else None,
            sampler=new_batch_sampler,
            batch_size=dataloader.batch_size,
            rng_types=rng_types,
            _drop_last=dataloader.drop_last,
            _non_blocking=non_blocking,
            synchronized_generator=synchronized_generator,
            use_stateful_dataloader=use_stateful_dataloader,
            **kwargs,
        )
    else:
        dataloader = DataLoaderShard(
            new_dataset,
            device=device if put_on_device and state.distributed_type != DistributedType.XLA else None,
            batch_sampler=new_batch_sampler,
            rng_types=rng_types,
            synchronized_generator=synchronized_generator,
            _drop_last=dataloader.drop_last,
            _non_blocking=non_blocking,
            use_stateful_dataloader=use_stateful_dataloader,
            **kwargs,
        )

    if isinstance(sampler, SeedableRandomSampler) and use_seedable_sampler:
        dataloader.set_sampler(sampler)
    return dataloader


def _prepare_data_loader(
    accelerator: Accelerator, data_loader: torch.utils.data.DataLoader, device_placement=None, slice_fn_for_dispatch=None
):
    """
    Prepares a PyTorch DataLoader for training in any distributed setup.

    Args:
        data_loader (`torch.utils.data.DataLoader`):
            A vanilla PyTorch DataLoader to prepare
        device_placement (`bool`, *optional*):
            Whether or not to place the batches on the proper device in the prepared dataloader. Will default to
            `self.device_placement`.
        slice_fn_for_dispatch (`Callable`, *optional*`):
            If passed, this function will be used to slice tensors across `num_processes`.
    """
    if getattr(data_loader, "_is_accelerate_prepared", False):
        if data_loader not in accelerator._dataloaders:
            accelerator._dataloaders.append(data_loader)
        return data_loader
    if device_placement is None:
        device_placement = accelerator.device_placement

    device_mesh = accelerator._prepare_device_mesh()

    prepared_data_loader = __prepare_data_loader(
        data_loader,
        accelerator.device,
        num_processes=accelerator.num_processes,
        process_index=accelerator.process_index,
        split_batches=accelerator.split_batches,
        put_on_device=device_placement,
        rng_types=accelerator.rng_types.copy(),
        dispatch_batches=accelerator.dispatch_batches,
        even_batches=accelerator.even_batches,
        slice_fn_for_dispatch=slice_fn_for_dispatch,
        use_seedable_sampler=accelerator.use_seedable_sampler,
        data_seed=accelerator.dataloader_config.data_seed,
        non_blocking=accelerator.non_blocking,
        use_stateful_dataloader=accelerator.use_stateful_dataloader,
        torch_device_mesh=device_mesh,
    )
    accelerator._dataloaders.append(prepared_data_loader)
    return prepared_data_loader
