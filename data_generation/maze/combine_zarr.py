import dask.array as da
import numpy as np
import zarr
import argparse
import os

from typing import Any, Dict, Union, Optional, Sequence, Tuple
ArrayLike = Union[np.ndarray, da.Array]
DType = Any

def main():
    """
    Converts maze data from multiple all zarr files in <data_dir>
    into a single zarr file.
    
    The script will store the generated zarr file in data_dir.

    Usage:
    python data_generation/maze/combine_zarr.py --data_dir <path_to_data_dir>
    """
    # parse data_dir from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()
    assert args.data_dir is not None

    # load in zarr arrays
    zarr_paths = []
    for file in os.listdir(args.data_dir):
        if file.endswith(".zarr"):
            zarr_paths.append(os.path.join(args.data_dir, file))
    zarr_paths = sorted(zarr_paths)
    zarrs = [zarr.open(zarr_path, mode='r') for zarr_path in zarr_paths]
    
    # concatenate and rechunk
    combined_states = concatenate_and_rechunk([z['data']['state'] for z in zarrs])
    combined_targets = concatenate_and_rechunk([z['data']['target'] for z in zarrs])
    combined_actions = concatenate_and_rechunk([z['data']['action'] for z in zarrs])
    
    # double chunk size for images
    chunk_size = zarrs[0]['data']['img'].chunks
    combined_imgs = concatenate_and_rechunk([z['data']['img'] for z in zarrs],
                                            chunks=chunk_size)

    # Concatenate episode ends separately
    combined_episode_ends = []
    current_end = 0
    for z in zarrs:
        episode_ends = z['meta']['episode_ends']
        for episode_end in episode_ends:
            combined_episode_ends.append(current_end+episode_end)
        current_end += episode_ends[-1]
    combined_episode_ends = np.array(combined_episode_ends)
    assert(combined_episode_ends[-1] == combined_states.shape[0])
    print("Concatenated data.")

    # save to zarr
    combined_zarr_path = os.path.join(args.data_dir, 'maze_image_dataset_combined.zarr')
    root = zarr.open_group(combined_zarr_path, mode='w')
    data_dir = root.create_group('data')
    meta_dir = root.create_group('meta')

    da.to_zarr(arr=combined_states, url=combined_zarr_path, 
               component='data/state', overwrite=False)
    da.to_zarr(arr=combined_actions, url=combined_zarr_path, 
               component='data/action', overwrite=False)
    da.to_zarr(arr=combined_targets, url=combined_zarr_path, 
               component='data/target', overwrite=False)
    da.to_zarr(arr=combined_imgs, url=combined_zarr_path, 
               component='data/img', overwrite=False)
    meta_dir.create_dataset('episode_ends', data=combined_episode_ends)
    print("Saved combined dataset to zarr file.")
    breakpoint()
    
    # print chunk sizes
    print("\nChunk sizes:")
    print("state:", combined_states.chunksize)
    print("action:", combined_actions.chunksize)
    print("target:", combined_targets.chunksize)
    print("img:", combined_imgs.chunksize)
    print('episode_ends:', meta_dir['episode_ends'].chunks)

    dataset = zarr.open(combined_zarr_path, mode='r')
    zarr_ends = [0]
    for z in zarrs:
        zarr_ends.append(z['meta']['episode_ends'][-1] + zarr_ends[-1])
    for i in range(len(zarrs)):
        # check 5 random point
        z = zarrs[i]
        start_idx = zarr_ends[i]
        end_idx = zarr_ends[i+1]
        length = end_idx - start_idx

        for j in range(10):
            idx = np.random.randint(0, length)
            assert np.allclose(dataset['data']['state'][idx+start_idx], 
                                z['data']['state'][idx])
            assert np.allclose(dataset['data']['action'][idx+start_idx], 
                                z['data']['action'][idx])
            assert np.allclose(dataset['data']['target'][idx+start_idx], 
                                z['data']['target'][idx])
            assert np.allclose(dataset['data']['img'][idx+start_idx], 
                                z['data']['img'][idx])
    print("\nPassed debug tests.")
    


# Code modified from:
# https://github.com/tomwhite/sgkit/blob/main/sgkit/io/utils.py

def concatenate_and_rechunk(
    zarrs: Sequence[zarr.core.Array],
    chunks: Optional[Tuple[int, ...]] = None,
    dtype: DType = None,
) -> da.Array:
    """Perform a concatenate and rechunk operation on a collection of Zarr arrays
    to produce an array with a uniform chunking, suitable for saving as
    a single Zarr array.

    In contrast to Dask's ``rechunk`` method, the Dask computation graph
    is embarrassingly parallel and will make efficient use of memory,
    since no Zarr chunks are cached by the Dask scheduler.

    The Zarr arrays must have matching shapes except in the first
    dimension.

    Parameters
    ----------
    zarrs
        Collection of Zarr arrays to concatenate.
    chunks : Optional[Tuple[int, ...]], optional
        The chunks to apply to the concatenated arrays. If not specified
        the chunks for the first array will be applied to the concatenated
        array.
    dtype
        The dtype of the concatenated array, by default the same as the
        first array.

    Returns
    -------
    A Dask array, suitable for saving as a single Zarr array.

    Raises
    ------
    ValueError
        If the Zarr arrays do not have matching shapes (except in the first
        dimension).
    """

    if len(set([z.shape[1:] for z in zarrs])) > 1:
        shapes = [z.shape for z in zarrs]
        raise ValueError(
            f"Zarr arrays must have matching shapes (except in the first dimension): {shapes}"
        )

    lengths = np.array([z.shape[0] for z in zarrs])
    lengths0 = np.insert(lengths, 0, 0, axis=0)
    offsets = np.cumsum(lengths0)
    total_length = offsets[-1]

    shape = (total_length, *zarrs[0].shape[1:])
    chunks = chunks or zarrs[0].chunks
    dtype = dtype or zarrs[0].dtype

    ar = da.empty(shape, chunks=chunks)

    def load_chunk(
        x: ArrayLike,
        zarrs: Sequence[zarr.Array],
        offsets: ArrayLike,
        block_info: Dict[Any, Any],
    ) -> ArrayLike:
        return _slice_zarrs(zarrs, offsets, block_info[0]["array-location"])

    return ar.map_blocks(load_chunk, zarrs=zarrs, offsets=offsets, dtype=dtype)


def _zarr_index(offsets: ArrayLike, pos: int) -> int:
    """Return the index of the zarr file that pos falls in"""
    index: int = np.searchsorted(offsets, pos, side="right") - 1  # type: ignore[assignment]
    return index


def _slice_zarrs(
    zarrs: Sequence[zarr.Array], offsets: ArrayLike, locs: Sequence[Tuple[int, ...]]
) -> ArrayLike:
    """Slice concatenated zarrs by locs"""
    # convert array locations to slices
    locs = [slice(*loc) for loc in locs]
    # determine which zarr files are needed
    start, stop = locs[0].start, locs[0].stop  # stack on first axis
    i0 = _zarr_index(offsets, start)
    i1 = _zarr_index(offsets, stop)
    if i0 == i1:  # within a single zarr file
        sel = slice(start - offsets[i0], stop - offsets[i0])
        return zarrs[i0][(sel, *locs[1:])]
    else:  # more than one zarr file
        slices = []
        slices.append((i0, slice(start - offsets[i0], None)))
        for i in range(i0 + 1, i1):  # entire zarr
            slices.append((i, slice(None)))
        if stop > offsets[i1]:
            slices.append((i1, slice(0, stop - offsets[i1])))
        parts = [zarrs[i][(sel, *locs[1:])] for (i, sel) in slices]
        return np.concatenate(parts)
    
if __name__ == '__main__':
    main()