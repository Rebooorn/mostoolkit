import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from io_utils import *
import scipy.ndimage
from skimage.transform import resize, rescale
import multiprocessing
from contextlib import closing

# slice volumes for 2D models
def slice_volume_worker_fn(vol, to_dir, keys):
    # params: 
    #   vol: pathlib path of hdf5 volume
    #   to_dir: where the slices are stored.
    #   keys: keys in hdf5 to slice, ['ct', 'seg'] when no helpers are used
    id = vol.stem
    vol_hdf = h5py.File(vol, 'r')
    vol_data = []
    for k in keys:
        vol_data.append(np.asarray(vol_hdf[k]))
    # vol_ct = np.asarray(vol_data['ct'])
    # vol_seg = np.array(vol_data['seg'])

    # vol_ct.shape = [nslices, w, h]
    nslices = vol_data[0].shape[0]
    for i in range(nslices):
        slice_data = []
        for j, _ in enumerate(keys):
            slice_data.append(vol_data[j][i])
        # slice_ct = vol_ct[i]
        # slice_seg = vol_seg[i]

        with h5py.File(str(to_dir / '{}_{}.h5'.format(id, i)), 'w') as f:
            for j, k in enumerate(keys):
                f.create_dataset(k, slice_data[j].shape, slice_data[j].dtype, slice_data[j])

            # f.create_dataset("ct", slice_ct.shape, slice_ct.dtype, slice_ct)
            # f.create_dataset("seg", slice_seg.shape, slice_seg.dtype, slice_seg)
    # print('Slicing {} done.'.format(id))
    del id, vol_data
    return 0

def slice_volume(data, to_dir, num_workers=2, keys=['ct', 'seg']):
    print(keys)
    args = [(vol, to_dir, keys) for vol in data]
    if num_workers == 0:
        for arg in args:
            slice_volume_worker_fn(*arg)
    else:
        with closing(multiprocessing.Pool(processes=num_workers)) as p:
            res = p.starmap(slice_volume_worker_fn, args)

# clip volumes into patches for 3D volumes
def clip_volume_worker_fn(vol, to_dir, patch_size, stride, keys):
    # params: see slice_volume_worker_fn()
    id = vol.stem
    vol_hdf = h5py.File(vol, 'r')
    vol_data = []
    for k in keys:
        vol_data.append(np.asarray(vol_hdf[k]))
    # vol_ct = np.asarray(vol_hdf['ct'])
    # vol_seg = np.array(vol_hdf['seg'])

    # vol_ct.shape = [d(nslices), w, h]
    # d, w, h = vol_ct.shape
    d, w, h = vol_data[0].shape
    if patch_size[1] // 2 == w - patch_size[1] // 2:
        # take the full w as patch
        w_grid_centers = np.array([patch_size[1] // 2])
    else:
        w_grid_centers = np.arange(patch_size[1] // 2, w - patch_size[1] // 2, stride[1])
    
    if patch_size[2] // 2 == h - patch_size[2] // 2:
        h_grid_centers = np.array([patch_size[2]//2])
    else:
        h_grid_centers = np.arange(patch_size[2] // 2, h - patch_size[2] // 2, stride[2])
    
    if patch_size[0] // 2 == d - patch_size[0] // 2:
        d_grid_centers = np.array([patch_size[0] // 2])
    else:
        d_grid_centers = np.arange(patch_size[0] // 2, d - patch_size[0] // 2, stride[0])

    if w % patch_size[1] != 0: np.append(w_grid_centers, w - patch_size[1] // 2)
    if h % patch_size[2] != 0: np.append(h_grid_centers, h - patch_size[2] // 2)
    if d % patch_size[0] != 0: np.append(d_grid_centers, d - patch_size[0] // 2)
    for ni, i in enumerate(w_grid_centers):
            for nj, j in enumerate(h_grid_centers):
                for nk, k in enumerate(d_grid_centers):
                    patch_data = []
                    for ki, _ in enumerate(keys):
                        patch_data.append(vol_data[ki][k - int(patch_size[0] // 2): k + int(patch_size[0] // 2),
                                       i - int(patch_size[1] // 2): i + int(patch_size[1] // 2),
                                       j - int(patch_size[2] // 2): j + int(patch_size[2] // 2),
                                      ])
                    # patch_img = vol_ct[k - int(patch_size[0] // 2): k + int(patch_size[0] // 2),
                    #                    i - int(patch_size[1] // 2): i + int(patch_size[1] // 2),
                    #                    j - int(patch_size[2] // 2): j + int(patch_size[2] // 2),
                    #                   ]
                    # patch_seg = vol_seg[k - int(patch_size[0] // 2): k + int(patch_size[0] // 2),
                    #                     i - int(patch_size[1] // 2): i + int(patch_size[1] // 2),
                    #                     j - int(patch_size[2] // 2): j + int(patch_size[2] // 2),
                    #                    ]

                    with h5py.File(str(to_dir / '{}_{}_{}_{}.h5'.format(id, nk, ni, nj)), 'w') as f:
                        for ki, k in enumerate(keys):
                            f.create_dataset(k, patch_data[ki].shape, patch_data[ki].dtype, patch_data[ki])
                        # f.create_dataset("ct", patch_img.shape, patch_img.dtype, patch_img)
                        # f.create_dataset("seg", patch_seg.shape, patch_seg.dtype, patch_seg)
    # print('Clip {} done.'.format(id))
    del id, vol_data
    return 0

def clip_volume(data, to_dir, patch_size=(64, 64, 64), stride=(64, 64, 64), num_workers=2, keys=['ct', 'seg']):

    args = [(vol, to_dir, patch_size, stride, keys) for vol in data]
    if num_workers == 1:
        for arg in args:
            clip_volume_worker_fn(*arg)
    else:
        with closing(multiprocessing.Pool(processes=num_workers)) as p:
            res = p.starmap(clip_volume_worker_fn, args)

def preprocessing_worker_fn(data, tar_dir, dataroot):
    image_dir = dataroot / Path(data['image'])
    label_dir = dataroot / Path(data['label'])

    # # specially for nnUNet's wierd naming
    # image_dir = Path(str(image_dir).replace('.nii.gz', '_0000.nii.gz'))
    # # label_dir = Path(str(label_dir).replace('.nii.gz', '_0000.nii.gz'))

    # preprocess image
    vol, _, pdim, _, _ = sitk_load_with_metadata(str(image_dir))
    
    vol[vol==np.amin(vol)] = -1024
    vol += 1024
    vol[vol < 0] = 0
    vol[vol > 4095] = 4095

    vol_standard_pdim = resample(vol, pdim, new_spacing=[1.0, 1.0, -1.0])[0]
    vol_standard_pdim_size = crop_to_standard(vol_standard_pdim, scale=320)

    # the label
    lab, _, pdim, _, _ = sitk_load_with_metadata(str(label_dir))

    lab_standard_pdim = resample(lab, pdim, new_spacing=[1.0, 1.0, -1.0], order=0)[0]
    lab_standard_pdim_size = crop_to_standard(lab_standard_pdim, scale=320).astype(np.uint8)

    output_shape = (vol_standard_pdim_size.shape[0], 256, 256)
    ct = resize(vol_standard_pdim_size, output_shape=output_shape, preserve_range=True).astype(np.int16)
    # slice_visualize_X(lab_standard_pdim_size)
    seg = resize(lab_standard_pdim_size, output_shape=output_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

    # save to .h5
    # example label_dir.name: segmentation-0.nii
    hdf5_fname = tar_dir / 'lits-{}.h5'.format(label_dir.name[13: -4])
    with h5py.File(str(hdf5_fname), 'w') as hf:
        hf.create_dataset('ct', ct.shape, ct.dtype, ct)
        hf.create_dataset("seg", seg.shape, seg.dtype, seg)

    print("Preprocess {} done".format(label_dir.stem))
    del vol, vol_standard_pdim, vol_standard_pdim_size, 
    return 0

def preprocessing(dataroot, tar_dir, num_workers=0):
    # from debug_utils import slice_visualize_XY
    # Things happened in the preprocessing pipeline:
    # 1. Change intensity range to [0, 4095]
    # 2. resample to pixdim = (1, 1, -1)
    # 3. crop to (320, 320, -1)
    # 4. resize to (256,256, -1) for training.
    
    
    # Note for inference:
    # 1. resample to pd = (1, 1, 5)
    # 2. resize (skimage.rescale) by factor = 256/320

    if isinstance(dataroot, str):
        dataroot = Path(dataroot)
    data_list = [{
        'image': str(dataroot / 'volumes' / 'volume-{}.nii'.format(i)),
        'label': str(dataroot / 'seg_masks' / 'segmentation-{}.nii'.format(i))
    } for i in range(116, 131)]

    for data in data_list:
        assert Path(data['image']).exists() and Path(data['label']).exists
    
    if num_workers == 0:
        for data in data_list:
            preprocessing_worker_fn(data, tar_dir, dataroot)
    else:
        args = [(data, tar_dir, dataroot) for data in data_list]
        with closing(multiprocessing.Pool(processes=num_workers)) as pool:
            res = pool.starmap(preprocessing_worker_fn, args)
        # break

def preprocessing_infer(data_dir, tar_dir):
    # data should be *.nii.gz
    data_dir = Path(data_dir)
    tar_dir = Path(tar_dir)
    # we need to record metadata for save
    metadata = dict()

    print('Preprocess for inference')
    # Note that Lits are kept in .nii
    infer_imgs = list(data_dir.glob('*.nii.gz'))
    pbar = tqdm(total=len(infer_imgs))
    for infer_fname in infer_imgs:
        infer_fname = data_dir / Path(infer_fname).name
        vol, direction, spacing, origin, size = sitk_load_with_metadata(str(infer_fname))
        
        vol[vol==np.amin(vol)] = -1024
        vol += 1024
        vol[vol < 0] = 0
        vol[vol > 4095] = 4095

        metadata[infer_fname.name] = {
            'direction': direction,
            'spacing': spacing,
            'origin': origin,
            'size': size,
        }
        vol_standard_pdim = resample(vol, spacing, new_spacing=[1.0, 1.0, -1.0])[0]
        l, w, h = vol_standard_pdim.shape
        output_shape = (l, int(w * 256 / 320), int(h * 256 / 320))
        ct = resize(vol_standard_pdim, output_shape=output_shape, preserve_range=True).astype(np.int16)
        # vol_standard_pdim_size = crop_to_standard(vol_standard_pdim, scale=320)
        
        with h5py.File(str(tar_dir / (infer_fname.name.replace('.nii.gz', '.h5'))), 'w') as hf:
            hf.create_dataset('ct', ct.shape, ct.dtype, ct)
            
        pbar.update(1)
    pbar.close()
    metadata_path = tar_dir / 'metadata.json'
    save_json(metadata_path, metadata)
    print('Preprocess infer data done.')
    return metadata_path

def postprocessing_infer(data_dir, tar_dir, metadata):
    # from skimage.transform import rescale
    from mostoolkit.vis_utils import slice_visualize_X
    data_dir = Path(data_dir)
    tar_dir = Path(tar_dir)
    metadata_dict = load_json(str(metadata))

    print('Post-process and generate final segmentation')
    pbar = tqdm(total=len(list(data_dir.glob('*.nii.gz'))))
    for ini_seg_fname in data_dir.glob('*.nii.gz'):
        seg, _, spacing, _, _ = sitk_load_with_metadata(str(ini_seg_fname))
        if str(ini_seg_fname).endswith('_dn.nii.gz'):
            # denoised image
            metadata = metadata_dict[ini_seg_fname.name.replace('_dn', '')]
            resample_order = 3
            res_dtype = np.int16
        else:
            metadata = metadata_dict[ini_seg_fname.name]
            resample_order = 0
            res_dtype = np.uint8

        # print(np.unique(seg))
        l, w, h = seg.shape
        output_shape = (l, int(w * 320 / 256), int(h * 320 / 256))
        # slice_visualize_X(seg)
        seg = resize(seg, output_shape=output_shape, preserve_range=True, anti_aliasing=False, order=resample_order)
        # slice_visualize_X(seg)
        seg_ori_pixdim = resample(seg, spacing=spacing, new_spacing=metadata['spacing'], order=resample_order)[0]
        # slice_visualize_X(seg_ori_pixdim)
        # seg = seg[:, ::-1, ::-1]

        # create a bigger container on purpose
        res = np.zeros([i + 10 for i in metadata['size'][::-1]]).astype(res_dtype)
        # seg is smaller than the original image
        res[:seg_ori_pixdim.shape[0], :seg_ori_pixdim.shape[1], :seg_ori_pixdim.shape[2]] = seg_ori_pixdim
        sh = metadata['size'][::-1]
        res = res[:sh[0],:sh[1],:sh[2]]
      
        # print(metadata['size'], seg.shape, res.shape)
        
        # print(np.unique(seg))
        # save to .nii.gz
        save_fname = tar_dir / ini_seg_fname.name
        sitk_save(str(save_fname), res, metadata['spacing'], metadata['origin'], metadata['direction'])
        pbar.update(1)
        # break
    pbar.close()

def preprocess_cli_main(dataroot, pro_dir):
    pro_dir = Path(pro_dir)
    pro_dir.mkdir(exist_ok=True)
    preprocessing(dataroot, pro_dir, num_workers=8)
    # slice_volume(pro_dir, pro_dir/'slices_train', num_workers=4)
    print('all done')

def fuse_hdf5_datasets(main_dataset, aux_dataset):
    # Only works for hdf5 dataset
    # params:
    #   main_dataset: directory of main dataset
    #   aux_dataset: directory of aux_dataset:
    main_data = sorted(list(Path(main_dataset).glob('*.h5')))
    aux_dataset = Path(aux_dataset)

    # check aux dataset
    for m in main_data:
        assert (aux_dataset / m.name).exists()

    for m in main_data:
        a = aux_dataset / m.name
        with h5py.File(m, 'a') as f:
            aux = h5py.File(a, 'r')
            aux_keys = list(aux.keys())
            for k in aux_keys:
                data = np.asarray(aux[k])
                f.create_dataset(k, shape=data.shape, dtype=data.dtype, data=data)
            print(str(m.name), ' is updated')
    
def dataset_analysis(ds_path):
    import nibabel as nib
    if isinstance(ds_path, str):
        ds_path = Path(ds_path)

    for data in ds_path.glob('*.nii'):
        img, _, _, _, size = sitk_load_with_metadata(str(data))
        # img = nib.Nifti1Image.from_filename(str(data)).get_fdata()
        print(img.shape)

if __name__ == '__main__':
    # dataroot = r'D:\Data\AMOS22\AMOS22'
    pro_dir = r'D:\Data\LiTS17\tmp\pro'
    dataroot = r'D:\Data\LiTS17\tmp'
    preprocess_cli_main(dataroot, pro_dir)

    # pro_result = Path(r'D:\Chang\filter_and_segmentation\lightning_logs\filterseg\version_504178\eval\last\pro_result')
    # result = Path(r'D:\Chang\filter_and_segmentation\lightning_logs\filterseg\version_504178\eval\last\results')
    # metadata = pro_result.parent / 'pro_infer' / 'metadata.json'
    # result.mkdir(exist_ok=True)
    # postprocessing_infer(pro_result, result, metadata)
    # main_dataset = r'D:\Data\AMOS22\AMOS22\pro_training'
    # aux_dataset = r'D:\Data\AMOS22\AMOS22\ctorg_masksTr\pro_helper'
    # fuse_hdf5_datasets(main_dataset, aux_dataset)

    # dataset_analysis(r'D:\Data\LiTS17\volumes')