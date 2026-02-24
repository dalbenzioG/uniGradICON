import torch
import numpy as np
import json
import collections
from tqdm import tqdm
import random
import os
import glob
import pickle
import footsteps
import itk
import SimpleITK
from typing import List, Tuple, Optional, Union, Dict, Any
from torch.utils.data import Dataset as TorchDataset


def reorient(moving):
    desired_coordinate_orientation = itk.ITKCommonBasePython.itkSpatialOrientationEnums.ValidCoordinateOrientations_ITK_COORDINATE_ORIENTATION_RAS

    if hasattr(itk, "AnatomicalOrientation"):
        desired_coordinate_orientation = itk.AnatomicalOrientation(desired_coordinate_orientation)

    return itk.orient_image_filter(
        moving, 
        desired_coordinate_orientation=desired_coordinate_orientation,
        use_image_direction=True)


def load_affine_transforms(
    affine_transforms_dir: Optional[str],
    affine_direction: str,
    affine_type: str,
) -> Dict[str, Any]:
    """
    Load pre-computed affine transforms from a directory. Returns a dict keyed by
    subject_id (e.g. '200_R') mapping to ITK transform objects.
    Uses pattern-based discovery: *_US_to_CT_{affine_type}.tfm or *_CT_to_US_{affine_type}.tfm.
    Optional: metadata .pkl with keys 'transforms_us_to_ct' / 'transforms_ct_to_us'.
    Missing dir or no files returns empty dict.
    """
    out = {}
    if not affine_transforms_dir or not os.path.isdir(affine_transforms_dir):
        return out
    dir_path = os.path.abspath(affine_transforms_dir)
    metadata_file = os.path.join(dir_path, f"transforms_metadata_{affine_type}.pkl")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
            key = "transforms_ct_to_us" if affine_direction == "ct_to_us" else "transforms_us_to_ct"
            transforms_dict = metadata.get(key) or metadata.get("transforms", {})
            for subject_id, transform_path in transforms_dict.items():
                if os.path.exists(transform_path):
                    try:
                        out[subject_id] = itk.transformread(transform_path)[0]
                    except Exception as e:
                        print(f"Warning: Failed to load transform for {subject_id}: {e}")
            if out:
                print(f"Loaded {len(out)} affine transforms from metadata ({affine_type}, {affine_direction})")
            return out
        except Exception as e:
            print(f"Warning: Failed to load transforms metadata: {e}")
    pattern_us2ct = f"*_US_to_CT_{affine_type}.tfm"
    pattern_ct2us = f"*_CT_to_US_{affine_type}.tfm"
    pattern = pattern_ct2us if affine_direction == "ct_to_us" else pattern_us2ct
    transform_files = glob.glob(os.path.join(dir_path, pattern))
    if not transform_files:
        transform_files = glob.glob(os.path.join(dir_path, f"*_{affine_type}.tfm"))
    suffix_us2ct = f"_US_to_CT_{affine_type}.tfm"
    suffix_ct2us = f"_CT_to_US_{affine_type}.tfm"
    suffix_generic = f"_{affine_type}.tfm"
    for tf_path in transform_files:
        basename = os.path.basename(tf_path)
        key = basename
        for suf in (suffix_us2ct, suffix_ct2us, suffix_generic):
            if key.endswith(suf):
                key = key[: -len(suf)]
                break
        if not key:
            continue
        try:
            out[key] = itk.transformread(tf_path)[0]
        except Exception as e:
            print(f"Warning: Failed to load transform {tf_path}: {e}")
    if out:
        print(f"Loaded {len(out)} affine transforms from {dir_path} ({affine_type}, {affine_direction})")
    return out

class Dataset(TorchDataset):
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 name: str,
                 data: List[Dict[str, str]],
                 read_type: str = "itk",
                 cache_filename: Optional[str] = None,
                 maximum_images: Optional[int] = None,
                 shuffle: bool = False,
                 is_ct: bool = False,
                 ct_window: Tuple[float, float] = (-1000, 1000),
                 quantile_range: Tuple[float, float] = (0.01, 0.99),
                 use_cache: bool = True):
        
        self.read_type = read_type
        self.name = name
        self.data = data
        self.input_shape = input_shape
        self.is_ct = is_ct
        self.ct_window = ct_window
        self.quantile_range = quantile_range
        self.use_cache = use_cache
        
        if not data:
            raise ValueError(f"Dataset {name}: 'data' must be provided (from JSON source)")
        
        if read_type == "itk":
            self.read_image = self.read_image_itk
        elif read_type == "sitk":
            self.read_image = self.read_image_sitk
        elif read_type == "dicom":
            self.read_image = self.read_image_dicom
        else:
            raise ValueError(f"Invalid read_type: {read_type}. Must be 'itk', 'sitk', or 'dicom'")

        if not use_cache:
            print(f"Loading images without cache...")
            self.store = {}
            paths = self.get_image_paths()
            if shuffle:
                random.shuffle(paths)
            if maximum_images:
                paths = paths[:maximum_images]
            for path in tqdm(paths):
                try:
                    self.store[path] = {"image": self.preprocess_image(path)}
                except Exception as e:
                    print(e)
        elif not cache_filename:
            self.store = {}
            paths = self.get_image_paths()
            if shuffle:
                random.shuffle(paths)
            if maximum_images:
                paths = paths[:maximum_images]
            for path in tqdm(paths):
                try:
                    self.store[path] = {"image": self.preprocess_image(path)}
                except Exception as e:
                    print(e)

            torch.save(
                {
                    "name": self.name,
                    "maximum_images": maximum_images,
                    "store": self.store,
                    "read_type": self.read_type,
                    "is_ct": self.is_ct,
                    "ct_window": self.ct_window,
                    "quantile_range": self.quantile_range,
                },
                footsteps.output_dir + self.name + "_cached_dataset.trch",
            )
        else:
            cache_path = cache_filename + "/" + self.name + "_cached_dataset.trch"
            if os.path.exists(cache_path):
                loaded_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
                
                assert self.name == loaded_cache["name"]
                assert maximum_images == loaded_cache["maximum_images"]
                assert self.read_type == loaded_cache["read_type"]
                assert self.is_ct == loaded_cache["is_ct"]
                if self.is_ct:
                    assert self.ct_window == loaded_cache["ct_window"]
                else:
                    assert self.quantile_range == loaded_cache["quantile_range"]
                
                paths = self.get_image_paths()
                self.store = loaded_cache["store"]
            else:
                os.makedirs(cache_filename, exist_ok=True)
                self.store = {}
                paths = self.get_image_paths()
                if shuffle:
                    random.shuffle(paths)
                if maximum_images:
                    paths = paths[:maximum_images]
                for path in tqdm(paths):
                    try:
                        self.store[path] = {"image": self.preprocess_image(path)}
                    except Exception as e:
                        print(e)
                
                torch.save(
                    {
                        "name": self.name,
                        "maximum_images": maximum_images,
                        "store": self.store,
                        "read_type": self.read_type,
                        "is_ct": self.is_ct,
                        "ct_window": self.ct_window,
                        "quantile_range": self.quantile_range,
                    },
                    cache_path,
                )
            
        self.keys = list(self.store.keys())
        print("Image count: ", len(self.keys))

    def get_image_paths(self) -> List[str]:
        return [item['image'] for item in self.data]

    def read_image_sitk(self, path: str):
        itk_image = SimpleITK.ReadImage(path)
        image = SimpleITK.GetArrayFromImage(itk_image)
        image = torch.tensor(image)
        return image[0]

    def read_image_itk(self, path: str):
        itk_image = reorient(itk.imread(path))
        image = itk.GetArrayFromImage(itk_image)
        image = torch.tensor(image)
        return image
    
    def read_image_dicom(self, path: str):
        namesGenerator = itk.GDCMSeriesFileNames.New()
        namesGenerator.SetUseSeriesDetails(True)
        namesGenerator.SetDirectory(path)
        seriesUID = namesGenerator.GetSeriesUIDs()

        dicom_files = namesGenerator.GetFileNames(seriesUID[0])

        reader = itk.ImageSeriesReader[itk.Image[itk.SS, 3]].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(dicom_files)
        reader.Update()
        image = reader.GetOutput()
        image = reorient(image)

        if (
            "ITK_non_uniform_sampling_deviation"
            in image.GetMetaDataDictionary().GetKeys()
        ):
            spacing_deviation = image.GetMetaDataDictionary().Get(
                "ITK_non_uniform_sampling_deviation"
            )
            spacing_deviation = (
                itk.MetaDataObject[itk.D]
                .cast(spacing_deviation)
                .GetMetaDataObjectValue()
            )

            if spacing_deviation > 5:
                raise ValueError(f"{path}: image has non-uniform-spacing: likely a mish-mash")

        image_array = itk.GetArrayFromImage(image)
        image_tensor = torch.tensor(image_array)
    
        if np.any(np.array(image_array.shape) < 20):
            raise ValueError(f"{path}: image too low resolution")

        return image_tensor
    
    def preprocess_image(self, path: str):
        image = self.read_image(path)

        image = image[None, None]
        image = image.float()
        image = torch.nn.functional.interpolate(
            image, self.input_shape, mode="trilinear"
        )

        im_min = self.ct_window[0] if self.is_ct else torch.quantile(image.view(-1), self.quantile_range[0])
        im_max = self.ct_window[1] if self.is_ct else torch.quantile(image.view(-1), self.quantile_range[1])
        
        image = torch.clip(image, im_min, im_max)
        image = image - im_min
        image = image / (im_max - im_min)

        return image[0]

    def get_image(self, key: str) -> torch.Tensor:
        unprepped_image = self.store[key]["image"]
        return unprepped_image

    def get_key_pair(self) -> Tuple[str, str]:
        return (random.choice(self.keys), random.choice(self.keys))

    def get_pair(self):
        pair = self.get_key_pair()
        return self.get_image(pair[0]), self.get_image(pair[1])
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        return self.get_pair()


class PairedDataset(Dataset):
    def __init__(
        self,
        input_shape,
        name: str,
        data: List[Dict[str, str]],
        cache_filename=None,
        maximum_images=None,
        is_ct: bool = False,
        ct_window: Tuple[float, float] = (-1000, 1000),
        quantile_range: Tuple[float, float] = (0.01, 0.99),
        read_type: str = "itk",
        shuffle: bool = False,
        use_cache: bool = True,
    ):
        super().__init__(
            input_shape,
            name,
            data,
            cache_filename=cache_filename,
            maximum_images=maximum_images,
            is_ct=is_ct,
            ct_window=ct_window,
            quantile_range=quantile_range,
            read_type=read_type,
            shuffle=shuffle,
            use_cache=use_cache,
        )

        self.pair_lookup = collections.defaultdict(list)
        self.pair_keys = {}

        for item in self.data:
            path = item['image']
            subject_id = item.get('subject_id')
            if path in self.store and subject_id:
                self.pair_keys[path] = subject_id
                self.pair_lookup[subject_id].append(path)
        
        self.keys = [k for k in self.keys if k in self.pair_keys and len(self.pair_lookup[self.pair_keys[k]]) > 1]
        print("Paired image count: ", len(self.keys))

    def get_key_pair(self):
        image_key_1 = random.choice(self.keys)
        subject_id = self.pair_keys[image_key_1]
        candidates = [k for k in self.pair_lookup[subject_id] if k != image_key_1]
        image_key_2 = random.choice(candidates)
        return (image_key_1, image_key_2)

class ImageSegmentationDataset(Dataset):
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 name: str,
                 data: List[Dict[str, str]],
                 read_type: str = "itk",
                 cache_filename: Optional[str] = None,
                 maximum_images: Optional[int] = None,
                 shuffle: bool = False,
                 is_ct: bool = False,
                 ct_window: Tuple[float, float] = (-1000, 1000),
                 quantile_range: Tuple[float, float] = (0.01, 0.99),
                 use_cache: bool = True):
        
        self.segmentation_map = {item['image']: item['segmentation'] for item in data}
        
        super().__init__(input_shape=input_shape,
                         name=name,
                         data=data,
                         read_type=read_type,
                         cache_filename=cache_filename,
                         maximum_images=maximum_images,
                         shuffle=shuffle,
                         is_ct=is_ct,
                         ct_window=ct_window,
                         quantile_range=quantile_range,
                         use_cache=use_cache)
        
        for path in self.keys:
            try:
                self.store[path]["segmentation"] = self.preprocess_segmentation(path)
            except Exception as e:
                print(f"Failed to process segmentation for {path}: {e}")
    
    def get_image_paths(self) -> List[str]:
         return [item['image'] for item in self.data]

    def get_segmentation_path(self, image_path: str) -> Optional[str]:
         return self.segmentation_map.get(image_path)

    def preprocess_segmentation(self, image_path: str) -> torch.Tensor:
        seg_path = self.get_segmentation_path(image_path)
        seg = self.read_image(seg_path)
        seg = seg[None, None].float()
        seg = torch.nn.functional.interpolate(seg, self.input_shape, mode="nearest")
        return seg[0]

    def get_image(self, key: str) -> torch.Tensor:
        return self.store[key]["image"]

    def get_segmentation(self, key: str) -> torch.Tensor:
        return self.store[key]["segmentation"]

    def get_pair(self):
        pair = self.get_key_pair()
        return (
            self.get_image(pair[0]),
            self.get_image(pair[1]),
            self.get_segmentation(pair[0]),
            self.get_segmentation(pair[1]),
        )
        
class PairedImageSegmentationDataset(ImageSegmentationDataset):
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 name: str,
                 data: List[Dict[str, str]],
                 read_type: str = "itk",
                 cache_filename: Optional[str] = None,
                 maximum_images: Optional[int] = None,
                 shuffle: bool = False,
                 is_ct: bool = False,
                 ct_window: Tuple[float, float] = (-1000, 1000),
                 quantile_range: Tuple[float, float] = (0.01, 0.99),
                 use_cache: bool = True):
        
        super().__init__(input_shape=input_shape,
                         name=name,
                         data=data,
                         read_type=read_type,
                         cache_filename=cache_filename,
                         maximum_images=maximum_images,
                         shuffle=shuffle,
                         is_ct=is_ct,
                         ct_window=ct_window,
                         quantile_range=quantile_range,
                         use_cache=use_cache)

        self.pair_lookup = collections.defaultdict(list)
        self.pair_keys = {}
        
        for item in self.data:
            path = item['image']
            subject_id = item.get('subject_id')
            if path in self.store and subject_id:
                self.pair_keys[path] = subject_id
                self.pair_lookup[subject_id].append(path)

        self.keys = [k for k in self.keys if k in self.pair_keys and len(self.pair_lookup[self.pair_keys[k]]) > 1]
        print("Paired segmentation count:", len(self.keys))

    def get_key_pair(self) -> Tuple[str, str]:
        image_key_1 = random.choice(self.keys)
        subject_id = self.pair_keys[image_key_1]
        candidates = [k for k in self.pair_lookup[subject_id] if k != image_key_1]
        image_key_2 = random.choice(candidates)
        return (image_key_1, image_key_2)

    def get_pair(self):
        k1, k2 = self.get_key_pair()
        return (
            self.get_image(k1),
            self.get_image(k2),
            self.get_segmentation(k1),
            self.get_segmentation(k2),
        )


class PairedCTUSDataset(TorchDataset):
    """
    Paired CT-US dataset: one CT and one US per subject_id, with modality-aware
    preprocessing (CT: ct_window; US: quantile_range). Always returns (moving=US, fixed=CT).
    Optional affine pre-transform: resample US to CT space (us_to_ct) or CT to US space (ct_to_us).
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        name: str,
        data: List[Dict[str, str]],
        read_type: str = "itk",
        cache_filename: Optional[str] = None,
        maximum_images: Optional[int] = None,
        shuffle: bool = False,
        ct_window: Tuple[float, float] = (-1000, 1000),
        quantile_range: Tuple[float, float] = (0.01, 0.99),
        use_cache: bool = True,
        affine_transforms_dir: Optional[str] = None,
        affine_direction: str = "us_to_ct",
        affine_type: str = "rigid_mask",
    ):
        self.name = name
        self.data = data
        self.input_shape = input_shape
        self.ct_window = ct_window
        self.quantile_range = quantile_range
        self.read_type = read_type
        self.cache_filename = cache_filename
        self.maximum_images = maximum_images
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.affine_transforms_dir = affine_transforms_dir
        self.affine_direction = affine_direction
        self.affine_type = affine_type

        if not data:
            raise ValueError(f"Dataset {name}: 'data' must be provided (from JSON source)")

        if read_type == "itk":
            self.read_image = self._read_image_itk
        else:
            raise ValueError(f"PairedCTUSDataset only supports read_type='itk', got {read_type}")

        path_to_modality = {}
        path_to_subject_id = {}
        for item in data:
            path = item["image"]
            mod = item.get("modality")
            if mod not in ("ct", "us"):
                raise ValueError(
                    f"Each entry must have modality 'ct' or 'us'; got {mod} for path {path}"
                )
            path_to_modality[path] = mod
            sid = item.get("subject_id")
            if not sid:
                raise ValueError(f"Missing subject_id for path {path}")
            path_to_subject_id[path] = sid

        subject_paths = collections.defaultdict(list)
        for item in data:
            path = item["image"]
            sid = item.get("subject_id")
            subject_paths[sid].append((path, path_to_modality[path]))

        subject_to_pair = {}
        for sid, paths_with_mod in subject_paths.items():
            ct_paths = [p for p, m in paths_with_mod if m == "ct"]
            us_paths = [p for p, m in paths_with_mod if m == "us"]
            if len(ct_paths) != 1 or len(us_paths) != 1:
                raise ValueError(
                    f"Subject {sid} must have exactly one CT and one US entry; "
                    f"got {len(ct_paths)} CT, {len(us_paths)} US"
                )
            subject_to_pair[sid] = (us_paths[0], ct_paths[0])

        self.path_to_modality = path_to_modality
        self.path_to_subject_id = path_to_subject_id
        self.subject_to_pair = subject_to_pair
        self.keys = sorted(subject_to_pair.keys())
        if maximum_images:
            self.keys = self.keys[:maximum_images]
        if shuffle:
            random.shuffle(self.keys)

        self.affine_transforms = load_affine_transforms(
            affine_transforms_dir, affine_direction, affine_type
        )

        all_paths = list(path_to_modality.keys())
        cache_meta = {
            "affine_transforms_dir": affine_transforms_dir,
            "affine_direction": affine_direction,
            "affine_type": affine_type,
        }
        if cache_filename and use_cache:
            cache_path = os.path.join(cache_filename, self.name + "_paired_ct_us_cached_dataset.trch")
            if os.path.exists(cache_path):
                loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
                if (
                    loaded.get("path_to_modality") == path_to_modality
                    and loaded.get("name") == self.name
                    and loaded.get("maximum_images") == maximum_images
                    and loaded.get("affine_transforms_dir") == cache_meta["affine_transforms_dir"]
                    and loaded.get("affine_direction") == cache_meta["affine_direction"]
                    and loaded.get("affine_type") == cache_meta["affine_type"]
                ):
                    self.store = loaded["store"]
                    print(f"PairedCTUSDataset '{self.name}': loaded {len(self.store)} images from cache")
                else:
                    self.store = self._build_store(all_paths)
                    os.makedirs(cache_filename, exist_ok=True)
                    torch.save(
                        {
                            "name": self.name,
                            "maximum_images": maximum_images,
                            "store": self.store,
                            "path_to_modality": path_to_modality,
                            **cache_meta,
                        },
                        cache_path,
                    )
            else:
                self.store = self._build_store(all_paths)
                os.makedirs(cache_filename, exist_ok=True)
                torch.save(
                    {
                        "name": self.name,
                        "maximum_images": maximum_images,
                        "store": self.store,
                        "path_to_modality": path_to_modality,
                        **cache_meta,
                    },
                    cache_path,
                )
        else:
            self.store = self._build_store(all_paths)

        print(f"PairedCTUSDataset '{self.name}': {len(self.keys)} pairs (moving=US, fixed=CT)")

    def _read_image_itk(self, path: str) -> torch.Tensor:
        itk_image = reorient(itk.imread(path))
        image = itk.GetArrayFromImage(itk_image)
        return torch.tensor(image)

    def _resample_to_reference(
        self, moving_path: str, reference_path: str, transform: Any
    ) -> torch.Tensor:
        """Resample moving image into reference image space using ITK; return tensor."""
        moving_itk = itk.imread(moving_path, itk.F)
        reference_itk = itk.imread(reference_path, itk.F)
        moving_itk = itk.CastImageFilter[type(moving_itk), itk.Image[itk.F, 3]].New()(moving_itk)
        reference_itk = itk.CastImageFilter[type(reference_itk), itk.Image[itk.F, 3]].New()(
            reference_itk
        )
        ResampleFilterType = itk.ResampleImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]]
        resample = ResampleFilterType.New()
        resample.SetTransform(transform)
        resample.SetInput(moving_itk)
        resample.SetReferenceImage(reference_itk)
        resample.SetUseReferenceImage(True)
        resample.SetDefaultPixelValue(0.0)
        InterpolatorType = itk.LinearInterpolateImageFunction[itk.Image[itk.F, 3], itk.D]
        resample.SetInterpolator(InterpolatorType.New())
        resample.Update()
        out = resample.GetOutput()
        arr = np.asarray(out)
        return torch.tensor(arr)

    def _resample_seg_to_reference(
        self, moving_seg_path: str, reference_path: str, transform: Any
    ) -> torch.Tensor:
        """Resample moving segmentation into reference image space (nearest-neighbor); return tensor."""
        moving_itk = itk.imread(moving_seg_path, itk.F)
        reference_itk = itk.imread(reference_path, itk.F)
        moving_itk = itk.CastImageFilter[type(moving_itk), itk.Image[itk.F, 3]].New()(moving_itk)
        reference_itk = itk.CastImageFilter[type(reference_itk), itk.Image[itk.F, 3]].New()(
            reference_itk
        )
        ResampleFilterType = itk.ResampleImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]]
        resample = ResampleFilterType.New()
        resample.SetTransform(transform)
        resample.SetInput(moving_itk)
        resample.SetReferenceImage(reference_itk)
        resample.SetUseReferenceImage(True)
        resample.SetDefaultPixelValue(0.0)
        interpolator = itk.NearestNeighborInterpolateImageFunction.New(moving_itk)
        resample.SetInterpolator(interpolator)
        resample.Update()
        out = resample.GetOutput()
        arr = np.asarray(out)
        return torch.tensor(arr)

    def _preprocess_by_modality(self, image: torch.Tensor, modality: str) -> torch.Tensor:
        image = image[None, None].float()
        image = torch.nn.functional.interpolate(image, self.input_shape, mode="trilinear")
        if modality == "ct":
            im_min = self.ct_window[0]
            im_max = self.ct_window[1]
        else:
            im_min = torch.quantile(image.view(-1), self.quantile_range[0])
            im_max = torch.quantile(image.view(-1), self.quantile_range[1])
        image = torch.clip(image, im_min, im_max)
        image = image - im_min
        image = image / (im_max - im_min + 1e-8)
        return image[0]

    def _build_store(self, paths: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        store = {}
        for path in tqdm(paths, desc=f"Loading PairedCTUSDataset {self.name}"):
            try:
                modality = self.path_to_modality[path]
                subject_id = self.path_to_subject_id[path]
                transform = self.affine_transforms.get(subject_id) if self.affine_transforms else None
                us_path, ct_path = self.subject_to_pair[subject_id]
                if transform is not None:
                    if self.affine_direction == "us_to_ct" and modality == "us":
                        image = self._resample_to_reference(us_path, ct_path, transform)
                    elif self.affine_direction == "ct_to_us" and modality == "ct":
                        image = self._resample_to_reference(ct_path, us_path, transform)
                    else:
                        image = self.read_image(path)
                else:
                    image = self.read_image(path)
                processed = self._preprocess_by_modality(image, modality)
                store[path] = {"image": processed}
            except Exception as e:
                print(f"Failed to load {path}: {e}")
        return store

    def get_image(self, path: str) -> torch.Tensor:
        return self.store[path]["image"]

    def get_key_pair(self) -> Tuple[str, str]:
        subject_id = random.choice(self.keys)
        return self.subject_to_pair[subject_id]

    def get_pair(self) -> Tuple[torch.Tensor, torch.Tensor]:
        us_path, ct_path = self.get_key_pair()
        return self.get_image(us_path), self.get_image(ct_path)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_pair()


class PairedCTUSSegmentationDataset(PairedCTUSDataset):
    """
    Paired CT-US with segmentation: same as PairedCTUSDataset but each entry has
    a segmentation path. Returns (moving_image, fixed_image, moving_seg, fixed_seg)
    for finetune_multi_segmentation (Dice loss via mask_A, mask_B).
    When an image is resampled by an affine transform, its segmentation is resampled
    with the same transform and reference grid.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        name: str,
        data: List[Dict[str, str]],
        read_type: str = "itk",
        cache_filename: Optional[str] = None,
        maximum_images: Optional[int] = None,
        shuffle: bool = False,
        ct_window: Tuple[float, float] = (-1000, 1000),
        quantile_range: Tuple[float, float] = (0.01, 0.99),
        use_cache: bool = True,
        affine_transforms_dir: Optional[str] = None,
        affine_direction: str = "us_to_ct",
        affine_type: str = "rigid_mask",
    ):
        for item in data:
            if "segmentation" not in item:
                raise ValueError(
                    f"PairedCTUSSegmentationDataset requires 'segmentation' in every entry; "
                    f"missing for image {item.get('image', '?')}. "
                    "Rebuild the JSON with build_trusted_pairs_json so every pair has segmentation paths. "
                    "Expected naming: 200R_imgCT.nii.gz -> 200R_segCT.nii.gz, 200R_imgUS.nii.gz -> 200R_segUS.nii.gz "
                    "(same directory). Use --seg_suffix_ct / --seg_suffix_us if your files use different suffixes, "
                    "or run with --require_segmentation to fail early. If you have no segmentations, use dataset type "
                    "'paired_ct_us' instead of 'paired_ct_us_seg'."
                )
        super().__init__(
            input_shape=input_shape,
            name=name,
            data=data,
            read_type=read_type,
            cache_filename=cache_filename,
            maximum_images=maximum_images,
            shuffle=shuffle,
            ct_window=ct_window,
            quantile_range=quantile_range,
            use_cache=use_cache,
            affine_transforms_dir=affine_transforms_dir,
            affine_direction=affine_direction,
            affine_type=affine_type,
        )
        seg_map = {item["image"]: item["segmentation"] for item in data}
        for path in tqdm(list(self.store.keys()), desc=f"Loading segmentations {self.name}"):
            try:
                seg_path = seg_map[path]
                subject_id = self.path_to_subject_id[path]
                modality = self.path_to_modality[path]
                us_path, ct_path = self.subject_to_pair[subject_id]
                transform = self.affine_transforms.get(subject_id) if self.affine_transforms else None
                if transform is not None:
                    if self.affine_direction == "us_to_ct" and modality == "us":
                        seg = self._resample_seg_to_reference(seg_path, ct_path, transform)
                    elif self.affine_direction == "ct_to_us" and modality == "ct":
                        seg = self._resample_seg_to_reference(seg_path, us_path, transform)
                    else:
                        seg = self.read_image(seg_path)
                else:
                    seg = self.read_image(seg_path)
                seg = seg[None, None].float()
                seg = torch.nn.functional.interpolate(seg, self.input_shape, mode="nearest")
                self.store[path]["segmentation"] = seg[0]
            except Exception as e:
                raise RuntimeError(f"Failed to load segmentation for {path}: {e}") from e
        print(f"PairedCTUSSegmentationDataset '{self.name}': segmentations loaded")

    def get_segmentation(self, path: str) -> torch.Tensor:
        return self.store[path]["segmentation"]

    def get_pair(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        us_path, ct_path = self.get_key_pair()
        return (
            self.get_image(us_path),
            self.get_image(ct_path),
            self.get_segmentation(us_path),
            self.get_segmentation(ct_path),
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.get_pair()
