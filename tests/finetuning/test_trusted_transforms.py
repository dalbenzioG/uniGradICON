"""Tests for TRUSTED-style init-transform integration in dataset loading."""
import pytest
import torch

from unigradicon.finetuning import dataset


def _write_pair_files(tmp_path, with_segmentation=False):
    ct_img = tmp_path / "200R_imgCT.nii.gz"
    us_img = tmp_path / "200R_imgUS.nii.gz"
    ct_img.write_bytes(b"x")
    us_img.write_bytes(b"x")
    entries = [
        {"image": str(ct_img), "subject_id": "200_R", "modality": "ct"},
        {"image": str(us_img), "subject_id": "200_R", "modality": "us"},
    ]
    if with_segmentation:
        ct_seg = tmp_path / "200R_seg.nii.gz"
        us_seg = tmp_path / "200R_maskUS.nii.gz"
        ct_seg.write_bytes(b"x")
        us_seg.write_bytes(b"x")
        entries[0]["segmentation"] = str(ct_seg)
        entries[1]["segmentation"] = str(us_seg)
    return entries


class _FakeTransform:
    def GetInverse(self):
        return self


class _FakeResampler:
    def __init__(self):
        self._moving = None

    def SetReferenceImage(self, _fixed):
        pass

    def SetTransform(self, _tx):
        pass

    def SetInterpolator(self, _interp):
        pass

    def SetDefaultPixelValue(self, _value):
        pass

    def Execute(self, moving):
        self._moving = moving
        return moving


class _FakeSitk:
    sitkLinear = 1
    sitkNearestNeighbor = 0

    @staticmethod
    def ReadTransform(_path):
        return _FakeTransform()

    @staticmethod
    def ReadImage(path):
        return path

    @staticmethod
    def ResampleImageFilter():
        return _FakeResampler()

    @staticmethod
    def DICOMOrient(image, _orientation):
        return image

    @staticmethod
    def GetArrayFromImage(_image):
        return torch.zeros(8, 8, 8).numpy()


def test_transform_settings_partition_cache_signature(tmp_path, fake_image_reader):
    data = _write_pair_files(tmp_path)
    cache_dir = tmp_path / "cache"
    transform_dir_a = tmp_path / "init_a"
    transform_dir_b = tmp_path / "init_b"
    transform_dir_a.mkdir()
    transform_dir_b.mkdir()

    ds_a = dataset.Dataset(
        input_shape=(8, 8, 8),
        name="trusted",
        data=data,
        cache_dir=str(cache_dir),
        use_cache=True,
        init_transform_dir=str(transform_dir_a),
    )
    ds_b = dataset.Dataset(
        input_shape=(8, 8, 8),
        name="trusted",
        data=data,
        cache_dir=str(cache_dir),
        use_cache=True,
        init_transform_dir=str(transform_dir_b),
    )
    assert ds_a.cache.signature != ds_b.cache.signature


def test_missing_transform_fails_fast(tmp_path, fake_image_reader):
    data = _write_pair_files(tmp_path)
    transform_dir = tmp_path / "init_transf"
    transform_dir.mkdir()

    with pytest.raises(RuntimeError, match="transform-aware loading failed"):
        dataset.Dataset(
            input_shape=(8, 8, 8),
            name="trusted",
            data=data,
            use_cache=False,
            init_transform_dir=str(transform_dir),
            init_transform_direction="ct_to_us",
        )


def test_directionality_controls_which_modality_is_resampled(tmp_path, fake_image_reader, monkeypatch):
    data = _write_pair_files(tmp_path)
    transform_dir = tmp_path / "init_transf"
    transform_dir.mkdir()
    (transform_dir / "200R_CT_to_US_pca_icp.tfm").write_text("fake", encoding="utf-8")

    calls = []

    def _spy_resample(self, moving, fixed, tx_moving_to_fixed, interpolator):
        calls.append((moving, fixed, interpolator, tx_moving_to_fixed))
        return moving

    monkeypatch.setattr(dataset.Dataset, "_get_sitk", lambda self: _FakeSitk)
    monkeypatch.setattr(dataset.Dataset, "_resample_moving_to_fixed", _spy_resample)

    dataset.Dataset(
        input_shape=(8, 8, 8),
        name="trusted_ct_to_us",
        data=data,
        use_cache=False,
        init_transform_dir=str(transform_dir),
        init_transform_direction="ct_to_us",
    )
    assert any("imgCT" in moving for moving, _, _, _ in calls)
    assert all("imgUS" in fixed for _, fixed, _, _ in calls)

    calls.clear()
    dataset.Dataset(
        input_shape=(8, 8, 8),
        name="trusted_us_to_ct",
        data=data,
        use_cache=False,
        init_transform_dir=str(transform_dir),
        init_transform_direction="us_to_ct",
    )
    assert any("imgUS" in moving for moving, _, _, _ in calls)
    assert all("imgCT" in fixed for _, fixed, _, _ in calls)


def test_label_maps_use_nearest_neighbor_for_resampling(tmp_path, fake_image_reader, monkeypatch):
    data = _write_pair_files(tmp_path, with_segmentation=True)
    transform_dir = tmp_path / "init_transf"
    transform_dir.mkdir()
    (transform_dir / "200R_CT_to_US_pca_icp.tfm").write_text("fake", encoding="utf-8")

    interpolators = []

    def _spy_resample(self, moving, fixed, tx_moving_to_fixed, interpolator):
        interpolators.append(interpolator)
        return moving

    monkeypatch.setattr(dataset.Dataset, "_get_sitk", lambda self: _FakeSitk)
    monkeypatch.setattr(dataset.Dataset, "_resample_moving_to_fixed", _spy_resample)

    dataset.Dataset(
        input_shape=(8, 8, 8),
        name="trusted_with_seg",
        data=data,
        use_cache=False,
        init_transform_dir=str(transform_dir),
        init_transform_direction="ct_to_us",
    )
    assert _FakeSitk.sitkLinear in interpolators
    assert _FakeSitk.sitkNearestNeighbor in interpolators
