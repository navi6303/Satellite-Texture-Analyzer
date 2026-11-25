import numpy as np
import pytest
import rasterio
from affine import Affine

from modules.glcm_analysis import compute_glcm, get_glcm_demonstration_data
from modules.image_loader import load_raster, load_envi_upload
from modules.neighbors import require_valid_pairs
from modules.scene import prepare_scene, equalize_scene, read_patch, inspect_patch
from modules.subset import crop_original
from modules.texture_extraction import QUANT_GLOBAL_EQ_FIXED5


def test_crop_all_bands_before_pan_and_equalization(tmp_path):
    """Audit every intermediate array with a 6-band HDR scene and a translated ROI."""
    rng = np.random.default_rng(18)
    data = rng.integers(1200, 55000, size=(6, 52, 79), dtype=np.uint16)
    data[:, :10] = 65000  # outside subset, must not influence the PAN stretch or CDF
    data[4, 15, 33] = 0  # a genuine zero in one band remains valid
    valid = np.ones(data.shape[1:], dtype=bool)
    valid[18, 32] = False
    path = tmp_path / "hdr.tif"
    aff = Affine(10, 0, 325000, 0, -10, 1620000)
    with rasterio.open(path, "w", driver="GTiff", width=79, height=52, count=6,
                       dtype="uint16", transform=aff, crs="EPSG:32643") as out:
        out.write(data)
        out.write_mask(valid)
    src = load_raster(path, path.name)
    box = dict(x=16,y=11,w=51,h=38)
    cropped = crop_original(src,box,tmp_path/"crop.tif")
    exp = data[:,11:49,16:67]
    mask = valid[11:49,16:67]
    with rasterio.open(cropped.file_path) as out:
        assert out.count==6 and out.dtypes==("uint16",)*6
        assert out.transform==aff*Affine.translation(16,11)
        np.testing.assert_array_equal(out.read(),exp)
        np.testing.assert_array_equal(out.dataset_mask()>0,mask)
    scene = equalize_scene(prepare_scene(cropped,list(range(6)),tmp_path/"analysis"))
    pan = exp.mean(0,dtype=np.float32)
    with rasterio.open(scene.folder/"native.tif") as f:
        got=f.read(1)
    np.testing.assert_array_equal(got[mask],pan[mask])
    np.testing.assert_array_equal(np.isfinite(got),mask)
    vals=pan[mask]
    display=np.clip(np.rint((pan-vals.min())*(255./float(vals.max()-vals.min()))),0,255).astype(np.uint8)
    hist=np.bincount(display[mask],minlength=256)
    np.testing.assert_array_equal(scene.hist_before,hist)
    cum=hist.cumsum()
    first=cum[cum>0][0]
    lut=np.clip(np.rint((cum-first)*(255./(cum[-1]-first))),0,255).astype(np.uint8)
    eq=scene.preview("eq")
    np.testing.assert_array_equal(eq[mask],lut[display[mask]])
    patch=read_patch(scene,dict(x=4,y=3,w=28,h=16),"class","p",QUANT_GLOBAL_EQ_FIXED5,"percentile",[0,1,2])
    q=(eq[3:19,4:32].astype("uint16")*5//256)
    np.testing.assert_array_equal(patch.quantized[patch.valid_mask],q[patch.valid_mask])
    demo=get_glcm_demonstration_data(patch,distance=2,angles_deg=[0,90])
    assert demo["step2"]["pan_image"].dtype==np.uint8
    np.testing.assert_allclose(demo["step2"]["sample_matrix"],got[3:3+demo["step2"]["sample_matrix"].shape[0],4:4+demo["step2"]["sample_matrix"].shape[1]],equal_nan=True)


def test_mask_with_134451_valid_pixels_and_horizontal_pairs():
    mask=np.ones((370,370),bool)
    mask.ravel()[134451:]=False
    assert mask.sum()==134451
    counts=require_valid_pairs(mask,1,[0,45,90,135])
    for angle,pairs in counts.items():
        _, expected=compute_glcm(np.zeros(mask.shape,np.uint8),mask,1,np.radians(angle))
        assert pairs==expected
    assert counts[0]==np.count_nonzero(mask[:,1:] & mask[:,:-1])


def test_direction_distance_and_minimum_use_directed_masked_pairs():
    mask=np.zeros((12,3),bool)
    mask[:,1]=True
    assert require_valid_pairs(mask,2,[90])=={90:10}
    with pytest.raises(ValueError,match=r"0 valid neighboring pixel pairs at 0°, d=1.*10"):
        require_valid_pairs(mask,1,[0])
    with pytest.raises(ValueError,match=r"8 valid neighboring pixel pairs at 90°, d=2.*10"):
        require_valid_pairs(mask[:10],2,[90])
    with pytest.raises(ValueError,match=r"0 valid neighboring pixel pairs.*10"):
        require_valid_pairs(mask[:1],3,[90])


def test_per_band_nodata_and_true_zeroes(tmp_path):
    path=tmp_path/"masked.tif"
    data=np.ones((4,18,20),dtype="int16")
    data[3,4,5]=-999
    data[:,3,8]=0
    with rasterio.open(path,"w",driver="GTiff",width=20,height=18,count=4,
                       dtype="int16",nodata=-999) as dst:
        dst.write(data)
    src=load_raster(path,path.name)
    crop=crop_original(src,dict(x=3,y=2,w=11,h=13),tmp_path/"subset.tif")
    s=equalize_scene(prepare_scene(crop,list(range(4)),tmp_path/"scene"))
    assert s.stats["valid_pixels"]==142
    _,counts=inspect_patch(s,dict(x=0,y=0,w=11,h=13),1,[0])
    assert counts[0]==128
    with pytest.raises(ValueError,match=r"0 valid neighboring pixel pairs.*10"):
        inspect_patch(s,dict(x=2,y=2,w=1,h=1),1,[0])
    with pytest.raises(ValueError,match="exceeds"):
        inspect_patch(s,dict(x=10,y=2,w=10,h=4),1,[0])


def test_envi_header_binary_upload_and_crop(tmp_path):
    path=tmp_path/"spectra.bil"
    cube=np.arange(5*20*16,dtype="float32").reshape(5,20,16)
    with rasterio.open(path,"w",driver="ENVI",width=16,height=20,count=5,dtype="float32",interleave="bil") as dst:
        dst.write(cube)
    with path.with_suffix(".hdr").open("rb") as header, path.open("rb") as binary:
        src=load_envi_upload(header,binary,tmp_path/"cached")
    target=crop_original(src,dict(x=2,y=5,w=12,h=9),tmp_path/"crop.tif")
    with rasterio.open(target.file_path) as c:
        np.testing.assert_array_equal(c.read(),cube[:,5:14,2:14])


def test_standalone_radiance_hdr_preserves_decoded_samples(tmp_path):
    import cv2

    path = tmp_path / "radiance.hdr"
    bgr = np.linspace(0.2, 12.5, 24 * 30 * 3, dtype=np.float32).reshape(24, 30, 3)
    assert cv2.imwrite(str(path), bgr)
    # RGBE encoding rounds the original floats. Compare to the decoded file.
    rgb = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    source = load_raster(path, path.name)
    crop = crop_original(source, dict(x=4, y=3, w=17, h=12), tmp_path / "crop.tif")
    with rasterio.open(crop.file_path) as src:
        assert src.count == 3 and src.dtypes == ("float32",) * 3
        np.testing.assert_array_equal(src.read(), np.moveaxis(rgb[3:15, 4:21], 2, 0))
