import cv2
import math
import numpy as np

import torch

import pydicom
from pydicom.filebase import DicomBytesIO
from pydicom.pixel_data_handlers import apply_windowing

import dicomsdl

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.types import DALIDataType
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from libs.image_processing import to_3_channels, resize_and_pad, check_mkdir

# j2k
class J2kIterator(object):
    def __init__(self, df, batch_size, img_dir):
        self.df = df
        self.batch_size = batch_size
        self.img_dir = img_dir

    def dicom_to_j2k(self, img_dir, patient_id, image_id):
        dcmfile = pydicom.dcmread(f'{img_dir}/{patient_id}/{image_id}.dcm')
        
        if dcmfile.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':
            with open(f'{img_dir}/{patient_id}/{image_id}.dcm', 'rb') as fp:
                raw = DicomBytesIO(fp.read())
                ds = pydicom.dcmread(raw)
            offset = ds.PixelData.find(b"\x00\x00\x00\x0C")

            return np.frombuffer(ds.PixelData[offset:], dtype=np.uint8), dcmfile.PhotometricInterpretation == 'MONOCHROME1'

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.df):
        #if self.i > 32:
            raise StopIteration

        compressed_imgs, is_monochrome_imgs, patient_ids, img_ids = [], [], [], []

        batch_start, batch_finish = self.i, min(self.i + self.batch_size, len(self.df))
        df_batch = self.df.iloc[range(batch_start, batch_finish)]

        for patient_id, img_id in df_batch[["patient_id", "image_id"]].values:
            compressed_img, is_monochrome_img = self.dicom_to_j2k(self.img_dir, patient_id, img_id)

            compressed_imgs.append(compressed_img)
            is_monochrome_imgs.append(np.array([is_monochrome_img], dtype = np.bool_))

            patient_ids.append(np.array([patient_id], dtype = np.int64))
            img_ids.append(np.array([img_id], dtype = np.int64))

        self.i += self.batch_size

        return compressed_imgs, is_monochrome_imgs, patient_ids, img_ids

@pipeline_def
def j2k_decode_pipeline(J2Kiterator, width, height):
    imgs, is_monochromes, patient_ids, img_ids = fn.external_source(
        source=J2Kiterator, num_outputs=4, device="cpu", dtype = [types.UINT8, types.BOOL, types.INT64, types.INT64],
    )

    imgs = fn.experimental.decoders.image(
        imgs, device='mixed', output_type=types.ANY_DATA, dtype=DALIDataType.UINT16
    )

    imgs = fn.resize(imgs, size=[width, height], device="gpu")

    imgs = fn.cast(imgs, dtype = types.INT16)
    
    return imgs, is_monochromes, patient_ids, img_ids

def j2k_postprocessing(j2k_out: tuple, index: int):
    img = j2k_out[0].as_cpu().as_array()[index].squeeze()
    is_monochrome =  j2k_out[1].as_array().squeeze().tolist()[index]
    patient_id = j2k_out[2].as_array().squeeze().tolist()[index]
    img_id = j2k_out[3].as_array().squeeze().tolist()[index]

    dcmfile = pydicom.dcmread(f"/home/data4/share/rsna-breast-cancer-detection/train_images/{patient_id}/{img_id}.dcm")
    img_window = apply_windowing(img, dcmfile)
    
    img = to_3_channels(img, is_monochrome)
    img_window = to_3_channels(img_window, is_monochrome)

    return img, img_window

# jll
class JllIterator(object):
    def __init__(self, df, batch_size, img_dir):
        self.df = df
        self.batch_size = batch_size
        self.img_dir = img_dir
       
    @staticmethod
    def decompress_jll(dm):
        info = dm.getPixelDataInfo()
        img = np.empty((info['Rows'], info['Cols']), dtype = info['dtype'])
        dm.copyFrameData(0, img)
        return img
    
    def process_img(self, img_dir, patient_id, image_id):
        dm = dicomsdl.open(f'{img_dir}/{patient_id}/{image_id}.dcm')
        return self.decompress_jll(dm).astype(np.uint16), dm.PhotometricInterpretation == 'MONOCHROME1'
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.df):
            raise StopIteration
        
        decompressed_imgs, is_monochrome_imgs, patient_ids, img_ids = [], [], [], []

        batch_start, batch_finish = self.i, min(self.i + self.batch_size, len(self.df))
        df_batch = self.df.iloc[range(batch_start, batch_finish)]
        
        for patient_id, img_id in df_batch[['patient_id', 'image_id']].values:
            decompressed_img, is_monochrome_img = self.process_img(self.img_dir, patient_id, img_id)
            
            decompressed_imgs.append(np.expand_dims(decompressed_img, axis = 2))
            is_monochrome_imgs.append(np.array([is_monochrome_img], dtype = np.bool_))
            patient_ids.append(np.array([patient_id], dtype = np.int64))
            img_ids.append(np.array([img_id], dtype = np.int64))
            
        self.i += self.batch_size
                    
        return decompressed_imgs, is_monochrome_imgs, patient_ids, img_ids

@pipeline_def
def jll_process_pipeline(JLLiterator, width, height):
    imgs, is_monochromes, patient_ids, img_ids = fn.external_source(
        source=JLLiterator, num_outputs=4, device="gpu", dtype = [types.UINT16, types.BOOL, types.INT64, types.INT64],
    )
    
    imgs = fn.reinterpret(imgs, layout = "HWC")
    
    imgs = fn.resize(imgs, size=[width, height], device="gpu")

    imgs = fn.cast(imgs, dtype = types.INT16)
    
    return imgs, is_monochromes, patient_ids, img_ids

def jll_postprocessing(jll_out: tuple, index: int):
    img = jll_out[0].as_cpu().as_array()[index].squeeze()
    is_monochrome = jll_out[1].as_cpu().as_array().squeeze().tolist()[index]
    patient_id = jll_out[2].as_cpu().as_array().squeeze().tolist()[index]
    img_id = jll_out[3].as_cpu().as_array().squeeze().tolist()[index]

    dcmfile = pydicom.dcmread(f"/home/data4/share/rsna-breast-cancer-detection/train_images/{patient_id}/{img_id}.dcm")
    img_window = apply_windowing(img, dcmfile)
    
    img = to_3_channels(img, is_monochrome)
    img_window = to_3_channels(img_window, is_monochrome)

    return img, img_window

# DALI iterator
def preprocess(batch, yolo_model, save_img=False):
    imgs = batch[0]["imgs"].cpu().numpy()
    is_monochromes = batch[0]["is_monochromes"].cpu().numpy()
    patient_ids = batch[0]["patient_ids"].cpu().numpy()
    img_ids = batch[0]["img_ids"].cpu().numpy()

    result = torch.zeros([len(imgs), 3, 2048, 1024], dtype=torch.float32)
    for i in range(len(imgs)):
        img, is_monochrome = imgs[i].squeeze(), is_monochromes[i][0]
        patient_id, img_id = patient_ids[i][0], img_ids[i][0]

        #dcmfile = pydicom.dcmread(f"/home/data4/share/rsna-breast-cancer-detection/train_images/{patient_id}/{img_id}.dcm")
        dcmfile = pydicom.dcmread(f"/home/data4/share/rsna-breast-cancer-detection/test_images/{patient_id}/{img_id}.dcm")
        img_window = apply_windowing(img, dcmfile)

        j2k_img = to_3_channels(img, is_monochrome)
        j2k_img_window = to_3_channels(img_window, is_monochrome)

        try:
            with torch.no_grad():
                roi = roi_extraction_yolov5(roi_model, j2k_img)
            j2k_img_crop = crop_img(j2k_img_window, roi)
        except:
            try:
                j2k_img_gray = cv2.cvtColor(j2k_img, cv2.COLOR_BGR2GRAY)
                roi = roi_extraction_cv2(j2k_img_gray)
                j2k_img_crop = crop_img(j2k_img_window, roi)
            except:
                j2k_img_crop = j2k_img
                
        j2k_img_pad = resize_and_pad(j2k_img_crop)
        if save_img:
            save_path = f"/home/FanHuang247817/train_images_png2_test/{patient_id}/{img_id}.png"
            check_mkdir(save_path)
            cv2.imwrite(save_path, j2k_img_pad)
        j2k_img_pad = torch.permute(torch.from_numpy(j2k_img_pad), (2,0,1)) # (2048, 1024, 3) --> (3, 2048, 1024)
        result[i, :, :, :]  = j2k_img_pad / 255.0
        
    return result

class CustomDALIGenericIterator(DALIGenericIterator):
    def __init__(self, yolo_model, length, save_img, pipelines):
        self.yolo_model = yolo_model
        self.length = length
        self.save_img = save_img
        super().__init__(pipelines, ['imgs', 'is_monochromes', 'patient_ids', 'img_ids'])

    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i > self.length:
            raise StopIteration
        batch = super().__next__()
        batch = preprocess(batch, self.yolo_model, self.save_img)
        self.i += self.batch_size
        return batch
    
    def __len__(self):
        return math.ceil(self.length / 32)