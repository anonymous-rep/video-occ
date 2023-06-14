from __future__ import annotations
import gc
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import torch.utils.data
from pytorchvideo.data.clip_sampling import ClipSampler
from .labeled_video_paths import LabeledVideoPaths
from .utils import MultiProcessSampler
logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Optional
import torch
from iopath.common.file_io import g_pathmgr
import numpy as np
import cv2 as cv
import random
import os
import xml.etree.ElementTree
import PIL.Image
import math
import json

import pytorchvideo
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)
import cv2 as cv


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    
    """
    factor = min(10,factor)
    if factor < 1e-4:
        factor = 1e-4
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv.INTER_LINEAR if factor > 1.0 else cv.INTER_AREA
    return cv.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def paste_over(im_src, im_dst, center,side):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.
    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    im_src` becomes visible).
    Args:
    im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
    im_dst: The target image.
    alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
    at each pixel. Large values mean more visibility for `im_src`.
    center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """
    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32)/255
    
    occ_height,occ_width,_ = region_src.shape
    img_height,img_width,_ = im_dst.shape 
    scale = np.sqrt((side*220*220)/(np.sum(region_src[:,:,3]>0))+1)
    
    im_src = resize_by_factor(im_src,scale)
    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])
    
    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src
    

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32)/255
    #print(np.sum(alpha>0)/(224*224))
    im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)


class occlude:
    def __init__(self,img_shape,occluder_index,occluder_size,occluder_motion):
        self.occluder_index = occluder_index
        self.occluders = self.load_occluders()
        self.occluder = [self.occluders[occluder_index]]
        self.occ_idx =  occluder_index
        self.occluder_size = occluder_size#in % of area to be covered
        self.occluder_motion = occluder_motion# String
        self.width_height = np.asarray([img_shape[1], img_shape[0]])
        self.center = self.random_placement()
        self.theta = random.randint(-89,89)
        self.scale = {10:1,20:1,30:1,40:1,50:1.1,60:1.2,80:1.27}
        self.occ_scale = {0:1,1:1.2,2:1.2,3:1.1,4:1.05,5:1.1,6:1.1}
        self.occluder_size = self.occluder_size*self.scale[self.occluder_size]*self.occ_scale.get(occluder_index,1.1)
        
        self.motion_dict = {"random_placement":self.random_placement,"random_motion":self.random_motion,"linear_motion":self.linear_motion,"circular_motion":self.circular_motion,"static":self.static,"sine_motion":self.sine_motion}
        self.motion_choice = occluder_motion
        self.motion = self.motion_dict[occluder_motion]
        self.radius = np.sqrt(np.sum((self.center-self.width_height/2)**2))
    
        #self.center = self.width_height/2
        
               
    def load_occluders(self):
        occluders = [] 
        structuring_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
        annotations = [os.path.join("./Data_256/occ_data/attributes",path) for path in os.listdir("./Data_256/occ_data/attributes/")]
        annotations.sort()
        annotations = [path for path in annotations if ".xml" in path ]
        for annotation_path in annotations:
            xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
            is_segmented = (xml_root.find('segmented').text != '0')
            if not is_segmented:
                continue

            boxes = []
            for i_obj, obj in enumerate(xml_root.findall('object')):
            #is_person = (obj.find('name').text == 'person')
            #is_difficult = (obj.find('difficult').text != '0')
            #is_truncated = (obj.find('truncated').text != '0')
            #if not is_person and not is_difficult and not is_truncated:
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append((i_obj, box))
        
        
        
            if not boxes:
                continue
        
        
            im_filename = xml_root.find('filename').text
            seg_filename = im_filename.replace('jpg', 'png')
            im_path = os.path.join("./Data_256/occ_data/images",im_filename)
            seg_path = os.path.join("./Data_256/occ_data/segmentation",seg_filename)
            im = np.asarray(PIL.Image.open(im_path))
            labels = np.asarray(PIL.Image.open(seg_path))  
       
            for i_obj, (xmin, ymin, xmax, ymax) in boxes:
                object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8)*255
                object_image = im[ymin:ymax, xmin:xmax]
                if cv.countNonZero(object_mask) < 2:
                #Ignore small objects
                    continue

            # Reduce the opacity of the mask along the border for smoother blending
                eroded = cv.erode(object_mask, structuring_element)
                object_mask[eroded < object_mask] = 192
                object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)
            
            # Downscale for efficiency
                object_with_mask = resize_by_factor(object_with_mask, 0.5)
                occluders.append(object_with_mask)
        return occluders

    def occlude_with_objects(self,im,epoch):
        """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""
        occluders = self.occluder
        im = im.permute(1,2,0)
        im = np.array(im)
        #print(im.shape)
        result = im.copy()
        h = im.shape[0]
        w,h = self.width_height
        width_height = np.asarray([im.shape[1], im.shape[0]])
        occluder = random.choice(occluders)
        center = self.motion(epoch)
        
        paste_over(im_src=occluder, im_dst=result, center=center,side = self.occluder_size/100)

        return result

    def random_placement(self,epoch=1):
        return np.random.uniform([0,0], self.width_height)
    def linear_motion(self,epoch=1):
        w,h = self.width_height
        x_step = 5
        y_step = 5*np.tan(math.pi*self.theta/180)
        st_h = self.center[1]
        st_w = self.center[0]
        new_c_h = (st_h+epoch*x_step)%h
        new_c_w = (st_w+epoch*y_step)%w
        return np.array([new_c_w,new_c_h])
    def random_motion(self,epoch):
        w,h = self.width_height
        delta_x = random.uniform(-.15,.15)*h
        delta_y = random.uniform(-.15,.15)*w
        st_h = (self.center[1] + delta_x)%h
        st_w = (self.center[0] + delta_y)%w
        self.center = np.array([st_w,st_h])
        return np.array([st_w,st_h])
    def train_randomize(self):
        self.center = self.random_placement(0)
        self.theta = random.randint(-89,89)
        motion_choices = list(self.motion_dict.keys())
        occ_choice =  random.choice(range(0,50)) 
        size_choices = [20,40]
        self.motion_choice = random.choice(motion_choices[1:3])
        size_choice = random.choice(size_choices)
        self.occluder_size = size_choice
        self.motion = self.motion_dict[self.motion_choice]
        self.occluder = [self.occluders[occ_choice]]
        self.occ_idx =  occ_choice
        
    def circular_motion(self,epoch):
        new_c_h = (self.radius/(1+epoch*.1))*np.cos(math.pi*(self.theta+epoch*20)/180)
        new_c_w = (self.radius/(1+.1*epoch))*np.sin(math.pi*(self.theta+epoch*20)/180)
        center = np.array([new_c_w,new_c_h])+self.width_height/2
        return  center
    def static(self,epoch):
        return self.center
    def sine_motion(self,epoch):
        w,h = self.width_height
        x_step = 5
        st_h = self.center[1]
        st_w = self.center[0]
        new_c_h = (st_h+epoch*x_step)%h
        new_c_2 = ((w/1.225)*np.sin(new_c_h))%w
        self.center = np.array([st_w,st_h])
        return np.array([st_w,st_h])
    
    def initialize(self):
        self.center = self.random_placement()
        self.theta = random.randint(-89,89)
        self.radius = np.sqrt(np.sum((self.center-self.width_height/2)**2))
        
    def test_randomizer(self):
        self.center = self.width_height/2#self.random_placement(0)
        self.theta = random.randint(-89,89)
        motion_choices = ["random_placement","circular_motion","static","sine_motion"]
        occ_choice =  random.choice(range(50)) 
        size_choices = [50,60]
        self.motion_choice = random.choice(motion_choices)
        size_choice = random.choice(size_choices)
        self.occluder_size = size_choice*self.scale[size_choice]*self.occ_scale.get(occ_choice,1.15)
        self.motion = self.motion_dict[self.motion_choice]
        self.occluder = [self.occluders[occ_choice]]
        self.occ_idx =  occ_choice
    def set_val(self,occ_dict):
        self.center = self.width_height/2#self.random_placement(0)
        self.theta = random.randint(-89,89)
        self.occluder_size = occ_dict["occluder_size"]#size_choice*self.scale[size_choice]*self.occ_scale.get(occ_choice,1.15)
        self.motion_choice = occ_dict["motion_choice"]
        self.motion = self.motion_dict[occ_dict["motion_choice"]]
        self.occluder = [self.occluders[occ_dict["occ_choice"]]]
        self.occ_idx =  occ_dict["occ_choice"]                
    
                         
    def get_val(self):
        return  self.occluder_size, self.motion_choice, self.occ_idx                     

    

        
def save_vid(vid,name):
    _,l,h,w = vid.shape
    out = cv.VideoWriter(name,cv.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (h,w))
    #print(vid.shape)
    for i in range(l):
        frame = vid[:,i,:,:]
        frame = frame.permute(1,2,0)
        frame = frame.numpy()
        frame = frame.astype(np.uint8)
        out.write(frame)
    out.release()

class VideoPathHandler(object):
    """
    Utility class that handles all deciphering and caching of video paths for
    encoded and frame videos.
    """

    def __init__(self) -> None:
        # Pathmanager isn't guaranteed to be in correct order,
        # sorting is expensive, so we cache paths in case of frame video and reuse.
        self.path_order_cache = {}
        

    def video_from_path(
        self, filepath, decode_video=True, decode_audio=False, decoder="pyav", fps=30
    ):
        try:
            is_file = g_pathmgr.isfile(filepath)
            is_dir = g_pathmgr.isdir(filepath)
        except NotImplementedError:

            # Not all PathManager handlers support is{file,dir} functions, when this is the
            # case, we default to assuming the path is a file.
            is_file = True
            is_dir = False

        if is_file:
            from pytorchvideo.data.encoded_video import EncodedVideo

            return EncodedVideo.from_path(
                filepath,
                #decode_video=decode_video,
                #decode_audio=decode_audio,
                decoder=decoder,
            )
        elif is_dir:
            from pytorchvideo.data.frame_video import FrameVideo

            assert not decode_audio, "decode_audio must be False when using FrameVideo"
            return FrameVideo.from_directory(
                filepath, fps, path_order_cache=self.path_order_cache
            )
        else:
            raise FileNotFoundError(f"{filepath} not found.")


class Video(ABC):
    """
    Video provides an interface to access clips from a video container.
    """

    @abstractmethod
    def __init__(
        self,
        file: BinaryIO,
        video_name: Optional[str] = None,
        decode_audio: bool = True,
    ) -> None:
        """
        Args:
            file (BinaryIO): a file-like object (e.g. io.BytesIO or io.StringIO) that
                contains the encoded video.
        """
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        """
        Returns:
            duration of the video in seconds
        """
        pass

    @abstractmethod
    def get_clip(
        self, start_sec: float, end_sec: float
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieves frames from the internal video at the specified start and end times
        in seconds (the video always starts at 0 seconds).
        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            video_data_dictonary: A dictionary mapping strings to tensor of the clip's
                underlying data.
        """
        pass

    def close(self):
        pass

class LabeledVideoDataset(torch.utils.data.IterableDataset):
    """
    LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as either an encoded video
    (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decode_video: bool = True,
        decoder: str = "pyav",
        occ: bool = False,
        train:bool = False
    ) -> None:
        """
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                    video file paths and associated labels. If video paths are a folder
                    it's interpreted as a frame video, otherwise it must be an encoded
                    video.
            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.
            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.
            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().
            decode_audio (bool): If True, decode audio from video.
            decode_video (bool): If True, decode video frames from a video container.
            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """
        self._decode_audio = decode_audio
        self._decode_video = decode_video
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        #print(self._labeled_videos)
        
        self._decoder = decoder
        self.occlusion = occlude([224,224],2,40,"random_motion")
        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._last_clip_end_time = None
        self.video_path_handler = VideoPathHandler()
        self.occ = occ
        self.train = train
        #self.occ_dict = self.get_vid_state() 
        self.tr = Compose(
                  [
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    
                 ]
                )
        self.save_path = "/home/sh009885/code/ucf101-supervised/ucf101-supervised-main/dataset/Kinetics-400-O"
        #with open("/home/sh009885/code/ucf101-supervised/ucf101-supervised-main/dataset/Kinetics-400-O/detail.json", "w") as outfile:
         #   json.dump(self.occ_dict,outfile)

    @property
    def video_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_sampler

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.video_sampler)
    def get_vid_state(self):
        occ_dict = {}
        
        for name,cl in self._labeled_videos:
           #print(name.split('/'))
            v_name = name.split('/')[-1]
            #print(v_name)
            occ_dict[v_name] = {}
            self.occlusion.test_randomizer()
            sz,motion,occ = self.occlusion.get_val()
            occ_dict[v_name]["size"] = sz
            occ_dict[v_name]["motion"] = motion
            occ_dict[v_name]["occ_index"] = occ
            occ_dict[v_name]["path"] = name 
        return occ_dict
    def vid_name(self,name):
        t=name.split("/")
        return t[-2],t[-1]
                
    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.
        Returns:
            A dictionary with the following format.
            .. code-block:: text
                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video, info_dict, video_index = self._loaded_video_label
            else:
                video_index = next(self._video_sampler_iter)
                try:
                    video_path, info_dict = self._labeled_videos[video_index]
                    video = self.video_path_handler.video_from_path(
                        video_path,
                        #decode_audio=self._decode_audio,
                        #decode_video=self._decode_video,
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video, info_dict, video_index)
                except Exception as e:
                    logger.debug(
                        "Failed to load video with error: {}; trial {}".format(
                            e,
                            i_try,
                        )
                    )
                    logger.exception("Video load exception")
                    continue

            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self._clip_sampler(self._last_clip_end_time, video.duration, info_dict)

            if isinstance(clip_start, list):  # multi-clip in each sample

                # Only load the clips once and reuse previously stored clips if there are multiple
                # views for augmentations to perform on the same clips.
                if aug_index[0] == 0:
                    self._loaded_clip = {}
                    loaded_clip_list = []
                    for i in range(len(clip_start)):
                        clip_dict = video.get_clip(clip_start[i], clip_end[i])
                        if clip_dict is None or clip_dict["video"] is None:
                            self._loaded_clip = None
                            break
                        loaded_clip_list.append(clip_dict)

                    if self._loaded_clip is not None:
                        for key in loaded_clip_list[0].keys():
                            self._loaded_clip[key] = [x[key] for x in loaded_clip_list]

            else:  # single clip case

                # Only load the clip once and reuse previously stored clip if there are multiple
                # views for augmentations to perform on the same clip.
                if aug_index == 0:
                    self._loaded_clip = video.get_clip(clip_start, clip_end)

            self._last_clip_end_time = clip_end

            video_is_null = (
                self._loaded_clip is None or self._loaded_clip["video"] is None
            )
            if (
                is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
            ) or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                self._loaded_video_label[0].close()
                self._loaded_video_label = None
                self._last_clip_end_time = None
                self._clip_sampler.reset()

                # Force garbage collection to release video container immediately
                # otherwise memory can spike.
                gc.collect()

                if video_is_null:
                    logger.debug(
                        "Failed to load clip {}; trial {}".format(video.name, i_try)
                    )
                    continue

            frames = self._loaded_clip["video"]
            #audio_samples = self._loaded_clip["audio"]
            sample_dict = {
                "video": frames,
                "video_name": video.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                **info_dict,
                #**({"audio": audio_samples} if audio_samples is not None else {}),
            }
            #print(sample_dict["video"].shape)
            vid_name = sample_dict["video_name"]
            #cl,name = self.vid_name(self.occ_dict[vid_name]["path"])
            #cl_path =os.path.join(self.save_path,cl)
            #print(cl_path)
            #try:
            #    if not os.path.exists(cl_path):
            #        os.mkdir(cl_path)
            #except:
            #    print("nothing")
            #save_path = os.path.join(cl_path,name)
            if self.train:
                self.occlusion.train_randomize()
            else:
                self.occlusion.test_randomizer()#
            #occ_dict = {}
            #occ_dict["motion_choice"] = self.occ_dict[vid_name]["motion"]
            #occ_dict["occ_choice"] = self.occ_dict[vid_name]["occ_index"]
            #occ_dict["occluder_size"] = self.occ_dict[vid_name]["size"]
            #self.occlusion.set_val(occ_dict)
        
            if self._transform is not None:
                
                sample_dict = self._transform(sample_dict)
                if self.occ:
                    #print(sample_dict["video"].shape)
                    clip = sample_dict["video"].permute(1,0,2,3)
                    clip = [torch.tensor(self.occlusion.occlude_with_objects(img,epoch)) for epoch,img in enumerate(clip)]
                #User can force dataset to continue by returning None in transform.22
                    clip = torch.stack(clip,0)
                    sample_dict["video"] = clip.permute(3,0,1,2)
                sample_dict["video"] = self.tr(sample_dict["video"])    
                #print(save_path)
             #   save_vid(sample_dict["video"],save_path)    
                #clip = sample_dict["video"].permute(1,0,2,3)
                #clip = [torch.tensor(self.occlusion.occlude_with_objects(img,epoch)) for epoch,img in enumerate(clip)]
                #User can force dataset to continue by returning None in transform.
                #clip = torch.stack(clip,0)
                #sample_dict["video"] = clip
                if sample_dict is None:
                    continue

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self


def labeled_video_dataset(
    cl: int=None,
    data_path: str=None,
    clip_sampler: ClipSampler=None,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = False,
    decoder: str = "pyav",
    occ: bool = False,
    train: bool = False
) -> LabeledVideoDataset:
    """
    A helper function to create ``LabeledVideoDataset`` object for Ucf101 and Kinetics datasets.
    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:
            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).
        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.
        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.
        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.
        video_path_prefix (str): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.
        decode_audio (bool): If True, also decode audio from video.
        decoder (str): Defines what type of decoder used to decode a video.
    """
    labeled_video_paths = LabeledVideoPaths.from_path(data = cl,data_path = data_path)
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = LabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
        occ = occ,
        train = train
    )
    return dataset
