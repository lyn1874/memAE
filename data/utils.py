import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch
import torch.utils.data as data
from PIL import Image


rng = np.random.RandomState(2020)


class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._time_step = time_step
        self._num_pred = num_pred
        
        video_string = self.setup()
        self.samples = self.get_all_samples(video_string)
        self.video_name = video_string
        
    def setup(self):
        if "venue" in self.dir:
            video_string = np.unique([v.strip().split('_frame')[0] for v in os.listdir(self.dir)])
            video_string = sorted(video_string, key=lambda s:int(s.strip().split('_')[-1]))
        elif "UCSD" in self.dir:
            video_string = np.unique([v.strip().split('_')[0] for v in os.listdir(self.dir)])
            video_string = sorted(video_string)
            print(video_string)
        all_video_frames = np.array([v for v in os.listdir(self.dir) if '.jpg' in v])
        for video in video_string:
            self.videos[video] = {}
            self.videos[video]['path'] = self.dir + video
            _subframe = [v for v in all_video_frames if video + '_' in v]
            if "venue" in self.dir:
                _subframe = sorted(_subframe, key=lambda s:int(s.strip().split('_')[-1].strip().split('.jpg')[0]))
            elif "UCSD" in self.dir:
                _subframe = sorted(_subframe)
            self.videos[video]['frame'] = np.array([self.dir + v  for v in _subframe])
            self.videos[video]['length'] = len(_subframe)
        return video_string
    
    def get_all_samples(self, video_string):
        frames = []
        for video in video_string:
            for i in range(len(self.videos[video]['frame']) - self._time_step):
                frames.append(self.videos[video]['frame'][i])
        return frames
    
    def load_image(self, filename):
        image = Image.open(filename)
        return image.convert('RGB')

    def __getitem__(self, index):
        frame_name = int(self.samples[index].split('/')[-1].split('_')[-1].split('.jpg')[0])
        if "venue" in self.dir:
            video_name = self.samples[index].split('/')[-1].split('_frame')[0] 
        elif "UCSD" in self.dir:
            video_name = self.samples[index].split('/')[-1].split('_')[0]
            frame_name -= 1
        batch = []
        
        for i in range(self._time_step+self._num_pred):
            image = self.load_image(self.videos[video_name]['frame'][frame_name+i])
#             print(i, self.videos[video_name]['frame'][frame_name + i])
            if self.transform is not None:
                batch.append(self.transform(image))
        return torch.cat(batch, 0)

    def __len__(self):
        return len(self.samples)

    

    