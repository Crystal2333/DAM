import os
import glob
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from util import Gaussian_Heatmap


transform_head = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_eye = transforms.Compose([
    transforms.Resize((36, 60)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_depth = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class Gazefollowing_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, annotation_dir, depth_dir, test = False):
        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.depth_dir = depth_dir
        self.test = test

        if self.test:
            df = pd.read_csv(self.annotation_dir, sep=',', header=None, index_col=False, 
            names=['path', 'idx', 
            'head_x_min', 'head_y_min', 'head_x_max', 'head_y_max', 
            'right_eye_x_min', 'right_eye_y_min', 'right_eye_x_max', 'right_eye_y_max', 
            'left_eye_x_min', 'left_eye_y_min', 'left_eye_x_max', 'left_eye_y_max', 
            'gaze_1', 'gaze_2', 'gaze_3', 
            'gaze_x', 'gaze_y', 
            'meta']
            )
            # have eye
            #df = df[df['right_eye_x_min'] != -1]
            #df.reset_index(inplace=True)
            
            self.df = df[['path', 'head_x_min', 'head_y_min', 'head_x_max', 'head_y_max', 
            'right_eye_x_min', 'right_eye_y_min', 'right_eye_x_max', 'right_eye_y_max', 
            'left_eye_x_min', 'left_eye_y_min', 'left_eye_x_max', 'left_eye_y_max', 
            'gaze_x', 'gaze_y']].groupby('path')
            self.path = list(self.df.groups.keys())
            self.length = len(self.path)
            
        else:
            df = pd.read_csv(self.annotation_dir, sep=',', header=None, index_col=False, 
            names=['path', 'idx', 
            'head_x_min', 'head_y_min', 'head_x_max', 'head_y_max', 
            'right_eye_x_min', 'right_eye_y_min', 'right_eye_x_max', 'right_eye_y_max', 
            'left_eye_x_min', 'left_eye_y_min', 'left_eye_x_max', 'left_eye_y_max', 
            'gaze_1', 'gaze_2', 'gaze_3', 
            'gaze_x', 'gaze_y', 
            'inout']
            )
            # have eye
            #df = df[df['right_eye_x_min'] != -1]
            #df.reset_index(inplace=True)
            #print(df)
            self.df = df[['path', 'head_x_min', 'head_y_min', 'head_x_max', 'head_y_max', 
            'right_eye_x_min', 'right_eye_y_min', 'right_eye_x_max', 'right_eye_y_max', 
            'left_eye_x_min', 'left_eye_y_min', 'left_eye_x_max', 'left_eye_y_max', 
            'gaze_x', 'gaze_y',
            'inout']]
            self.length = len(self.df)
        #print(self.length)
    
    def __getitem__(self, index):
        if self.test:
            data = self.df.get_group(self.path[index])
            gaze_collect = []
            for i in range(len(data)):
                path, head_x_min, head_y_min, head_x_max, head_y_max, right_eye_x_min, right_eye_y_min, right_eye_x_max, right_eye_y_max, left_eye_x_min, left_eye_y_min, left_eye_x_max, left_eye_y_max, gaze_x, gaze_y = data.iloc[i]
                gaze_collect.append([gaze_x, gaze_y])
            gaze_collect = torch.FloatTensor(gaze_collect)
            gaze_average = torch.mean(gaze_collect, dim = 0)
            #print(gaze_collect, gaze_average)

        else:
            path, head_x_min, head_y_min, head_x_max, head_y_max, right_eye_x_min, right_eye_y_min, right_eye_x_max, right_eye_y_max, left_eye_x_min, left_eye_y_min, left_eye_x_max, left_eye_y_max, gaze_x, gaze_y, inout = self.df.iloc[index]

        #print(right_eye_x_min)
        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        width, height = img.size
        imsize = torch.IntTensor([width, height])

        depth_path = path[:-4] + '.png'
        #print(depth_path)
        depthmap = Image.open(os.path.join(self.depth_dir, depth_path))
        depthmap = transform_depth(depthmap).squeeze(0)
        
        if right_eye_x_min == -1:
            eye_weight = torch.Tensor([0.0])

        else:
            eye_weight = torch.Tensor([1.0])
            right_eye_width = abs(right_eye_x_max - right_eye_x_min)
            right_eye_height = abs(right_eye_y_max - right_eye_y_min)
            left_eye_width = abs(left_eye_x_max - left_eye_x_min)
            left_eye_height = abs(left_eye_y_max - left_eye_y_min)
            k = 0.5
            right_eye_x_min = max(right_eye_x_min - k * right_eye_width, 0)
            right_eye_y_min = max(right_eye_y_min - k * right_eye_height, 0)
            right_eye_x_max = min(right_eye_x_max + k * right_eye_width, width)
            right_eye_y_max = min(right_eye_y_max + k * right_eye_height, height)
            left_eye_x_min = max(left_eye_x_min - k * left_eye_width, 0)
            left_eye_y_min = max(left_eye_y_min - k * left_eye_height, 0)
            left_eye_x_max = min(left_eye_x_max + k * left_eye_width, width)
            left_eye_y_max = min(left_eye_y_max + k * left_eye_height, height)


        if self.test == False and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            x_max = width - head_x_min
            x_min = width - head_x_max
            head_x_max = x_max
            head_x_min = x_min

            gaze_x = 1 - gaze_x

            if right_eye_x_min != -1:
                x_max = width - left_eye_x_min
                x_min = width - left_eye_x_max
                left_eye_x_max = x_max
                left_eye_x_min = x_min

                x_max = width - right_eye_x_min
                x_min = width - right_eye_x_max
                right_eye_x_max = x_max
                right_eye_x_min = x_min

        head = img.crop((head_x_min, head_y_min, head_x_max, head_y_max))
        head = transform_head(head)
        image = transform_img(img)


        if right_eye_x_min == -1:
            right_eye = torch.zeros(3, 36, 60)
            left_eye = torch.zeros(3, 36, 60)

        else:
            right_eye = img.crop((right_eye_x_min, right_eye_y_min, right_eye_x_max, right_eye_y_max))
            left_eye = img.crop((left_eye_x_min, left_eye_y_min, left_eye_x_max, left_eye_y_max))
            right_eye = transform_eye(right_eye)
            left_eye = transform_eye(left_eye)
        
        head_position = [(head_x_min+head_x_max)/(2*width), (head_y_min+head_y_max)/(2*height)]
        gaze_field = self.generate_data_field(head_position)
        gaze_field = torch.FloatTensor(gaze_field)

        head_channel = torch.zeros(224, 224)
        #head_channel[int(head_y_min*224/height):int(head_y_max*224/height), int(head_x_min*224/width):int(head_x_max*224/width)] = 1
        head_channel[int(head_position[1]*224), int(head_position[0]*224)] = 1
        head_position = torch.FloatTensor(head_position)
        if self.test:
            seperate_gaze_direction = gaze_collect - head_position
            average_gaze_direction = gaze_average - head_position


            return image, depthmap, head, head_channel, left_eye, right_eye, eye_weight, gaze_field, gaze_collect, gaze_average, seperate_gaze_direction, average_gaze_direction, imsize, head_position

        else:
            gaze_point = torch.FloatTensor([gaze_x, gaze_y])
            gaze_direction = gaze_point - head_position

            heatmap = torch.zeros(64, 64)
            heatmap = Gaussian_Heatmap(heatmap.numpy(), [gaze_x*64, gaze_y*64], 3)

            inout = torch.FloatTensor([inout])

            return image, depthmap, head, head_channel, left_eye, right_eye, eye_weight, gaze_field, heatmap, gaze_direction, inout, imsize, gaze_point

    def generate_data_field(self, head_position):
        """eye_point is (x, y) and between 0 and 1"""
        height, width = 224, 224
        x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
        y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
        grid = np.stack((x_grid, y_grid)).astype(np.float32)

        x, y = head_position
        x, y = x * width, y * height

        grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
        norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
        # avoid zero norm
        norm = np.maximum(norm, 0.1)
        grid /= norm
        return grid

    def __len__(self):
        return self.length


class Videoatttarget_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, annotation_dir, depth_dir, test = False):
        self.data_dir = data_dir
        self.annotation_dir = annotation_dir
        self.depth_dir = depth_dir
        self.test = test

        all_sequence_paths = glob.glob(os.path.join(self.annotation_dir, '*', '*', '*.txt'))
        #print(all_sequence_paths)
        #print(len(all_sequence_paths))
        df = []
        for seq in all_sequence_paths:
            #print(seq)
            sep_df = pd.read_csv(seq, sep=',', header=None, index_col=False, 
            names=['path', 'head_x_min', 'head_y_min', 'head_x_max', 'head_y_max', 
            'right_eye_x_min', 'right_eye_y_min', 'right_eye_x_max', 'right_eye_y_max', 
            'left_eye_x_min', 'left_eye_y_min', 'left_eye_x_max', 'left_eye_y_max', 
            'gaze_1', 'gaze_2', 'gaze_3', 
            'gaze_x', 'gaze_y']
            )

            show_name = seq.split('/')[-3]
            #print(show_name)
            clip_name = seq.split('/')[-2]
            #print(clip_name)
            sep_df['path'] = sep_df['path'].apply(lambda x : show_name + '/' + clip_name + '/' + x)
            df.append(sep_df)
        
        df = pd.concat(df)
        df.reset_index(inplace=True)
        self.df = df[['path', 'head_x_min', 'head_y_min', 'head_x_max', 'head_y_max', 
        'right_eye_x_min', 'right_eye_y_min', 'right_eye_x_max', 'right_eye_y_max', 
        'left_eye_x_min', 'left_eye_y_min', 'left_eye_x_max', 'left_eye_y_max', 
        'gaze_x', 'gaze_y']]
        self.length = len(self.df)
        #print(self.df)
    
    def __getitem__(self, index):
        path, head_x_min, head_y_min, head_x_max, head_y_max, right_eye_x_min, right_eye_y_min, right_eye_x_max, right_eye_y_max, left_eye_x_min, left_eye_y_min, left_eye_x_max, left_eye_y_max, gaze_x, gaze_y = self.df.iloc[index]

        #print(right_eye_x_min)
        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        width, height = img.size
        imsize = torch.IntTensor([width, height])
        
        depth_path = path[:-4] + '.png'
        #print(depth_path)
        depthmap = Image.open(os.path.join(self.depth_dir, depth_path))
        depthmap = transform_depth(depthmap).squeeze(0)
        
        if right_eye_x_min == -1:
            eye_weight = torch.Tensor([0.0])

        else:
            eye_weight = torch.Tensor([1.0])
            right_eye_width = abs(right_eye_x_max - right_eye_x_min)
            right_eye_height = abs(right_eye_y_max - right_eye_y_min)
            left_eye_width = abs(left_eye_x_max - left_eye_x_min)
            left_eye_height = abs(left_eye_y_max - left_eye_y_min)
            k = 0.5
            right_eye_x_min = max(right_eye_x_min - k * right_eye_width, 0)
            right_eye_y_min = max(right_eye_y_min - k * right_eye_height, 0)
            right_eye_x_max = min(right_eye_x_max + k * right_eye_width, width)
            right_eye_y_max = min(right_eye_y_max + k * right_eye_height, height)
            left_eye_x_min = max(left_eye_x_min - k * left_eye_width, 0)
            left_eye_y_min = max(left_eye_y_min - k * left_eye_height, 0)
            left_eye_x_max = min(left_eye_x_max + k * left_eye_width, width)
            left_eye_y_max = min(left_eye_y_max + k * left_eye_height, height)

        if gaze_x == -1 or gaze_y == -1:
            inout = 0
        else:
            inout = 1
            gaze_x = gaze_x / width
            gaze_y = gaze_y / height
        
        if self.test == False and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            x_max = width - head_x_min
            x_min = width - head_x_max
            head_x_max = x_max
            head_x_min = x_min

            gaze_x = 1 - gaze_x

            if right_eye_x_min != -1:
                x_max = width - left_eye_x_min
                x_min = width - left_eye_x_max
                left_eye_x_max = x_max
                left_eye_x_min = x_min

                x_max = width - right_eye_x_min
                x_min = width - right_eye_x_max
                right_eye_x_max = x_max
                right_eye_x_min = x_min

        head = img.crop((head_x_min, head_y_min, head_x_max, head_y_max))
        head = transform_head(head)
        image = transform_img(img)


        if right_eye_x_min == -1:
            right_eye = torch.zeros(3, 36, 60)
            left_eye = torch.zeros(3, 36, 60)

        else:
            right_eye = img.crop((right_eye_x_min, right_eye_y_min, right_eye_x_max, right_eye_y_max))
            left_eye = img.crop((left_eye_x_min, left_eye_y_min, left_eye_x_max, left_eye_y_max))
            right_eye = transform_eye(right_eye)
            left_eye = transform_eye(left_eye)
        
        head_position = [(head_x_min+head_x_max)/(2*width), (head_y_min+head_y_max)/(2*height)]
        gaze_field = self.generate_data_field(head_position)
        gaze_field = torch.FloatTensor(gaze_field)

        head_channel = torch.zeros(224, 224)
        #head_channel[int(head_y_min*224/height):int(head_y_max*224/height), int(head_x_min*224/width):int(head_x_max*224/width)] = 1
        head_channel[int(head_position[1]*224), int(head_position[0]*224)] = 1
        head_position = torch.FloatTensor(head_position)

        gaze_point = torch.FloatTensor([gaze_x, gaze_y])
        gaze_direction = gaze_point - head_position

        inout = torch.FloatTensor([inout])

        if gaze_x == -1 or gaze_y == -1:
            heatmap = torch.zeros(64, 64)
        else:
            heatmap = torch.zeros(64, 64)
            heatmap = Gaussian_Heatmap(heatmap.numpy(), [gaze_x*64, gaze_y*64], 3)

        if self.test:
            return image, depthmap, head, head_channel, left_eye, right_eye, eye_weight, gaze_field, heatmap, gaze_direction, inout, imsize, gaze_point, head_position

        else:
            return image, depthmap, head, head_channel, left_eye, right_eye, eye_weight, gaze_field, heatmap, gaze_direction, inout, imsize, gaze_point

    def generate_data_field(self, head_position):
        """eye_point is (x, y) and between 0 and 1"""
        height, width = 224, 224
        x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
        y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
        grid = np.stack((x_grid, y_grid)).astype(np.float32)

        x, y = head_position
        x, y = x * width, y * height

        grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
        norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
        # avoid zero norm
        norm = np.maximum(norm, 0.1)
        grid /= norm
        return grid

    def __len__(self):
        return self.length
