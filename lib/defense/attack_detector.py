import cv2
import mmcv 
import decord
import numpy as np
from numpy.random import randint
import torch
import torchvision
import argparse
import os

from lib.data.transform.video_transforms import *
from lib.modeling import VideoModelWrapper

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        '''
        0: benign video
        1: sparse attack video
        2: dense attack video
        '''
        return int(self._data[1])

class AttackDetector(object):
    def __init__(self, video_length,
                       root, 
                       list_file,
                       transform,
                       model,
                       gamma1=0.175, gamma2=0.3,
                       video_loader='mmcv', device='cuda:0'):
        self.video_length = video_length
        self.root = root
        self.list_file = list_file
        self.transform = transform
        self.model = model
        self.device = device

        self.gamma1 = gamma1
        self.gamma2 = gamma2

        self.video_loader = video_loader
        assert video_loader in ['mmcv', 'cv2', 'decord']

        # video 
        self.benign_video, self.sparse_video, self.dense_video = [], [], []

        self._parse_list()

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def opencv_video_loader(self, path):
        
        cap = cv2.VideoCapture(path)
        frames = []
        while(cap.isOpened()):
            _, frame = cap.read()
            assert frame is not None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = Image.fromarray(frame)
            frames.append(frame)
        cap.release()
        return frames, len(frames)

    def mmcv_video_loader(self, path):
        
        video_loader = mmcv.VideoReader(path)
        return video_loader, len(video_loader)

    def decord_video_loader(self, path):

        video_loader = decord.VideoReader(path)
        return video_loader, len(video_loader)

    def sample_indices(self, num_frames, step=2):
        expanded_sample_length = self.video_length * step  
        if num_frames >= expanded_sample_length:
            start_pos = randint(num_frames - expanded_sample_length + 1)
            offsets = range(start_pos, start_pos + expanded_sample_length, step)
        elif num_frames > self.video_length*(step//2):
            start_pos = randint(num_frames - self.video_length*(step//2) + 1)
            offsets = range(start_pos, start_pos + self.video_length*(step//2), (step//2))
        elif num_frames > self.video_length:
            start_pos = randint(num_frames - self.video_length + 1)
            offsets = range(start_pos, start_pos + self.video_length, 1)
        else:
            offsets = np.sort(randint(num_frames, size=self.video_length))

        offsets =np.array([int(v) for v in offsets])  

        return offsets + 1

    def get_frame_output(self, video_loader):
        frame_outputs = []
        frame_buffers = []
        for frame in video_loader:
            frame_buffers.append(frame)
            if frame_buffers < self.video_length:
                continue
            else:
                input_frames = frame_buffers[-self.video_length:]
                input_frames = self.transform(input_frames)
                input_frames = input_frames.unsqueeze(0).to(self.device)
                output = self.model(input_frames)
                class_idx = torch.argmax(output).item()
                frame_outputs.append(class_idx)

        return frame_outputs

    def cal_exception_index(self, frame_outputs, num_frames):
        f = lambda x1, x2, x3: x1 != x2 and x2 != x3
        frame_outputs_pair = [(frame_outputs[i], frame_outputs[i+1], frame_outputs[i+2]) for i in range(len(frame_outputs)-2)]
        calculate = map(f, frame_outputs_pair)
        alpha = sum(calculate) / num_frames
        return alpha

    def adv_detection(self, index):
        video_path = self.video_list[index].path
        video_path = os.path.join(self.root, video_path)

        if self.video_loader == 'mmcv':
            video_loader, num_frames = self.mmcv_video_loader(video_path)
        elif self.video_loader == 'decord':
            video_loader, num_frames = self.decord_video_loader(video_path)
        else:
            video_loader, num_frames = self.opencv_video_loader(video_path)

        frame_outputs = self.get_frame_output(video_loader)
        alpha = self.cal_exception_index(frame_outputs, num_frames)

        if alpha < self.gamma1:
            self.benign_video.append(self.video_list[index].label)
        elif alpha > self.gamma1 and alpha < self.gamma2:
            self.sparse_video.append(self.video_list[index].label)
        else:
            self.dense_video.append(self.video_list[index].label)


def main_worker(model, root, list_file, clip_length, transform, gamma1, gamma2):
    attack_detector = AttackDetector(video_length=clip_length, root=root,
                list_file=list_file, transform=transform, model=model, gamma1=gamma1, gamma2=gamma2)

    for index in range(len(attack_detector.video_list)):
        attack_detector.adv_detection(index)

    benign_video, sparse_video, dense_video = attack_detector.benign_video, attack_detector.sparse_video, attack_detector.dense_video

    # sparse attack perforamce indicator
    tp_sparse = sum(np.array(sparse_video) == 1)
    tn_sparse = sum(np.array(benign_video) == 0)
    fn_sparse = sum(np.array(sparse_video) == 0)
    fp_sparse = sum(np.array(benign_video) == 1)

    # dense attack perforamce indicator
    tp_dense = sum(np.array(dense_video) == 2)
    tn_dense = sum(np.array(sparse_video) == 1)
    fn_dense = sum(np.array(dense_video) == 1)
    fp_dense = sum(np.array(sparse_video) == 2)

    # accuray
    sparse_acc = (tp_sparse + tn_sparse) / (tp_sparse + tn_sparse + fn_sparse + fp_sparse)
    dense_acc = (tp_dense + tn_dense) / (tp_dense + tn_dense + fn_dense + fp_dense)

    # precision
    sparse_precision = tp_sparse / (tp_sparse + fp_sparse)
    dense_precision = tp_dense / (tp_dense + fp_dense) 

    # recall
    sparse_recall = tp_sparse / (tp_sparse + fn_sparse)
    dense_recall = tp_dense / (tp_dense + fn_dense) 

    # f1-score
    sprase_f1_score = (2 * sparse_precision * sparse_recall) / (sparse_precision + sparse_recall)
    dense_f1_score = (2 * dense_precision * dense_recall) / (dense_precision + dense_recall)

    print('=======================================================')
    print('sparse attack detection accuracy: ', sparse_acc * 100.)
    print('dense attack detection accuracy: ', dense_acc * 100.)
    print('=======================================================')
    print('sparse attack detection precision: ', sparse_precision * 100.)
    print('dense attack detection precision: ', dense_precision * 100.)
    print('=======================================================')
    print('sparse attack detection recall: ', sparse_recall * 100.)
    print('dense attack detection recall: ', dense_recall * 100.)
    print('=======================================================')
    print('sparse attack detection F1 score: ', sprase_f1_score)
    print('dense attack detectin F1 score: ', dense_f1_score)

def get_parser():
    # options
    parser = argparse.ArgumentParser(
        description="Standard video-level testing")
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'hmdb51', 'kinetics'])
    parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'Flow', 'RGBDiff'])
    parser.add_argument('--root', default='', type=str)
    parser.add_argument('--file_list', default='', type=str)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--arch', type=str, default="")   
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--scale_size', type=int, default=256)
    parser.add_argument('--pool_fun', type=str, default='avg',
                        choices=['avg', 'max', 'topk'])
    parser.add_argument('--mean', type=list, nargs='+', default=[0.485, 0.456, 0.406])
    parser.add_argument('--std', type=list, nargs='+', default=[0.229, 0.224, 0.225])
    parser.add_argument('--video_length', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--model_type', type=str, default='3D')
    parser.add_argument('--gamma1', type=float, default=0.175)
    parser.add_argument('--gamma2', type=float, default=0.3)
 

    args = parser.parse_args()

    return args

def create_model(args):

    if args.dataset == 'ucf101':
        args.num_class = 101
    elif args.dataset == 'hmdb51':
        args.num_class = 51
    elif args.dataset == 'kinetics':
        args.num_class = 400
    else:
        raise ValueError('Unknown dataset '+args.dataset)


    model = VideoModelWrapper(args.num_class, args.video_length, args.modality,
                backbone_name=args.arch, backbone_type=args.model_type, agg_fun=args.pool_fun, dropout=args.dropout)
    


    checkpoint = torch.load(args.checkpoint)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_pred']))

    base_dict = checkpoint['state_dict']
    model.load_state_dict(base_dict)

    if args.gpus is not None:
        devices = args.gpus
    else:
        devices = list(range(args.workers))

    model = torch.nn.DataParallel(model.cuda(devices[0]), device_ids=devices)
    model.eval()

    return model


def get_transforms(args):
    cropping = torchvision.transforms.Compose([
        VideoResize(args.scale_size),
        VideoCenterCrop(args.input_size),
    ])

    transform = torchvision.transforms.Compose([
                        cropping,
                        VideoNormalize(mean=args.mean, std=args.std),
                        VideoToTensor(backbone_type=args.model_type),
                    ])

    return transform

if __name__ == "__main__":
    args = get_parser()
    model = create_model(args)
    transform = get_transforms(args)
    main_worker(model, args.root, args.list_file, 
            args.clip_length, transform, args.gamma1, args.gamma2)
