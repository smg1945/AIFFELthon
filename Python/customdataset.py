import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from torchvision import transforms
import torchaudio.transforms as ta_transforms
import math
import torchaudio
import cv2
import cmapy

SAMPLE_RATE = 4000
HOP_LENGTH = 40
N_MELS = 128
N_FFT = 1024
WIN_LENGTH = 800

class Biquad:

  # pretend enumeration
  LOWPASS, HIGHPASS, BANDPASS, PEAK, NOTCH, LOWSHELF, HIGHSHELF = range(7)

  def __init__(self, typ, freq, srate, Q, dbGain=0):
    types = {
      Biquad.LOWPASS : Biquad.lowpass,
      Biquad.HIGHPASS : Biquad.highpass,
      Biquad.BANDPASS : Biquad.bandpass,
      Biquad.PEAK : Biquad.peak,
      Biquad.NOTCH : Biquad.notch,
      Biquad.LOWSHELF : Biquad.lowshelf,
      Biquad.HIGHSHELF : Biquad.highshelf
    }
    assert typ in types
    self.typ = typ
    self.freq = float(freq)
    self.srate = float(srate)
    self.Q = float(Q)
    self.dbGain = float(dbGain)
    self.a0 = self.a1 = self.a2 = 0
    self.b0 = self.b1 = self.b2 = 0
    self.x1 = self.x2 = 0
    self.y1 = self.y2 = 0
    # only used for peaking and shelving filter types
    A = math.pow(10, dbGain / 40)
    omega = 2 * math.pi * self.freq / self.srate
    sn = math.sin(omega)
    cs = math.cos(omega)
    alpha = sn / (2*Q)
    beta = math.sqrt(A + A)
    types[typ](self,A, omega, sn, cs, alpha, beta)
    # prescale constants
    self.b0 /= self.a0
    self.b1 /= self.a0
    self.b2 /= self.a0
    self.a1 /= self.a0
    self.a2 /= self.a0

  def lowpass(self,A, omega, sn, cs, alpha, beta):
    self.b0 = (1 - cs) /2
    self.b1 = 1 - cs
    self.b2 = (1 - cs) /2
    self.a0 = 1 + alpha
    self.a1 = -2 * cs
    self.a2 = 1 - alpha

  def highpass(self, A, omega, sn, cs, alpha, beta):
    self.b0 = (1 + cs) /2
    self.b1 = -(1 + cs)
    self.b2 = (1 + cs) /2
    self.a0 = 1 + alpha
    self.a1 = -2 * cs
    self.a2 = 1 - alpha

  def bandpass(self, A, omega, sn, cs, alpha, beta):
    self.b0 = alpha
    self.b1 = 0
    self.b2 = -alpha
    self.a0 = 1 + alpha
    self.a1 = -2 * cs
    self.a2 = 1 - alpha

  def notch(self, A, omega, sn, cs, alpha, beta):
    self.b0 = 1
    self.b1 = -2 * cs
    self.b2 = 1
    self.a0 = 1 + alpha
    self.a1 = -2 * cs
    self.a2 = 1 - alpha

  def peak(self, A, omega, sn, cs, alpha, beta):
    self.b0 = 1 + (alpha * A)
    self.b1 = -2 * cs
    self.b2 = 1 - (alpha * A)
    self.a0 = 1 + (alpha /A)
    self.a1 = -2 * cs
    self.a2 = 1 - (alpha /A)

  def lowshelf(self, A, omega, sn, cs, alpha, beta):
    self.b0 = A * ((A + 1) - (A - 1) * cs + beta * sn)
    self.b1 = 2 * A * ((A - 1) - (A + 1) * cs)
    self.b2 = A * ((A + 1) - (A - 1) * cs - beta * sn)
    self.a0 = (A + 1) + (A - 1) * cs + beta * sn
    self.a1 = -2 * ((A - 1) + (A + 1) * cs)
    self.a2 = (A + 1) + (A - 1) * cs - beta * sn

  def highshelf(self, A, omega, sn, cs, alpha, beta):
    self.b0 = A * ((A + 1) + (A - 1) * cs + beta * sn)
    self.b1 = -2 * A * ((A - 1) + (A + 1) * cs)
    self.b2 = A * ((A + 1) + (A - 1) * cs - beta * sn)
    self.a0 = (A + 1) - (A - 1) * cs + beta * sn
    self.a1 = 2 * ((A - 1) - (A + 1) * cs)
    self.a2 = (A + 1) - (A - 1) * cs - beta * sn

  # perform filtering function
  def __call__(self, x):
    y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
    self.x2 = self.x1
    self.x1 = x
    self.y2 = self.y1
    self.y1 = y
    return y

  # provide a static result for a given frequency f
  def result(self, f):
    phi = (math.sin(math.pi * f * 2/(2*self.srate)))**2
    r =((self.b0+self.b1+self.b2)**2 - \
    4*(self.b0*self.b1 + 4*self.b0*self.b2 + \
    self.b1*self.b2)*phi + 16*self.b0*self.b2*phi*phi) / \
    ((1+self.a1+self.a2)**2 - 4*(self.a1 + 4*self.a2 + \
    self.a1*self.a2)*phi + 16*self.a2*phi*phi)
    if(r < 0):
      r = 0
    return r**(.5)

  # provide a static log result for a given frequency f
  def log_result(self, f):
    try:
      r = 20 * math.log10(self.result(f))
    except:
      r = -200
    return r

  # return computed constants
  def constants(self):
    return self.a1, self.a2, self.b0, self.b1, self.b2

  def __str__(self):
    return "Type:%d,Freq:%.1f,Rate:%.1f,Q:%.1f,Gain:%.1f" % (self.typ,self.freq,self.srate,self.Q,self.dbGain)


class CustomDataset(Dataset):
    def __init__(self, path, txt_list,
                 filter_params=False,
                 multi_channels=False,
                 clipping=False,
                 target_size=(300, 300),
                 th=5,
                 resizing=False):
        self.path = path
        self.txt_list = txt_list

        self.filter_params = filter_params
        self.multi_channels = multi_channels
        self.clipping = clipping
        self.target_size = target_size
        self.th = int(th * SAMPLE_RATE / HOP_LENGTH)
        self.resizing = resizing

        self.get_file_list()

        self.delete_list = []
        self.x = self.get_mel_spectrogram()
        self.y = self.get_label()
        self.delete_data()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_file_list(self):
        self.heas = []
        self.wavs = []
        self.tsvs = []

        for path_txt in self.txt_list:
            with open(path_txt, "r") as f:
                P_id, n, sr = f.readline().split()
                for _ in range(int(n)):
                    _, hea, wav, tsv = f.readline().split()
                    self.heas.append(hea)
                    self.wavs.append(wav)
                    self.tsvs.append(tsv)
        self.heas.sort()
        self.wavs.sort()
        self.tsvs.sort()

    # torchaudio로 필터링 적용
    def apply_filter(self, audio):
        # 필터 파라미터가 2개일 경우 필터 2회 적용
        if len(self.filter_params) == 2:
            for filter_param in self.filter_params:
                audio = self.filter_torchaudio(audio, filter_param)
        # 필터 파라미터가 1개일 경우 필터 1회 적용
        elif len(self.filter_params) in [4, 5]:
            audio = self.filter_torchaudio(audio, self.filter_params)
        else:
            raise ValueError("2개 이하의 필터를 적용해주세요.")
        return audio

    def filter_torchaudio(self, _audio, _params):
        biquad_filter = Biquad(*_params)
        a1, a2, b0, b1, b2 = biquad_filter.constants()
        _filtered_audio = torchaudio.functional.biquad(
            waveform=_audio,
            b0=b0,
            b1=b1,
            b2=b2,
            a0=1.0,
            a1=a1,
            a2=a2
        )
        return _filtered_audio

    def blank_clipping(self, img):
        img[img < 10/255] = 0
        img = np.transpose(np.array(img), (1, 2, 0))  # 텐서 > 넘파이
        # 3채널 이미지의 경우
        if self.multi_channels is True:
            copy = img.copy()   # 사본 생성
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)   # 흑백으로
        # 1채널 이미지의 경우
        else:
            copy = img
        # 행별로 black_percent 계산
        for row in range(img.shape[0] - 1, 0, -1):
            black_percent = len(np.where(img[row,:]==0)[0])/len(img[row,:])
            if black_percent < 0.80:
                break
        # clipping
        if (row - 1) > 0:
            copy = copy[:(row - 1), :, :]
        return transforms.ToTensor()(copy)

    def padding(self, spec, target_length, padding_value=0):
        pad_width = target_length - spec.shape[-1]
        padded_spec = torch.nn.functional.pad(spec, (0, pad_width, 0, 0), "constant", 0)
        return padded_spec

    def resize_spectrogram(self, spec, new_shape):
        resized_spec = transforms.functional.resize(img=spec, size=new_shape, antialias=None)
        return resized_spec

    def normalize_spectrogram(self, spec):
        normalized = (spec-spec.min()) / (spec.max() - spec.min())
        return normalized

    def get_mel_spectrogram(self):
        audio_list = []
        self.scale_list = []
        self.iter_list = []

        for path_wav in self.wavs:
            path = os.path.join(self.path, path_wav)
            # Torchaudio 이용하여 파일 로드
            x = torchaudio.load(path)[0]
            # Filtering
            if self.filter_params != False:
                x = self.apply_filter(x)

            # 멜스펙트로그램 변환
            ms = ta_transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                            n_fft=N_FFT,
                                            win_length=WIN_LENGTH,
                                            n_mels=N_MELS,
                                            hop_length=HOP_LENGTH)(x)
            ms = torchaudio.functional.amplitude_to_DB(ms, multiplier=10.,
                                                    amin=1e-10,
                                                    db_multiplier=1.0,
                                                    top_db=80.0)
            # 0~1로 정규화
            ms = self.normalize_spectrogram(ms)

            # 3채널 기능
            if self.multi_channels is True:
                ms *= 255
                ms = np.array(ms[0])
                ms = ms.astype(np.uint8)
                ms = cv2.applyColorMap(ms.astype(np.uint8), cmapy.cmap('magma'))
                ms = transforms.ToTensor()(ms)

            # Blank region clipping
            if self.clipping is True:
                ms = self.blank_clipping(ms)

            # 원본 wav의 길이가 th보다 길다면 Slicing
            if ms.shape[-1] > self.th + 1:
                scale = 1
                num_splits = ms.shape[-1] // self.th    # wav길이 == th의 배수
                if ms.shape[-1] % self.th != 0: # wav길이 != th의 배수
                    num_splits += 1
                self.iter_list.append(num_splits)

                for i in range(num_splits):
                    start_idx = i * self.th
                    end_idx = (i + 1) * self.th + 1
                    split = ms[..., start_idx:end_idx]

                    # th보다 길이가 짧다면
                    if split.shape[-1] < self.th + 1:
                        # Resizing
                        if self.resizing is True:
                            scale = (self.th + 1) / split.shape[-1]
                            target_shape = (split.shape[-2], self.th + 1)
                            split = self.resize_spectrogram(split, target_shape)
                        # Padding
                        else:
                            split = self.padding(split, self.th + 1)
                    # 최종 Resizing
                    resized = self.resize_spectrogram(split, self.target_size)
                    audio_list.append(resized)
                    if self.resizing is True:
                        self.scale_list.append(scale)

            # 원본 wav의 길이가 th보다 짧거나 같다면
            else:
                self.iter_list.append(1)
                scale = 1
                # th보다 짧다면
                if ms.shape[-1] < self.th + 1:
                    # Resizing
                    if self.resizing is True:
                        scale = (self.th + 1) / ms.shape[-1]
                        target_shape = (ms.shape[-2], self.th + 1)
                        ms = self.resize_spectrogram(ms, target_shape)
                    # Padding
                    else:
                        ms = self.padding(ms, self.th + 1)
                # 최종 resizing
                ms = self.resize_spectrogram(ms, self.target_size)
                audio_list.append(ms)
                if self.resizing is True:
                    self.scale_list.append(scale)
        return torch.stack(audio_list)

    def get_label(self):
        labels = []
        idx = 0
        for i, path_tsv in enumerate(self.tsvs):
            path = os.path.join(self.path, path_tsv)
            tsv_data = pd.read_csv(path, sep='\t', header=None)
            iter = self.iter_list[i]
            for _iter in range(iter):
                label = []
                if self.resizing is True:
                    scale = self.scale_list[sum(self.iter_list[:i]) + _iter]
                for _, tsv_row in tsv_data.iterrows():
                    # S1, S2에 속한다면
                    if tsv_row[2] in [1, 3]:
                        # 구간 불러와서 sr값 곱하고 hop_legth로 나누기
                        tsv_row[0] = tsv_row[0] * SAMPLE_RATE / HOP_LENGTH - (_iter * self.th)
                        tsv_row[1] = tsv_row[1] * SAMPLE_RATE / HOP_LENGTH - (_iter * self.th)
                        tsv_row[2] = 1 if tsv_row[2] == 1 else 2    # S1=1, S2=2
                        # 시작점 혹은 끝점이 구간 안에 존재한다면
                        if (0 <= tsv_row[0] < self.th or \
                            self.th >= tsv_row[1] > 0):
                            # 시작점이 0보다 작은 경우 0으로
                            if tsv_row[0] < 0:
                                tsv_row[0] = 0
                            # 끝점이 구간보다 큰 경우 구간의 끝점으로
                            if tsv_row[1] > self.th:
                                tsv_row[1] = self.th
                            # resize를 하였다면 라벨 값도 스케일링
                            if self.resizing is True:
                                tsv_row[0] *= scale
                                tsv_row[1] *= scale
                            # 최종 resize한 값 으로 보간
                            tsv_row[0] *= (self.target_size[1] - 1) / self.th
                            tsv_row[1] *= (self.target_size[1] - 1) / self.th
                            # label.append((int(tsv_row[2]), tsv_row[0], tsv_row[1]))
                            label.append([tsv_row[0] / self.target_size[1], 0 / self.target_size[0],
                                tsv_row[1] / self.target_size[1], self.target_size[0] / self.target_size[0],
                                int(tsv_row[2])])# xmin, ymin, xmax, ymax, cls
                        # 시작점 혹은 끝점이 구간 안에 존재하지 않는다면
                        else: continue
                    # S1, S2에 속하면서 시작점 혹은 끝점이 구간 안에 존재하는 경우를 제외한 나머지 경우
                    else: continue
                if(len(label)==0):
                    self.delete_list.append(idx)
                idx += 1
                labels.append(label)
        return labels

    def delete_data(self):
        delete_count=0
        for i in self.delete_list:
            del self.y[i-delete_count]
            delete_count+=1
        self.x = self.x[[i for i in range(self.x.size(0)) if i not in self.delete_list]]