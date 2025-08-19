import random

from typing import Iterable, Callable, Optional, Tuple, List

import numpy as np

from biosppy.signals import ecg


class Compose():
    def __init__(self, transforms: Iterable[Callable], transform_flag: Optional[List[int]] = None):
        self.transforms = transforms
        self.transform_flag = transform_flag

    def __call__(self, dat):
        if self.transform_flag is not None:
            for flag, t in zip(self.transform_flag, self.transforms):
                if flag:
                    dat = t(dat)
        else:
            for t in self.transforms:
                dat = t(dat)
        return dat

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


class RandomShift():
    def __init__(self, shift_tolerance: int):
        self.shift_tolerance = shift_tolerance

    def __call__(self, lead_data: np.array) -> np.array:
        roll_shift = random.randint(0, self.shift_tolerance)
        lead_data = lead_data[:, roll_shift:]
        lead_data = np.pad(lead_data, ((0, 0), (0, roll_shift)), 'constant', constant_values=(0, 0))
        return lead_data

    def __repr__(self):
        return f'{self.__class__.__name__}(shift_tolerance={self.shift_tolerance})'


class ZNormalize_1D():
    def __init__(self, mean: Optional[np.array] = None, std: Optional[np.array] = None, eps: float = 1e-5):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, lead_data: np.array) -> np.array:
        if self.mean is None:
            self.mean = np.mean(lead_data, axis=-1).reshape([-1, 1])
        if self.std is None:
            self.std = np.std(lead_data, axis=-1).reshape([-1, 1])
            self.std[self.std < self.eps] = 1.0
        lead_data = (lead_data - self.mean) / self.std
        return lead_data

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std}, eps={self.eps})'

class MINMAX_1D():
    def __init__(self, mean: Optional[np.array] = None, std: Optional[np.array] = None, eps: float = 1e-5):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, lead_data: np.array) -> np.array:
        self.min = np.min(lead_data, axis=-1).reshape([-1, 1])
        self.max = np.max(lead_data, axis=-1).reshape([-1, 1])
        if (self.max - self.min) == 0:
            print(lead_data)
        lead_data = (lead_data - self.min) / (self.max - self.min)
        return lead_data

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std}, eps={self.eps})'


class RandomCrop():
    def __init__(self, length: int, validate: bool = False):
        self.length = length
        self.validate = validate

    def __call__(self, lead_data: np.array) -> np.array:
        max_startpoint = len(lead_data[0]) - self.length
        start_point = random.randint(0, max_startpoint) if not self.validate else 0
        return lead_data[:, start_point:start_point + self.length]

    def __repr__(self):
        return f'{self.__class__.__name__}(length={self.length})'


class RandomSelectHB():
    def __init__(self, hb_cnt: int, sampling_rate: int, extract_length: Tuple[float, float] = (0.2, 0.4)):
        self.hb_cnt = hb_cnt
        self.sampling_rate = sampling_rate
        self.extract_length = extract_length

    def __call__(self, lead_data: np.array) -> np.array:
        # Use Lead II to get the R-peaks
        rpeak_lead = lead_data[1]
        rpeaks = ecg.hamilton_segmenter(rpeak_lead, self.sampling_rate)[0]
        _, new_rpeaks = ecg.extract_heartbeats(
            rpeak_lead, rpeaks, self.sampling_rate, before=self.extract_length[0], after=self.extract_length[1])

        if self.hb_cnt != -1:
            assert len(new_rpeaks) >= self.hb_cnt, "R-peaks less than random select count!"
            selected_cnt = self.hb_cnt
        else:
            selected_cnt = len(new_rpeaks)

        index = random.sample(range(0, len(new_rpeaks)), selected_cnt)
        selected_rpeaks = new_rpeaks[index]

        lead_data_split = []
        for one_channel in lead_data:
            template, _ = ecg.extract_heartbeats(
                one_channel, selected_rpeaks, self.sampling_rate,
                before=self.extract_length[0], after=self.extract_length[1])
            assert len(template) == len(selected_rpeaks), "Heartbeat Extract Error!"
            lead_data_split.append(template)

        return np.transpose(np.array(lead_data_split), (1, 0, 2))

    def __repr__(self):
        return f'{self.__class__.__name__}(hb_number={self.hb_cnt})'


class RandomSelectAnchorPositiveHB():
    def __init__(self,
                 sampling_rate: int,
                 positive_samples: Optional[int] = None,
                 extract_length: Tuple[float, float] = (0.2, 0.4)):
        self.sampling_rate = sampling_rate
        self.positive_samples = positive_samples
        self.extract_length = extract_length

    def __call__(self, lead_data: np.array) -> np.array:
        # Use Lead II to get the R-peaks
        rpeak_lead = lead_data[1]
        rpeaks = ecg.hamilton_segmenter(rpeak_lead, self.sampling_rate)[0]
        _, new_rpeaks = ecg.extract_heartbeats(
            rpeak_lead, rpeaks, self.sampling_rate, before=self.extract_length[0], after=self.extract_length[1])

        if self.positive_samples is not None:
            anchor_index = random.sample(range(1, len(new_rpeaks) - 1), 1)
            positive_index = random.sample(list(
                set(range(0, len(new_rpeaks))).difference(set(anchor_index))), self.positive_samples)
            anchor_rpeaks = new_rpeaks[anchor_index]
            positive_rpeaks = new_rpeaks[positive_index]
            rpeak_list = [anchor_rpeaks, positive_rpeaks]
        else:
            anchor_index = random.sample(range(1, len(new_rpeaks) - 1), 1)
            positive_index = random.sample([anchor_index[0] - 1, anchor_index[0] + 1], 1)

        three_hb = []
        for selected_rpeak in rpeak_list:
            lead_data_split = []
            for one_channel in lead_data:
                template, _ = ecg.extract_heartbeats(
                    one_channel, selected_rpeak, self.sampling_rate,
                    before=self.extract_length[0], after=self.extract_length[1])
                assert len(template) == len(selected_rpeak), "Heartbeat Extract Error!"
                lead_data_split.append(template)
            three_hb.append(lead_data_split)
        if self.positive_samples is not None:
            return np.array(
                [np.reshape(np.array(three_hb[0]), (-1, lead_data.shape[0], 300)).squeeze(0),
                 np.reshape(np.array(three_hb[1]), (-1, lead_data.shape[0], 300))])
        else:
            return np.array(
                [np.reshape(np.array(three_hb[0]), (-1, lead_data.shape[0], 300)).squeeze(0),
                 np.reshape(np.array(three_hb[1]), (-1, lead_data.shape[0], 300)).squeeze(0)])

    def __repr__(self):
        return f'{self.__class__.__name__}(positive_samples={self.positive_samples})'


class MITBIHRandomSelectThreeHB():
    def __init__(self, positive_samples: Optional[int] = None):
        self.positive_samples = positive_samples

    def __call__(self, lead_data: np.array) -> np.array:
        # Select Anchor, Positive, Negative heart beats
        if self.positive_samples is not None:
            anchor_index = random.sample(range(1, len(lead_data) - 1), 1)
            positive_index = random.sample(list(
                set(range(0, len(lead_data))).difference(set(anchor_index))), self.positive_samples)
        else:
            anchor_index = random.sample(range(1, len(lead_data) - 1), 1)
            positive_index = random.sample([anchor_index[0] - 1, anchor_index[0] + 1], 1)

        return np.array([np.reshape(lead_data[anchor_index], (-1, 280)),
                         np.reshape(lead_data[positive_index], (-1, 1, 280))])

    def __repr__(self):
        return f'{self.__class__.__name__}(positive_samples={self.positive_samples})'


class MITBIHConcateFullHB():
    def __init__(self, signal_len: Optional[int] = None):
        self.signal_len = signal_len

    def __call__(self, lead_data: np.array) -> np.array:

        if self.signal_len is not None:
            return np.reshape(lead_data, (1, -1))[:, :self.signal_len]
        else:
            return np.reshape(lead_data, (1, -1))

    def __repr__(self):
        return f'{self.__class__.__name__}'


class NoiseAddition():
    def __init__(self, signal_len: int):
        self.signal_len = signal_len

    def __call__(self, lead_data: np.array) -> np.array:
        noised_lead_data = []
        for one_channel_data in lead_data:
            one_channel_data += np.random.normal(0, 1, self.signal_len)
            noised_lead_data.append(one_channel_data)
        return np.array(noised_lead_data)

    def __repr__(self):
        return f'{self.__class__.__name__}(signal_len={self.signal_len})'


class Scaling():
    def __init__(self, scaling_ratio: float = 0.2):
        self.scaling_ratio = scaling_ratio

    def __call__(self, lead_data: np.array) -> np.array:
        return lead_data * self.scaling_ratio

    def __repr__(self):
        return f'{self.__class__.__name__}(scaling_ratio={self.scaling_ratio})'


class Negation():
    def __init__(self, negation_ratio: int = -1):
        self.negation_ratio = negation_ratio

    def __call__(self, lead_data: np.array) -> np.array:
        return lead_data * self.negation_ratio

    def __repr__(self):
        return f'{self.__class__.__name__}(negation_ratio={self.negation_ratio})'


class HorizontalFlipping():
    def __init__(self):
        pass

    def __call__(self, lead_data: np.array) -> np.array:
        for row in lead_data:
            for i in range((len(row) + 1) // 2):
                previuous_row = row[i]
                row[i] = row[len(row) - 1 - i]
                row[len(row) - 1 - i] = previuous_row
        return lead_data

    def __repr__(self):
        return f'{self.__class__.__name__}'


class Permutation():
    def __init__(self, num_subsegments: int = 10):
        self.num_subsegments = num_subsegments

    def __call__(self, lead_data: np.array) -> np.array:
        # Segment lead data
        subsegments = np.array(np.array_split(lead_data, self.num_subsegments, axis=1))  # shape: (10, 12, 500) for 5000

        # Random shuffle subsegments
        arr = np.random.permutation(np.arange(self.num_subsegments))
        shuffled_lead_data = []
        for idx in range(lead_data.shape[0]):
            for j in arr:
                shuffled_lead_data.append(subsegments[j, idx, :])
        shuffled_lead_data = np.array(shuffled_lead_data).reshape((lead_data.shape[0], -1))
        return shuffled_lead_data

    def __repr__(self):
        return f'{self.__class__.__name__}(num_subsegments={self.num_subsegments})'


class TimeWarping():
    def __init__(self, signal_len, num_subsegments: int = 10):
        self.signal_len = signal_len
        self.num_subsegments = num_subsegments

    def __call__(self, lead_data: np.array) -> np.array:
        # Segment lead data
        subsegments = np.array(np.array_split(lead_data, self.num_subsegments, axis=1))  # shape: (10, 12, 500)
        num_timewarping = np.random.choice(range(1, self.num_subsegments), 1, replace=False)[0]
        num_stretch = np.random.choice(num_timewarping, 1, replace=False)[0]
        timewarping_indices = np.random.choice(self.num_subsegments, num_timewarping, replace=False)
        timewarped_lead_data = []
        for idx in range(lead_data.shape[0]):
            timewarped_lead_segment: List[float] = []
            for j in range(self.num_subsegments):
                # Do stretch
                if j in timewarping_indices[:num_stretch]:
                    timewarped_lead_segment.extend(np.repeat(subsegments[j, idx, :], 2))
                # Do squeeze
                elif j in timewarping_indices[num_stretch:]:
                    timewarped_lead_segment.extend(subsegments[j, idx, ::2])
                else:
                    timewarped_lead_segment.extend(subsegments[j, idx, :])
            if len(timewarped_lead_segment) >= self.signal_len:
                timewarped_lead_data.append(np.array(timewarped_lead_segment)[:self.signal_len])
            else:
                full_lead = np.zeros(self.signal_len, dtype=np.float32)
                full_lead[:len(timewarped_lead_segment)] = timewarped_lead_segment
                timewarped_lead_data.append(np.array(full_lead))
        timewarped_lead_data = np.array(timewarped_lead_data)
        return timewarped_lead_data

    def __repr__(self):
        return f'{self.__class__.__name__}'
