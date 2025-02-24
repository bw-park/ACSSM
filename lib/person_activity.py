"""
This source code is adapted and modified from
    https://github.com/reml-lab/mTAN
	https://github.com/plus-rkwitt/LatentSDEonHS
Authors: Satya Narayan Shukla and Benjamin M. Marlin. (ICLR 2021)
Authores Sebastian Zeng, Florian Graf, Roland Kwitt (NeurIPS 2023)
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
from sklearn.model_selection import train_test_split


class DatasetProvider():
    def __init__(self):
        pass
        
    @property
    def data_min(self):
        raise NotImplementedError()
    
    @property
    def data_max(self):
        raise NotImplementedError()
    
    @property 
    def num_test_samples(self):
        raise NotImplementedError()
    
    @property 
    def num_train_samples(self):
        raise NotImplementedError()
    
    @property 
    def num_val_samples(self):
        raise NotImplementedError()
    
    def get_train_loader(self, **kwargs):
        raise NotImplementedError()
    
    def get_test_loader(self, **kwargs):
        raise NotImplementedError()
    
    def get_val_loader(self, **kwargs):
        raise NotImplementedError()
    
    def decomposer(self):
        raise NotImplementedError()

class PersonActivity(object):
	urls = [
		'https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt',
	]

	tag_ids = [
		"010-000-024-033", # "ANKLE_LEFT",
		"010-000-030-096", # "ANKLE_RIGHT",
		"020-000-033-111", # "CHEST",
		"020-000-032-221"  # "BELT"
	]
	
	tag_dict = {k: i for i, k in enumerate(tag_ids)}

	label_names = [
		"walking",
		"falling",
		"lying down",
		"lying",
		"sitting down",
		"sitting",
		"standing up from lying",
		"on all fours",
		"sitting on the ground",
		"standing up from sitting",
		"standing up from sit on grnd"
	]

	label_dict = {
		"walking": 0,
		"falling": 1,
		"lying": 2,
		"lying down": 2,
		"sitting": 3,
		"sitting down" : 3,
		"standing up from lying": 4,
		"standing up from sitting": 4,
		"standing up from sit on grnd": 4,
		"on all fours": 5,
		"sitting on the ground": 6
	}

	def __init__(self, root, download=False, reduce='average', max_seq_length = 50, n_samples = None):
		self.root = root
		self.reduce = reduce
		self.max_seq_length = max_seq_length

		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found. You can use download=True to download it')
		
		self.data = torch.load(os.path.join(self.processed_folder, self.data_file))

		if n_samples is not None:
			self.data = self.data[:n_samples]

	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return len(self.data)

	def download(self):
		if self._check_exists():
			return

		os.makedirs(self.raw_folder, exist_ok=True)
		os.makedirs(self.processed_folder, exist_ok=True)

		def save_record(records, record_id, tt, vals, mask, labels):
			tt = torch.tensor(tt)
			vals = torch.stack(vals)
			mask = torch.stack(mask)
			labels = torch.stack(labels)

			# flatten the measurements for different tags
			vals = vals.reshape(vals.size(0), -1)
			mask = mask.reshape(mask.size(0), -1)
			assert(len(tt) == vals.size(0))
			assert(mask.size(0) == vals.size(0))
			assert(labels.size(0) == vals.size(0))

			seq_length = len(tt)
			# split the long time series into smaller ones
			offset = 0
			slide = self.max_seq_length // 2

			while (offset + self.max_seq_length < seq_length):
				idx = range(offset, offset + self.max_seq_length)
				first_tp = tt[idx][0]
				records.append((record_id, tt[idx] - first_tp, vals[idx], mask[idx], labels[idx]))
				offset += slide

		for url in self.urls:
			filename = url.rpartition('/')[2]
			download_url(url, self.raw_folder, filename, None)

			dirname = os.path.join(self.raw_folder)
			records = []
			first_tp = None

			for txtfile in os.listdir(dirname):
				with open(os.path.join(dirname, txtfile)) as f:
					lines = f.readlines()
					prev_time = -1
					tt = []

					record_id = None
					for l in lines:
						cur_record_id, tag_id, time, date, val1, val2, val3, label = l.strip().split(',')
						value_vec = torch.Tensor((float(val1), float(val2), float(val3)))
						time = float(time)

						if cur_record_id != record_id:
							if record_id is not None:
								save_record(records, record_id, tt, vals, mask, labels)
							tt, vals, mask, nobs, labels = [], [], [], [], []
							record_id = cur_record_id
						
							tt = [torch.zeros(1)]
							vals = [torch.zeros(len(self.tag_ids),3)]
							mask = [torch.zeros(len(self.tag_ids),3)]
							nobs = [torch.zeros(len(self.tag_ids))]
							labels = [torch.zeros(len(self.label_names))]
							
							first_tp = time
							time = round((time - first_tp)/ 10**5)
							prev_time = time
						else:
							# for speed -- we actually don't need to quantize it in Latent ODE 
                            # quatizing by 100 ms. 10,000 is one millisecond, 10,000,000 is one second
							time = round((time - first_tp)/ 10**5) 

						if time != prev_time:
							tt.append(time)
							vals.append(torch.zeros(len(self.tag_ids),3))
							mask.append(torch.zeros(len(self.tag_ids),3))
							nobs.append(torch.zeros(len(self.tag_ids)))
							labels.append(torch.zeros(len(self.label_names)))
							prev_time = time

						if tag_id in self.tag_ids:
							n_observations = nobs[-1][self.tag_dict[tag_id]]
							if (self.reduce == 'average') and (n_observations > 0):
								prev_val = vals[-1][self.tag_dict[tag_id]]
								new_val = (prev_val * n_observations + value_vec) / (n_observations + 1)
								vals[-1][self.tag_dict[tag_id]] = new_val
							else:
								vals[-1][self.tag_dict[tag_id]] = value_vec

							mask[-1][self.tag_dict[tag_id]] = 1
							nobs[-1][self.tag_dict[tag_id]] += 1

							if label in self.label_names:
								if torch.sum(labels[-1][self.label_dict[label]]) == 0:
									labels[-1][self.label_dict[label]] = 1
						else:
							assert tag_id == 'RecordID', 'Read unexpected tag id {}'.format(tag_id)
					save_record(records, record_id, tt, vals, mask, labels)
			
			torch.save(
				records,
				os.path.join(self.processed_folder, 'data.pt')
			)
				
	def _check_exists(self):
		for url in self.urls:
			filename = url.rpartition('/')[2]
			if not os.path.exists(
				os.path.join(self.processed_folder, 'data.pt')
			):
				return False
		return True

	@property
	def raw_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'processed')

	@property
	def data_file(self):
		return 'data.pt'

class HumanActivityDataset(Dataset):    
    
	input_dim = 12       # 4 (x,y,z) coordinates
	num_timepoints = 228 # 221 unique ones in {0,...,227}
	num_classes = 7      # originally 11, aggregated into 7
    
	def __init__(self, data_dir, mode='train', download=False, random_state=42):
        
		self.data = PersonActivity(data_dir, download=download, n_samples=8000)

		val, msk, tps, lab = [], [], [], []
		for r in self.data:
			val.append(r[2].unsqueeze(0))
			msk.append(r[3].unsqueeze(0))
			tps.append(r[1].unsqueeze(0))
			lab.append(r[4].unsqueeze(0))

		val = torch.cat(val)
		msk = torch.cat(msk)
		tid = torch.cat(tps)
		lab = torch.cat(lab)[:,:,0:HumanActivityDataset.num_classes].argmax(dim=2).long()

		generator = torch.Generator().manual_seed(random_state)
		full_idx = torch.arange(val.shape[0])
		idx_trn, idx_val, idx_tst = torch.utils.data.random_split(full_idx, [0.64, 0.16, 0.2], generator=generator) 

		if mode == 'train':
			idx = idx_trn.indices
		elif mode == 'test':
			idx = idx_tst.indices
		elif mode == 'valid':
			idx = idx_val.indices

		val = val[idx]
		msk = msk[idx].long()
		tid = tid[idx].long()
		lab = lab[idx]

		self.inp_obs, self.evd_obs = val, val
		self.inp_msk, self.evd_msk = msk, msk
		self.inp_tid, self.evd_tid = tid, tid
		
		self.aux_obs = lab
		self.aux_tid = tid
    
	def __getitem__(self, idx):
		inp_and_evd = {
			'inp_obs' : self.inp_obs[idx],
			'inp_msk' : self.inp_msk[idx],
			'inp_tid' : self.inp_tid[idx],
			'inp_tps' : self.inp_tid[idx]/HumanActivityDataset.num_timepoints,
			'evd_obs' : self.evd_obs[idx],
			'evd_msk' : self.evd_msk[idx],
			'evd_tid' : self.evd_tid[idx]/HumanActivityDataset.num_timepoints,
			'aux_obs' : self.aux_obs[idx],
			'aux_tid' : self.aux_tid[idx],
		}
		return inp_and_evd

	def __len__(self):
		return len(self.inp_obs)

	@property
	def has_aux(self):
		return True
        

class HumanActivityProvider(DatasetProvider):
    def __init__(self, data_dir=None, download=False, random_state=42):
        DatasetProvider.__init__(self)
        
        self._ds_trn = HumanActivityDataset(data_dir, 'train', download=download, random_state=random_state)
        self._ds_tst = HumanActivityDataset(data_dir, 'test', download=download, random_state=random_state)
        self._ds_val = HumanActivityDataset(data_dir, 'valid', download=download, random_state=random_state) 
        
    @property 
    def num_classes(self):
        return HumanActivityDataset.num_classes

    @property 
    def input_dim(self):
        return HumanActivityDataset.input_dim
    
    @property    
    def num_timepoints(self):
        return HumanActivityDataset.num_timepoints 
    
    @property
    def num_train_samples(self):
        return len(self._ds_trn)
    
    @property 
    def num_test_samples(self):
        return len(self._ds_tst)
    
    @property
    def num_val_samples(self):
        return len(self._ds_val)
    
    def get_train_loader(self, **kwargs):
        return DataLoader(self._ds_trn, **kwargs)
    
    def get_test_loader(self, **kwargs):
        return DataLoader(self._ds_tst, **kwargs)
    
    def get_val_loader(self, **kwargs):
        return DataLoader(self._ds_val, **kwargs)

if __name__ == '__main__':
    torch.manual_seed(1991)
    
    provider = HumanActivityProvider("./data/person_activity", download=False)
    dl_trn = provider.get_train_loader(batch_size=50, shuffle=True)
    batch = next(iter(dl_trn))
    
    obs = batch['inp_obs']
    truth = batch['evd_obs']
    obs_times = batch['inp_tid']
    b, t, d = obs.size()
    obs_valid = torch.ones(b, t).to(obs.device)
    labels = batch['aux_obs']
