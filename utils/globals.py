import torch
import time

class Timer:

    def __init__(self, enable=False, len_cache=1000):
        self.val_i = None
        self.val_time = None
        self.train_i = None
        self.train_time = None
        self.neighbor_sample_i = None
        self.neighbor_sample_time = None
        self.load_feature_i = None
        self.load_feature_time = None
        self.encodeT_i = None
        self.encodeT_time = None
        self.encodeN_i = None
        self.encodeN_time = None
        self.encodeE_i = None
        self.encodeE_time = None
        self.encodeCo_i = None
        self.encodeCo_time = None
        self.construct_patchs_i = None
        self.construct_patchs_time = None
        self.transform_i = None
        self.transform_time = None
        self.try_time = None
        
        self.enable = enable
        self.len_cache = len_cache

        if self.enable:            
            self.train_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.train_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

            self.val_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.val_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

            self.neighbor_sample_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.neighbor_sample_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]            
            
            self.load_feature_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.load_feature_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

            self.encodeT_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.encodeT_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

            self.encodeN_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.encodeN_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

            self.encodeE_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.encodeE_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

            self.encodeCo_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.encodeCo_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            
            self.construct_patchs_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.construct_patchs_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            
            self.try_time_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.try_time_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

        self.reset()

    def reset(self):
        self.val_i = 0
        self.val_time = 0
        
        self.train_i = 0
        self.train_time = 0
        
        self.neighbor_sample_i = 0
        self.neighbor_sample_time = 0
        
        self.load_feature_i = 0
        self.load_feature_time = 0
        
        self.encodeT_i = 0
        self.encodeT_time = 0
        
        self.encodeN_i = 0
        self.encodeN_time = 0
        
        self.encodeE_i = 0
        self.encodeE_time = 0
        
        self.encodeCo_i = 0
        self.encodeCo_time = 0
        
        self.construct_patchs_i = 0
        self.construct_patchs_time = 0
        
        self.transform_i = 0
        self.transform_time = 0
        
        self.try_i = 0
        self.try_time = 0      

    def compute_all(self):
        torch.cuda.synchronize()        
        
        for s, e in zip(self.train_s[:self.train_i], self.train_e[:self.train_i]):
            self.train_time += s.elapsed_time(e) / 1000
        self.train_i = 0

        for s, e in zip(self.val_s[:self.val_i], self.val_e[:self.val_i]):
            self.val_time += s.elapsed_time(e) / 1000
        self.val_i = 0

        for s, e in zip(self.neighbor_sample_s[:self.neighbor_sample_i],
                        self.neighbor_sample_e[:self.neighbor_sample_i]):
            self.neighbor_sample_time += s.elapsed_time(e) / 1000
        self.neighbor_sample_i = 0
        
        for s, e in zip(self.load_feature_s[:self.load_feature_i], self.load_feature_e[:self.load_feature_i]):
            self.load_feature_time += s.elapsed_time(e) / 1000
        self.load_feature_i = 0
        
        for s, e in zip(self.encodeT_s[:self.encodeT_i], self.encodeT_e[:self.encodeT_i]):
            self.encodeT_time += s.elapsed_time(e) / 1000
        self.encodeT_i = 0
        
        for s, e in zip(self.encodeN_s[:self.encodeN_i], self.encodeN_e[:self.encodeN_i]):
            self.encodeN_time += s.elapsed_time(e) / 1000
        self.encodeN_i = 0
        
        for s, e in zip(self.encodeE_s[:self.encodeE_i], self.encodeE_e[:self.encodeE_i]):
            self.encodeE_time += s.elapsed_time(e) / 1000
        self.encodeE_i = 0
        
        for s, e in zip(self.encodeCo_s[:self.encodeCo_i], self.encodeCo_e[:self.encodeCo_i]):
            self.encodeCo_time += s.elapsed_time(e) / 1000
        self.encodeCo_i = 0
        
        for s, e in zip(self.construct_patchs_s[:self.construct_patchs_i], self.construct_patchs_e[:self.construct_patchs_i]):
            self.construct_patchs_time += s.elapsed_time(e) / 1000
        self.construct_patchs_i = 0
        
        for s, e in zip(self.transform_s[:self.transform_i], self.transform_e[:self.transform_i]):
            self.transform_time += s.elapsed_time(e) / 1000
        self.transform_i = 0
        
        for s, e in zip(self.try_time_s[:self.try_i], self.try_time_e[:self.try_i]):
            self.try_time += s.elapsed_time(e) / 1000
        self.try_i = 0
        

    def print(self, prefix, epoch, logger):
        self.compute_all()
        logger.info('{}train time:{:.4f}s  val time:{:.4f}s'.format(prefix,
                                                              self.train_time/epoch,
                                                              self.val_time/epoch))
        ans = 'load feature time:{:.4f}s  encodeCo time:{:.4f}s  construct patchs time:{:.4f}s  transform time:{:.4f}s'\
              .format(self.load_feature_time/epoch, self.encodeCo_time/epoch, self.construct_patchs_time/epoch, self.transform_time/epoch)
        ans += '  neighbor sample time:{:.4f}s'.format(self.neighbor_sample_time/epoch)
        ans += '  other time:{:.4f}s'.format((self.train_time-self.load_feature_time-self.encodeCo_time-self.construct_patchs_time-self.transform_time-self.neighbor_sample_time)/epoch)
        ans += '  try time:{:.4f}s'.format(self.try_time/epoch)
        logger.info('{}{}'.format(prefix, ans))
        
    def get_all(self, epoch):
        self.compute_all()
        return {
                "train time": self.train_time/epoch,
                "val time": self.val_time/epoch,
                "load feature time": self.load_feature_time/epoch,
                "encodeCo time": self.encodeCo_time/epoch,
                "construct patchs time": self.construct_patchs_time/epoch,
                "transform time": self.transform_time/epoch,
                "neighbor sample time": self.neighbor_sample_time/epoch,
                "other time": (self.train_time-self.load_feature_time-self.encodeCo_time-self.construct_patchs_time-self.transform_time-self.neighbor_sample_time)/epoch
                }

    def set_enable(self):
        if not self.enable:
            self.train_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.train_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]

            self.val_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.val_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]

            self.neighbor_sample_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.neighbor_sample_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            
            self.load_feature_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.load_feature_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            
            self.encodeT_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.encodeT_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            
            self.encodeN_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.encodeN_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            
            self.encodeE_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.encodeE_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            
            self.encodeCo_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.encodeCo_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            
            self.construct_patchs_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.construct_patchs_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            
            self.transform_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.transform_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            
            self.try_time_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.try_time_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]

            self.enable = True

    
    def start_train(self):
        if self.enable:
            self.train_s[self.train_i].record()

    def end_train(self):
        if self.enable:
            self.train_e[self.train_i].record()
            self.train_i += 1
            if self.train_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.train_s[:self.train_i], self.train_e[:self.train_i]):
                    self.train_time += s.elapsed_time(e) / 1000
                self.train_i = 0

    def start_val(self):
        if self.enable:
            self.val_s[self.val_i].record()

    def end_val(self):
        if self.enable:
            self.val_e[self.val_i].record()
            self.val_i += 1
            if self.val_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.val_s[:self.val_i], self.val_e[:self.val_i]):
                    self.val_time += s.elapsed_time(e) / 1000
                self.val_i = 0
                
    def start_neighbor_sample(self):
        if self.enable:
            self.neighbor_sample_s[self.neighbor_sample_i].record()

    def end_neighbor_sample(self):
        if self.enable:
            self.neighbor_sample_e[self.neighbor_sample_i].record()
            self.neighbor_sample_i += 1
            if self.neighbor_sample_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.neighbor_sample_s[:self.neighbor_sample_i],
                                self.neighbor_sample_e[:self.neighbor_sample_i]):
                    self.neighbor_sample_time += s.elapsed_time(e) / 1000
                self.neighbor_sample_i = 0
    
    def start_load_feature(self):
        if self.enable:
            self.load_feature_s[self.load_feature_i].record()
            
    def end_load_feature(self):
        if self.enable:
            self.load_feature_e[self.load_feature_i].record()
            self.load_feature_i += 1
            if self.load_feature_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.load_feature_s[:self.load_feature_i], self.load_feature_e[:self.load_feature_i]):
                    self.load_feature_time += s.elapsed_time(e) / 1000
                self.load_feature_i = 0
    
    def start_encodeT(self):
        if self.enable:
            self.encodeT_s[self.encodeT_i].record()
    
    def end_encodeT(self):
        if self.enable:
            self.encodeT_e[self.encodeT_i].record()
            self.encodeT_i += 1
            if self.encodeT_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.encodeT_s[:self.encodeT_i], self.encodeT_e[:self.encodeT_i]):
                    self.encodeT_time += s.elapsed_time(e) / 1000
                self.encodeT_i = 0
    
    def start_encodeN(self):
        if self.enable:
            self.encodeN_s[self.encodeN_i].record()
    
    def end_encodeN(self):
        if self.enable:
            self.encodeN_e[self.encodeN_i].record()
            self.encodeN_i += 1
            if self.encodeN_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.encodeN_s[:self.encodeN_i], self.encodeN_e[:self.encodeN_i]):
                    self.encodeN_time += s.elapsed_time(e) / 1000
                self.encodeN_i = 0
    
    def start_encodeE(self):
        if self.enable:
            self.encodeE_s[self.encodeE_i].record()
    
    def end_encodeE(self):
        if self.enable:
            self.encodeE_e[self.encodeE_i].record()
            self.encodeE_i += 1
            if self.encodeE_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.encodeE_s[:self.encodeE_i], self.encodeE_e[:self.encodeE_i]):
                    self.encodeE_time += s.elapsed_time(e) / 1000
                self.encodeE_i = 0
    
    def start_encodeCo(self):
        if self.enable:
            self.encodeCo_s[self.encodeCo_i].record()
    
    def end_encodeCo(self):
        if self.enable:
            self.encodeCo_e[self.encodeCo_i].record()
            self.encodeCo_i += 1
            if self.encodeCo_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.encodeCo_s[:self.encodeCo_i], self.encodeCo_e[:self.encodeCo_i]):
                    self.encodeCo_time += s.elapsed_time(e) / 1000
                self.encodeCo_i = 0
    
    def start_construct_patchs(self):
        if self.enable:
            self.construct_patchs_s[self.construct_patchs_i].record()
    
    def end_construct_patchs(self):
        if self.enable:
            self.construct_patchs_e[self.construct_patchs_i].record()
            self.construct_patchs_i += 1
            if self.construct_patchs_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.construct_patchs_s[:self.construct_patchs_i], self.construct_patchs_e[:self.construct_patchs_i]):
                    self.construct_patchs_time += s.elapsed_time(e) / 1000
                self.construct_patchs_i = 0
    
    def start_transform(self):
        if self.enable:
            self.transform_s[self.transform_i].record()
    
    def end_transform(self):
        if self.enable:
            self.transform_e[self.transform_i].record()
            self.transform_i += 1
            if self.transform_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.transform_s[:self.transform_i], self.transform_e[:self.transform_i]):
                    self.transform_time += s.elapsed_time(e) / 1000
                self.transform_i = 0
                
    def start_try(self):
        if self.enable:
            self.try_time_s[self.try_i].record()
            
    def end_try(self):
        if self.enable:
            self.try_time_e[self.try_i].record()
            self.try_i += 1
            if self.try_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.try_time_s[:self.try_i], self.try_time_e[:self.try_i]):
                    self.try_time += s.elapsed_time(e) / 1000
                self.try_i = 0


timer = Timer()
