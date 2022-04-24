import time, pickle
import numpy as np
import cv2

def save_pickle(experiment_name, experiment_obj):
    with open('{}.pickle'.format(experiment_name), 'wb') as handle:
        pickle.dump(experiment_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_saliency_map(image):
    saliency_spectral = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap_spectral) = saliency_spectral.computeSaliency(image)
    return (saliencyMap_spectral * 255).astype("uint8")

def sample_saliency(saliencyMap):
    h, w = saliencyMap.shape

    sample_id = np.random.choice(range(h*w), p = saliencyMap.reshape(-1)/np.sum(saliencyMap))

    return sample_id//h, sample_id%w


class Experiment:
    def __init__(self, data_size, imsize):
        self.gt = np.zeros(data_size)
        self.imgs = np.zeros((data_size, 3, imsize[0], imsize[1]))
        
        # before attack
        self.init_pred = np.zeros(data_size)
        self.init_conf = np.zeros(data_size)
        
        # after attack
        self.last_pred = np.zeros(data_size)
        self.last_conf = np.zeros(data_size)
        
        self.noise = np.zeros((data_size, 3, imsize[0], imsize[1]))
        
        self.n_queries = np.zeros(data_size)
        
        # stats 
        self.stats = np.zeros(data_size) #0 if wrong prediction, 1 successful attack, 2 unsuccessful

    
    def add_record(self, record):
        idx = record.idx
        
        self.gt[idx] = record.gt
        self.imgs[idx] = record.orig_input
        
        self.init_pred[idx] = record.init_pred
        self.init_conf[idx] = record.init_conf
        
        self.last_pred[idx] = record.last_pred
        self.last_conf[idx] = record.last_conf
        
        
        self.noise[idx] = record.noise
        
        self.n_queries[idx] = record.n_queries
        
        self.stats[idx] = record.statue
        
class Record:
    def __init__(self, idx, imsize):
        self.idx = idx
        
        self.orig_input = None
        self.gt = None
        
        self.init_pred = None
        self.init_conf = None
        
        # after attack
        self.last_pred = None
        self.last_conf = None
        
        self.noise = np.zeros((3, imsize[0], imsize[1]))
        
        self.n_queries = None
        
        # stats 
        self.statue = None #0 if wrong prediction, 1 successful attack, 2 unsuccessful
    
    def __str__(self):
        
        output = "ID: {} | ".format(self.idx)
        
        if self.statue == 0:
            output += "Wrong Prediction! | GT: {} | Pred: {} | Conf: {:.2f}".format(self.gt, self.init_pred, self.init_conf)
        elif self.statue == 1:
            output += "Attack Successful!| "
            output += "GT: {} | Init. Pred: {} | Init. Conf: {:.2f} | ".format(self.gt, self.init_pred, self.init_conf)
            output += "Last Pred: {} | Last Conf: {:.2f} | Query: {}".format(self.last_pred, self.last_conf, self.n_queries)
            
            #output.format(self.gt, self.init_pred, self.init_conf, self.last_pred, self.last_conf, self.n_queries)
            
        elif self.statue == 2:
            output += "Attack Unsuccessful!| "
            output += "GT: {} | Init. Pred: {} | Init. Conf: {:.2f} | ".format(self.gt, self.init_pred, self.init_conf)
            output += "Last Pred: {} | Last Conf: {:.2f} | Query: {}".format(self.last_pred, self.last_conf, self.n_queries)
            
            #output.format(self.gt, self.init_pred, self.init_conf, self.last_pred, self.last_conf, self.n_queries)
        else:
            raise ValueError('Record statue is not set!')
            
        return output
    
class Timer:
    def __init__(self, max_iter):
        self.max_iter = max_iter
        
        self.durations = []
        self.current_iter = 0
        self.eta = 0
        
        
    def start(self):
        self.last_start = time.time()
        
    def end(self):
        self.last_end = time.time()
        
        self.last_duration = self.last_end - self.last_start
        
        self.durations.append(self.last_duration)
        
        self.current_iter = len(self.durations)
        
        self.eta = np.mean(self.durations) * (self.max_iter - self.current_iter)
        
        self.eta /= 60
        
    def __str__(self):
        return "Remaining Time: {:.2f} mins".format(self.eta)