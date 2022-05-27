import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from utils import *

class_dict = {
    0 : "Airplane",
    1 : "Automobile",
    2 : "Bird",
    3 : "Cat",
    4 : "Deer",
    5 : "Dog",
    6 : "Frog",
    7 : "Horse",
    8 : "Ship",
    9 : "Truck"
}

def load_pickle(experiment_name):
    with open('{}.pickle'.format(experiment_name), 'rb') as handle:
        experiment = pickle.load(handle)
        
    return experiment

def visualize_preds(experiment):
    plt.clf()
    
    # calculate stats
    gts = experiment.gt
    classes = np.unique(gts).astype(int)
    
    init_preds = experiment.init_pred
    
    last_preds = experiment.last_pred
    
    heatmap_init = np.zeros((len(classes), len(classes)))
    heatmap_last = np.zeros((len(classes), len(classes)))
    
    for class_id in classes:
        init_class_pred = init_preds[gts == class_id]
        last_class_pred = last_preds[gts == class_id]
        
        (unique_init, counts_init) = np.unique(init_class_pred, return_counts=True)

        freqs_init = dict(zip(unique_init, counts_init))
        
        (unique_last, counts_last) = np.unique(last_class_pred, return_counts=True)

        freqs_last = dict(zip(unique_last, counts_last))
        
        for sub_class_id in classes:
            if sub_class_id in freqs_init.keys():
                heatmap_init[class_id, sub_class_id] = freqs_init[sub_class_id]
                
            if sub_class_id in freqs_last.keys():
                heatmap_last[class_id, sub_class_id] = freqs_last[sub_class_id]
                
    
    # visualize
    
    f, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    sn.heatmap(data = heatmap_init, ax=axes[0])
    sn.heatmap(data = heatmap_last, ax=axes[1])
    
    axes[0].set_title("Original Results")
    axes[0].set_ylabel("GroundTruths")
    axes[0].set_xlabel("Predictions")
    
    axes[1].set_title("Attack Results")
    axes[1].set_ylabel("GroundTruths")
    axes[1].set_xlabel("Predictions")
    
    plt.savefig("heatmaps.png")
    
    return heatmap_init, heatmap_last
        
    
    
def statistics(experiment):
    stats = experiment.stats

    (unique, counts) = np.unique(stats, return_counts=True)

    freqs = dict(zip(unique, counts))
    
    orig_acc = 1 - freqs[0]/len(stats)
    
    attack_success = freqs[1] / (len(stats) - freqs[0])
    
    final_acc = freqs[2] / len(stats)
    
    print("Number of success: {}".format(freqs[1]))
    print("Number of mispreds: {}".format(freqs[0]))
    print("Number of fails: {}".format(freqs[2]))
    
    print("Original Acc: {:.2f}".format(orig_acc))
    print("Attack Success Rate: {:.2f}".format(attack_success))
    print("Accuracy After Attack: {:.3f}".format(final_acc))
    
    init_conf_success = experiment.init_conf[experiment.stats == 1]
    last_conf_success = experiment.last_conf[experiment.stats == 1]
    
    avg_init_conf_success_attacks = np.mean(init_conf_success)
    avg_last_conf_success_attacks = np.mean(last_conf_success)
    
    print("Average Init. Conf. Successful Attacks: {:.2f}".format(avg_init_conf_success_attacks))
    print("Average Last. Conf. Successful Attacks: {:.2f}".format(avg_last_conf_success_attacks))
    
    
    init_conf_unsuccess = experiment.init_conf[experiment.stats == 2]
    last_conf_unsuccess = experiment.last_conf[experiment.stats == 2]
    
    avg_init_conf_unsuccess_attacks = np.mean(init_conf_unsuccess)
    avg_last_conf_unsuccess_attacks = np.mean(last_conf_unsuccess)
    
    print("Average Init. Conf. Unsuccessful Attacks: {:.2f}".format(avg_init_conf_unsuccess_attacks))
    print("Average Last. Conf. Unsuccessful Attacks: {:.2f}".format(avg_last_conf_unsuccess_attacks))
    
    
def visualize_queries(experiment):
    plt.clf()
    queries = experiment.n_queries[experiment.stats == 1]
    
    print("Avg. number of queries: {:.1f}".format(np.mean(queries)))
    print("Median of queries: {:.0f}".format(np.median(queries)))
    
    plt.hist(queries, bins=int(queries.max() / 10) )
    
    plt.title("Num. Queries")
    
    plt.savefig("queries.png")
    
def vis_noise(experiment):
    plt.clf()
    
    success_noise = np.mean(experiment.noise[experiment.stats == 1], axis=0).transpose(1, 2, 0)

    plt.imshow(success_noise * 150)
    plt.savefig("noise")
    
def vis_imgs(experiment):
    plt.clf()
    num_imgs = 5
    plt.axis('off')

    f, axes = plt.subplots(3, num_imgs, figsize=(16, 5))
    
    imgs = experiment.imgs[experiment.stats == 1]
    noise = experiment.noise[experiment.stats == 1]
    
    init_conf = experiment.init_conf[experiment.stats == 1]
    last_conf = experiment.last_conf[experiment.stats == 1]
    
    init_pred = experiment.init_pred[experiment.stats == 1]
    last_pred = experiment.last_pred[experiment.stats == 1]
    
    
    indices = np.random.choice(range(imgs.shape[0]), num_imgs)
    
    for i, idx in enumerate(indices):
        axes[0, i].imshow(imgs[idx].transpose(1, 2, 0))
        axes[1, i].imshow((imgs[idx] + noise[idx]).transpose(1, 2, 0))
        
        axes[2, i].imshow(noise[idx].transpose(1, 2, 0) * 150)
        
        axes[0, i].set_xlabel("Conf: {:.2f}".format(init_conf[idx]))
        axes[1, i].set_xlabel("Conf: {:.2f}".format(last_conf[idx]))
        
        axes[0, i].set_title("Class: {}".format(class_dict[init_pred[idx]]))
        axes[1, i].set_title("Class: {}".format(class_dict[last_pred[idx]]))
        
        
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Adversarial")
    axes[2, 0].set_ylabel("Noise x 150")
    
    
    f.set_tight_layout(True)
    plt.savefig("imgs.png", bbox_inches='tight')

def compare_queries(experiment0, experiment1, experiment2):
    plt.clf()
    
    queries0 = experiment0.n_queries[experiment0.stats == 1]
    
    queries1 = experiment1.n_queries[experiment1.stats == 1]

    queries2 = experiment2.n_queries[experiment2.stats == 1]

    f, axes = plt.subplots(3, 1, figsize=(16, 10))

    axes[0].hist(queries0, bins=int(queries0.max() / 10) )
    axes[1].hist(queries1, bins=int(queries1.max() / 10) )
    axes[2].hist(queries2, bins=int(queries2.max() / 10) )
    
    #f.suptitle('Histogram of Queries', fontsize=16)
    
    axes[0].set_xlabel("Uniform Sampling")
    axes[1].set_xlabel("Gaussian Sampling")
    axes[2].set_xlabel("Saliency Sampling")
    
    plt.savefig("compare_queries.png")
    
def compare_noises(experiment0, experiment1, experiment2):
    plt.clf()
    
    success_noise_0 = np.mean(experiment0.noise[experiment0.stats == 1], axis=0).transpose(1, 2, 0)
    
    success_noise_1 = np.mean(experiment1.noise[experiment1.stats == 1], axis=0).transpose(1, 2, 0)

    success_noise_2 = np.mean(experiment2.noise[experiment2.stats == 1], axis=0).transpose(1, 2, 0)

    f, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(success_noise_0 * 150)
    axes[1].imshow(success_noise_1 * 150)
    axes[2].imshow(success_noise_2 * 150)
    
    axes[0].set_xlabel("Uniform Sampling")
    axes[1].set_xlabel("Gaussian Sampling")
    axes[2].set_xlabel("Saliency Sampling")
    
    plt.savefig("compare_noise.png")