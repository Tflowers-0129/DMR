#实现MMD指标，评估真实样本和pesudo-features的相似度

from ast import Lambda
from asyncio import selector_events
import logging
from pyexpat import features
from turtle import end_fill
#from types import NoneType
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.baselineplus import Baseline
from utils.toolkit import target2onehot, tensor2numpy
from utils.mydata_manager import MyDataManager
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from torchvision.utils import save_image
#from sklearn.metrics import ConfusionMatrixDisplay
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip,ColorJitter
#from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.manifold import TSNE
# For the UCI ML handwritten digits dataset
from sklearn.datasets import load_digits
import matplotlib.patheffects as pe
import seaborn as sns
import math
import torch.distributions as dist
import os
from sklearn.cluster import KMeans
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from scipy.stats import beta
EPSILON = 1e-8

init_epoch = 1

class MMD(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._protos = []
        self._std = []
        self._radius = 0
        self._radiuses = []
        self.rgb_radiuses = []
        self.flow_radiuses = []
        self.acc_radiuses = []
        self.gyro_radiuses = []

        self.cluster_centers = []
        self.cluster_std = []

        self._gen_data = []
        self.targets_all = []

        self._batch_size = args["batch_size"]
        self._num_workers = args["workers"]
        self._lr = args["lr"]
        self._epochs = args["epochs"]
        self._momentum = args["momentum"]
        self._weight_decay = args["weight_decay"]
        self._lr_steps = args["lr_steps"]
        self._modality = args["modality"]
        self._num_segments = args["num_segments"]
        self._partialbn = args["partialbn"]
        self._freeze = args["freeze"]
        self._clip_gradient = args["clip_gradient"]
        self.r_drop = nn.Dropout(p=1)#0.6#0.4#0.6#0.4#0.5#0.5
        self.f_drop = nn.Dropout(p=1)#0.4#0.2#0.2#0.0#0.2#0.3
        self.r_mask = nn.Dropout(p=0.6)
        self.f_mask = nn.Dropout(p=0.2)
        #self.inc_drop = nn.Dropout(p=0.5)
        self._network = Baseline(args["num_segments"], args["modality"], args["arch"],
                                          consensus_type=args["consensus_type"],
                                          dropout=args["dropout"], midfusion=args["midfusion"])

        self.transform = self.transform = nn.Sequential(
            RandomResizedCrop(size=(224, 224), scale=(0.8, 1.)),
            ColorJitter(),
            RandomHorizontalFlip(),
        )

        self.flow_transform = self.transform = nn.Sequential(
            RandomResizedCrop(size=(224, 224), scale=(0.8, 1.)),
            RandomHorizontalFlip(),
        )


    def after_task(self, data_manager):
        self._known_classes = self._total_classes
        
    def inter_guided(self, inputs):
        for m in self._modality:
            inputs[m] = inputs[m].to(self._device)
        comfeat = self._network.feature_extractor(inputs)["mire"]  # type: ignore
        return comfeat

    def incremental_train(self, data_manager,args):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes, self._known_classes)  # type: ignore
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers
        )
        
        tsne_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="train",
            mode="train",
        )
        self.tsne_loader = DataLoader(
            tsne_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), 
            source="test", 
            mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)  # type: ignore
        self._train(self.train_loader, self.test_loader,self.tsne_loader, args, data_manager)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
     
    
    def _train(self, train_loader, test_loader,tsne_loader, args,data_manager):
        self._network.to(self._device)

        optimizer = self._choose_optimizer()

        if type(optimizer) == list:
            scheduler_adam = optim.lr_scheduler.MultiStepLR(optimizer[0], self._lr_steps, gamma=0.1)
            scheduler_sgd = optim.lr_scheduler.MultiStepLR(optimizer[1], self._lr_steps, gamma=0.1)
            #scheduler_sgd = optim.lr_scheduler.CosineAnnealingLR(
            #    optimizer[1], 10)
            scheduler = [scheduler_adam, scheduler_sgd]
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self._lr_steps, gamma=0.1)  # type: ignore

        if self._cur_task == 0:
            self._init_train(train_loader, test_loader, optimizer, scheduler,args, data_manager)
        else:
            self._update_representation(train_loader, test_loader, tsne_loader, optimizer, scheduler,args, data_manager)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler,args, data_manager):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()  # type: ignore

            if self._partialbn:
                self._network.feature_extract_network.freeze_fn('partialbn_statistics')  # type: ignore
            if self._freeze:
                self._network.feature_extract_network.freeze_fn('bn_statistics')  # type: ignore

            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                targets = targets.to(self._device)
                comfeat = self.inter_guided(inputs)
                logits = self._network.fc(comfeat)['logits']  # type: ignore
                loss_clf = F.cross_entropy(logits, targets)
                loss = loss_clf

                if type(optimizer) == list:
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
                else:
                    optimizer.zero_grad()

                loss.backward()
               
                if self._clip_gradient is not None:
                        total_norm = nn.utils.clip_grad_norm_(self._network.parameters(), self._clip_gradient, 2)  # type: ignore

                if type(optimizer) == list:
                    optimizer[0].step()
                    optimizer[1].step()
                else:
                    optimizer.step()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if type(scheduler) == list:
                scheduler[0].step()
                scheduler[1].step()
            else:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    20,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    20,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(logging.INFO)
        # self.protoSave(data_manager)        
        # for class_idx in range(self._known_classes, self._total_classes):
        #     data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train', 
        #                                                         mode='test', ret_data=True)
        #     n_sample = len(targets)
        #     idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size , shuffle=False, num_workers=4)
        #     gen_data = self._cluster_fit(idx_loader,"gmm", 1, n_sample)  # type: ignore
        #     self._gen_data.append(gen_data)
        #     n_targets = np.full(n_sample,class_idx)
        #     self.targets_all.append(n_targets)
        # self._gen_data = np.vstack(self._gen_data)
        # self.targets_all = np.hstack(self.targets_all)
        # np.save("./gmm_gen/gmm_gen_data.npy",self._gen_data)
        
        for class_idx in range(0, 32):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train', 
                                                                mode='test', ret_data=True)
            n_sample = int(len(targets) / 2) * 2 + 20 
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size , shuffle=False, num_workers=4)
            gen_data = self._cluster_fit_kmeans(idx_loader)
            gen_data = gen_data
            self._gen_data.append(gen_data)
            n_targets = np.full(n_sample,class_idx)
            self.targets_all.append(n_targets)
        self._gen_data = np.vstack(self._gen_data)
        self.targets_all = np.hstack(self.targets_all)
        np.save("./gmm_gen/kmeans_gen_data.npy",self._gen_data)
        
        
        
        
    def _update_representation(self, train_loader, test_loader, tsne_loader, optimizer, scheduler, args, data_manager):
        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()  # type: ignore
            self._network.feature_extract_network.freeze_fn('modalities')  # type: ignore
            self._network.feature_extract_network.freeze_fn('bn_statistics')  # type: ignore
            
            for name, parms in self._network.named_parameters():  # type: ignore
                      if '.flow.' in name or '.rgb.'in name or '.stft.' in name or '.stft_2.' in name or '.acce.' in name or '.gyro.' in name:
                            parms.requires_grad = False

            losses = 0.0
            losses_clf = 0.0
            correct, total = 0, 0
            losses_protoAug = 0.0
            losses_mixup = 0.0
            for i, (_, inputs, targets) in enumerate(train_loader):
                comfeat = self.inter_guided(inputs)
                targets = targets.to(self._device)

                logits = self._network.fc(comfeat)["logits"]   # type: ignore
                loss_clf = F.cross_entropy(logits, targets)
                ###############################################################################################
                loss_protoAug = self._pesudo_prototypes()
                ###############################################################################################
                loss = loss_clf + loss_protoAug
                if type(optimizer) == list:
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()
                else:
                    optimizer.zero_grad()

                loss.backward()
                
                if self._clip_gradient is not None:
                        total_norm = nn.utils.clip_grad_norm_(self._network.parameters(), 20)  # type: ignore

                if type(optimizer) == list:
                    optimizer[0].step()
                    optimizer[1].step()
                else:
                    optimizer.step()
                losses += loss.item()
                # losses_protoAug += loss_protoAug#.item()
                losses_clf +=loss_clf.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if type(scheduler) == list:
                scheduler[0].step()
                scheduler[1].step()
            else:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f},  ClfLoss {:.3f} ,PR_loss {:.3f},Mix_loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_protoAug / len(train_loader),
                    losses_mixup / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
        logging.info(logging.INFO)
        
        # self.protoSave(data_manager)
        
        # for class_idx in range(self._known_classes, self._total_classes):
        #     data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train', 
        #                                                         mode='test', ret_data=True)
        #     n_sample = len(targets)
        #     idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size , shuffle=False, num_workers=4)
        #     gen_data = self._cluster_fit(idx_loader,"gmm", 1, n_sample)  # type: ignore
        #     self._gen_data = np.concatenate((self._gen_data,gen_data),axis=0)
        #     n_targets = np.full(n_sample,class_idx)
        #     self.targets_all = np.concatenate((self.targets_all, n_targets),axis=0)
        # self._gen_data = np.vstack(self._gen_data)
        # self.targets_all = np.hstack(self.targets_all)
        # np.save("./gmm_gen/gmm_gen_data.npy",self._gen_data)
        
    
        
        mmd_gmm = []
        mmd_p = []
        mmd_kmeans = []
    #     ############################################################# mmd_gmm #########################################################
    
    #     if self._cur_task == 7:
    #         length = 0
    #         for class_idx in range(0, self._total_classes):
    #             data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train', 
    #                                                                 mode='test', ret_data=True)
    #             idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size , shuffle=False, num_workers=4)
    #             with torch.no_grad():
    #                 comfeat_per_class, _ = self._extract_all_vectors(idx_loader)  # type: ignore
    #                 peduso_features = np.load("./gmm_gen/gmm_gen_data.npy")
                    
    #                 # 使用Scikit-learn计算高斯核
    #                 xx_kernel_sklearn = gaussian_kernel_sklearn(comfeat_per_class, comfeat_per_class, sigma=3.0)
    #                 yy_kernel_sklearn = gaussian_kernel_sklearn(peduso_features[length:length+len(targets)] , peduso_features[length:length+len(targets)], sigma=3.0)
    #                 xy_kernel_sklearn = gaussian_kernel_sklearn(comfeat_per_class, peduso_features[length:length+len(targets)], sigma=3.0)
    #                 length += len(targets)
    #                 # 转换为PyTorch张量
    #                 xx_kernel_sklearn = torch.tensor(xx_kernel_sklearn)
    #                 yy_kernel_sklearn = torch.tensor(yy_kernel_sklearn)
    #                 xy_kernel_sklearn = torch.tensor(xy_kernel_sklearn)

    #                 mmd_value_sklearn = torch.mean(xx_kernel_sklearn) - 2 * torch.mean(xy_kernel_sklearn) + torch.mean(yy_kernel_sklearn)
    #                 mmd = np.sqrt(np.abs(mmd_value_sklearn))  # type: ignore
    #                 mmd_gmm.append(mmd)
    #         np.savetxt("./mmd_gmm.txt",mmd_gmm)
    # ############################################################ mmd_prior ##############################################################
    
    #     if self._cur_task == 7:
    #         length_p = 0
    #         for class_idx in range(0, self._total_classes):
    #             data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train', 
    #                                                                 mode='test', ret_data=True)
    #             idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size , shuffle=False, num_workers=4)
    #             with torch.no_grad():
    #                 comfeat_per_class, _ = self._extract_all_vectors(idx_loader)  # type: ignore
    #                 pseudo_features = np.zeros([len(targets), 4096])
    #                 for i in range(length_p,length_p + len(targets) ):
    #                     pseudo_features[i] = self._protos[class_idx] + np.random.normal(0, 1) * np.mean(self._std[class_idx] )
                                                
    #                 # 使用Scikit-learn计算高斯核
    #                 xx_kernel_sklearn = gaussian_kernel_sklearn(comfeat_per_class, comfeat_per_class, sigma=3.0)
    #                 yy_kernel_sklearn = gaussian_kernel_sklearn(pseudo_features , pseudo_features, sigma=3.0)
    #                 xy_kernel_sklearn = gaussian_kernel_sklearn(comfeat_per_class, pseudo_features, sigma=3.0)

    #                 # 转换为PyTorch张量
    #                 xx_kernel_sklearn = torch.tensor(xx_kernel_sklearn)
    #                 yy_kernel_sklearn = torch.tensor(yy_kernel_sklearn)
    #                 xy_kernel_sklearn = torch.tensor(xy_kernel_sklearn)

    #                 mmd_value_sklearn = torch.mean(xx_kernel_sklearn) - 2 * torch.mean(xy_kernel_sklearn) + torch.mean(yy_kernel_sklearn)
    #                 mmd = np.sqrt(np.abs(mmd_value_sklearn))  # type: ignore
    #                 mmd_p.append(mmd)
    #         np.savetxt("./mmd_prior.txt",mmd_p)
        
        if  self._cur_task == 7:
            length = 0
            for class_idx in range(0, self._total_classes):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train', 
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size , shuffle=False, num_workers=4)
                with torch.no_grad():
                    comfeat_per_class, _ = self._extract_all_vectors(idx_loader)  # type: ignore
                    peduso_features = np.load("./gmm_gen/kmeans_gen_data.npy")
                    
                    # 使用Scikit-learn计算高斯核
                    xx_kernel_sklearn = gaussian_kernel_sklearn(comfeat_per_class, comfeat_per_class, sigma=3.0)
                    yy_kernel_sklearn = gaussian_kernel_sklearn(peduso_features[length:length + len(targets)] , peduso_features[length:length+len(targets)], sigma=3.0)
                    xy_kernel_sklearn = gaussian_kernel_sklearn(comfeat_per_class, peduso_features[length:length+len(targets)], sigma=3.0)
                    length += len(targets)
                    # 转换为PyTorch张量
                    xx_kernel_sklearn = torch.tensor(xx_kernel_sklearn)
                    yy_kernel_sklearn = torch.tensor(yy_kernel_sklearn)
                    xy_kernel_sklearn = torch.tensor(xy_kernel_sklearn)

                    mmd_value_sklearn = torch.mean(xx_kernel_sklearn) - 2 * torch.mean(xy_kernel_sklearn) + torch.mean(yy_kernel_sklearn)
                    mmd = np.sqrt(np.abs(mmd_value_sklearn))  # type: ignore
                    mmd_kmeans.append(mmd)
            np.savetxt("./mmd_kmeans.txt",mmd_kmeans)

            
    def _pesudo_prototypes(self):
        return 0

    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(loader):
                outputs = self.inter_guided(inputs)
                targets = targets.to(self._device)
                outputs = self._network.fc(outputs)['logits']   # type: ignore
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts == targets).sum()
                total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()  # type: ignore
        y_pred, y_true = [], []
        with torch.no_grad():
            for _, (_, inputs, targets) in enumerate(loader):
               
                comfeat = self.inter_guided(inputs)
                outputs = self._network.fc(comfeat)["logits"]  # type: ignore
                predicts = torch.topk(
                    outputs, k=self.topk, dim=1, largest=True, sorted=True
                )[
                    1
                ]  # [bs, topk]
                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

   
    def protoSave(self, data_manager):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
              data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source="train", mode="test", ret_data=True,)
              idx_loader = DataLoader(
                    idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
                )
              if len(self._modality) > 1:   
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                class_std = np.std(vectors, axis=0)
                self._protos.append(class_mean)
                self._std.append(class_std)
              else:
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                class_std = np.std(vectors, axis=0)
                self._protos.append(class_mean)
                self._std.append(class_std)

    def _extract_vectors(self, loader):
        self._network.eval()  # type: ignore
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            proto_w = []
            for m in self._modality:
                _inputs[m] = _inputs[m].to(self._device)
            _targets = _targets.numpy()
            comfeat = self.inter_guided(_inputs)
            logits = self._network.fc(comfeat)["logits_pre"].to(self._device)  # type: ignore
            score = self._network.softmax(logits)  # type: ignore

            if isinstance(self._network, nn.DataParallel):  # type: ignore
                for i in range(_targets.shape[0]):
                    proto_w.append()  # type: ignore
                proto_w = proto_w / proto_w.sum()  # type: ignore
                _vectors = tensor2numpy(
                    np.dot(proto_w, self._network.module.extract_vector(_inputs))  # type: ignore
                )

            else:
                for i in range(_targets.shape[0]):
                    for j in range(score.shape[1]):
                        test = score[i][j]
                        p = 0
                        for k in test:
                            if k>0:
                                p += (-k) * math.log(k)
                            else:
                                p += 0
                        proto_w.append(p)
                tensor_list = [torch.tensor(item) for item in proto_w]
                # 将列表中的Tensor元素堆叠在一起形成一个新的Tensor
                stacked_tensor = torch.stack(tensor_list)
                # 使用view方法将堆叠后的Tensor重塑为8x8的形状
                proto_w = stacked_tensor.view(-1, 8).cpu().numpy()
                sum_vector = np.sum(proto_w,axis=1,keepdims=True)  # type: ignore
                proto_w = proto_w / sum_vector
                ################################################################################################################
                c = self.inter_guided(_inputs)
                d = c.view((-1,self._num_segments) + c.size()[1:])  #8*8*1024

                proto_w = np.expand_dims(proto_w,2).repeat(d.shape[2],axis=2)
                _vectors = np.multiply(proto_w, d.cpu().numpy())  # type: ignore
                _vectors = np.sum(_vectors,axis=1)
                """c = self._network.extract_vector(_inputs)
                d = c.view((-1,self._num_segments) + c.size()[1:])  #8*8*1024
                _vectors = d.cpu().numpy()
                _vectors = np.mean(_vectors,axis=1)"""
            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _cluster_fit_kmeans(self, loader):
        with torch.no_grad():
            data, _ = self._extract_all_vectors(loader)

        kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
        labels = kmeans.labels_

        data_labeled = np.column_stack((data, labels))

        cluster_variances = []
        for label in np.unique(labels):
            cluster_data = data_labeled[data_labeled[:, -1] == label][:, :-1]
            cluster_variance = np.trace(np.cov(cluster_data.T))/cluster_data.shape[1]
            cluster_variances.append(cluster_variance)
        ################################生成######################################
        
        combined_samples = []
        samples = []
        for center, variance in zip(kmeans.cluster_centers_, cluster_variances):
            per_center_samples = []
            for i in range(int(len(data) / 2)):
                sample = center  + np.random.normal(0, 1, size=center.shape)*variance
                per_center_samples.append(sample)
            samples.append(per_center_samples)
            selected_samples = np.array(per_center_samples)[np.random.choice(int(len(data) / 2), size=20, replace=False)]
            combined_samples.append(selected_samples)
        beta_dist = beta(0.5, 0.5)
        lambda_value = beta_dist.rvs(size=combined_samples[0].shape[0]).reshape(-1, 1) 
        final_samples = combined_samples[0] * lambda_value + combined_samples[1] * (1 - lambda_value)
        samples.append(final_samples)
        samples = np.vstack(samples)
        final_selected_samples = np.array(samples)[np.random.choice(samples.shape[0], size=len(data), replace=False)]
        return final_selected_samples


    
    def _cluster_fit(self, loader, select ,k, n_sample):
        with torch.no_grad():
                data, _ = self._extract_all_vectors(loader)  # type: ignore
        
        if select == "gmm":
            gmm = GaussianMixture(n_components=k).fit(data)
            n_sample = len(data)
            generated_data = gmm.sample(n_samples=n_sample)[0]
            return generated_data
        else:
            pass
            
        
    def _extract_all_vectors(self, loader):
        self._network.eval()  # type: ignore
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            # for m in self._modality:
            #     _inputs[m] = _inputs[m].to(self._device)
            _targets = _targets.numpy()
            
            _vectors = tensor2numpy(
                self._consensus(self.inter_guided(_inputs))  # type: ignore
            )
            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
        
    def _consensus(self, x):
        output = x.view((-1, self._num_segments) + x.size()[1:])
        output = output.mean(dim=1, keepdim=True)
        output = output.squeeze(1)
        return output

from sklearn.metrics.pairwise import pairwise_kernels

def gaussian_kernel_sklearn(x, y, sigma=3.0):
    """
    Computes the Gaussian kernel between two sets of vectors x and y using Scikit-learn.

    Args:
    - x: A numpy array of shape (N, D) representing the first set of vectors.
    - y: A numpy array of shape (M, D) representing the second set of vectors.
    - sigma: The bandwidth parameter for the Gaussian kernel.

    Returns:
    - kernel_matrix: A numpy array of shape (N, M) containing the pairwise Gaussian kernel values.
    """
    return pairwise_kernels(x, y, metric='rbf', gamma=1.0 / (2 * sigma**2))


