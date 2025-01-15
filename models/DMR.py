#DMR & DMR-L

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
import scipy.linalg

EPSILON = 1e-8
T = 2
lamda = 1
yigou = 1
k = 1 #4
k1 = 2
init_epoch = 8#40
set_class_num = 32

class DMR(BaseLearner):
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
        #
        self.cluster_centers = []
        self.cluster_std = []
        #
        self._gen_data = []
        self.targets_all = []
        #
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

        
    def Confusion_Index(self):
        with torch.no_grad():
            y_pred, y_true = self._eval_cnn(self.test_loader)
            y_pred = y_pred.reshape(-1)  # type: ignore

            past_as_current_count = np.sum((y_true < self._known_classes) & (y_pred >= self._known_classes))
            new_as_past_count = np.sum((y_true >= self._known_classes) & (y_pred < self._known_classes))
            past_classes_count = np.sum(y_true < self._known_classes)
            new_classes_count = np.sum(y_true >= self._known_classes)
            ci = past_as_current_count / past_classes_count + new_as_past_count / new_classes_count
            return ci 
            
            
    def after_task(self):
        self._old_network = self._network.copy().freeze()  # type: ignore
        self._known_classes = self._total_classes
    
    def calculate_kl_divergence(self, pseudo_features, new_class_vectors, epsilon=1e-6):
        # 计算旧类和新类向量的均值
        mu_old = np.mean(pseudo_features, axis=0)
        mu_new = np.mean(new_class_vectors, axis=0)
        
        # 计算协方差矩阵
        cov_old = np.cov(pseudo_features, rowvar=False)
        cov_new = np.cov(new_class_vectors, rowvar=False)

        # 添加扰动项以确保协方差矩阵是可逆的
        cov_old += np.eye(cov_old.shape[0]) * epsilon
        cov_new += np.eye(cov_new.shape[0]) * epsilon

        # 计算trace项
        inv_cov_new = scipy.linalg.solve(cov_new, np.eye(cov_new.shape[0]), assume_a='pos')
        trace_term = np.trace(inv_cov_new @ cov_old)

        # 计算均值差异项
        mean_diff = mu_new - mu_old
        mean_diff_term = mean_diff.T @ inv_cov_new @ mean_diff

        # 使用对数行列式来计算行列式项
        log_det_cov_new = np.linalg.slogdet(cov_new)[1]
        log_det_cov_old = np.linalg.slogdet(cov_old)[1]
        log_det_term = log_det_cov_new - log_det_cov_old

        # 计算KL散度
        kl_divergence = 0.5 * (trace_term + mean_diff_term - pseudo_features.shape[1] + log_det_term)
        return kl_divergence

    def extract_class_vectors(self, class_indices, data_manager):
        # 获取特定类别的数据集
        _,_, dataset = data_manager.get_dataset(class_indices, source='train', mode='test', ret_data=True)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)

        # 提取特征向量
        vectors, _ = self._extract_all_vectors(loader)
        return vectors

    def observe_kl_divergence(self, data_manager):
        kl_divergences = []
        
        for class_idx in range(self._known_classes, self._total_classes):
            new_class_vectors = self.extract_class_vectors(np.arange(self._known_classes, self._total_classes), data_manager)
            
        pseudo_features = np.load('./gmm_gen/gmm_gen_data.npy')
        
        kl_div = self.calculate_kl_divergence(pseudo_features, new_class_vectors, 1e-6)
        kl_divergences.append(kl_div)
        
        # 计算KL散度的平均值
        min_kl_divergence = np.min(kl_divergences)
        return min_kl_divergence
    
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
        
        self._train(self.train_loader, self.test_loader,self.tsne_loader, args, data_manager)
    
    def _train(self, train_loader, test_loader,tsne_loader, args,data_manager):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

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
                    30,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(logging.INFO)
        # self.protoSave(data_manager)
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train', 
                                                                mode='test', ret_data=True)
            n_sample = len(targets)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size , shuffle=False, num_workers=4)
            gen_data = self._cluster_fit(idx_loader,"gmm-sigma", 2, n_sample)  # type: ignore
            self._gen_data.append(gen_data)  # type: ignore
            n_targets = np.full(n_sample,class_idx)
            self.targets_all.append(n_targets)  # type: ignore
        self._gen_data = np.vstack(self._gen_data)
        self.targets_all = np.hstack(self.targets_all)
        np.save("./gmm_gen/gmm_gen_data.npy",self._gen_data)
        
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
                #total_norm = nn.utils.clip_grad_norm_(self._network.feature_extract_network.rgb.parameters(), self._clip_gradient)
                #total_norm = nn.utils.clip_grad_norm_(self._network.feature_extract_network.flow.parameters(), self._clip_gradient)
                #total_norm = nn.utils.clip_grad_norm_(self._network.fc.parameters(), self._clip_gradient)
                if type(optimizer) == list:
                    optimizer[0].step()
                    optimizer[1].step()
                else:
                    optimizer.step()
                losses += loss.item()
                losses_protoAug += loss_protoAug.item()
                # losses_mixup += loss_mixup.item()
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

        self.protoSave(data_manager) 
        
        if self._total_classes == set_class_num:
            with torch.no_grad():
                min_kl_divergence = self.observe_kl_divergence(data_manager)
                logging.info(f'KL Divergence: {min_kl_divergence/4096}')
        
        
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train', 
                                                                mode='test', ret_data=True)
            n_sample = len(targets)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size , shuffle=False, num_workers=4)
            gen_data = self._cluster_fit(idx_loader,"gmm-sigma", 2, n_sample)  # type: ignore
            self._gen_data = np.concatenate((self._gen_data,gen_data),axis=0)
            n_targets = np.full(n_sample,class_idx)
            self.targets_all = np.concatenate((self.targets_all, n_targets),axis=0)
        self._gen_data = np.vstack(self._gen_data)
        self.targets_all = np.hstack(self.targets_all)
        np.save("./gmm_gen/gmm_gen_data.npy",self._gen_data)
        
        
        
    def _pesudo_prototypes(self):
        ###################################################################after clustering protos##########################################################################
        len_index = len(self.targets_all)
        index = np.random.choice(len_index,size=int(self._batch_size * int(self._known_classes/(self._total_classes-self._known_classes))))
        pesudo_protos = torch.from_numpy(self._gen_data[index])
        pesudo_targets = torch.from_numpy(self.targets_all[index])
        pesudo_protos = pesudo_protos.to(self._device).float().unsqueeze(1).repeat(1,8,1)
        pesudo_protos = pesudo_protos.view(-1,4096)
        pesudo_targets = pesudo_targets.to(self._device)
        soft_feat_aug = self._network.fc(pesudo_protos)["logits"]  # type: ignore
        loss_protoAug =  F.cross_entropy(soft_feat_aug, pesudo_targets)*100000
        
        return loss_protoAug

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

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

   
    def protoSave(self, data_manager):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
              data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
                )
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
                ##############################################################################################################
                # -1 / 1 choose strategy
                # for i in range(_targets.shape[0]): 
                #     ###########1.生成[1,-1,-1,...,-1]############
                #     one_hot = torch.nn.functional.one_hot(torch.tensor(_targets[i]), logits.shape[2]).to(self._device)
                #     encode = one_hot + one_hot - torch.ones(1,logits.shape[2]).to(self._device)
                #     a = torch.matmul(logits[i],encode.t())
                #     #a[a<0] = 0
                #     b = a.cpu().numpy()
                #     b = b / np.sum(b)
                #     proto_w.append(b)
                
                # proto_w = np.squeeze(proto_w,axis=2)

                ################################## information entropy choose strategy ########################################
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

    def _cluster_fit(self, loader, select ,k, n_sample): # "gmm" or "kmeans" choose one
        with torch.no_grad():
                data, _ = self._extract_all_vectors(loader)  # type: ignore
        if select == "gmm-sigma":
            gmm = GaussianMixture(n_components=k).fit(data)
            covariances = gmm.covariances_
            for i in range(len(covariances)):
                diag_matrix = np.diag(np.repeat(np.mean(np.diag(covariances[i])), covariances[0].shape[0]))
                covariances[i] = diag_matrix
            gmm.covariances_ = covariances
            n_sample = len(data)
            generated_data = gmm.sample(n_samples=n_sample)[0]
            
            return generated_data
        elif select == "gmm-Sigma":
            gmm = GaussianMixture(n_components=k).fit(data)
            n_sample = len(data)
            generated_data = gmm.sample(n_samples=n_sample)[0]
            
            return generated_data
        
    def _extract_all_vectors(self, loader):
        self._network.eval()  # type: ignore
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            for m in self._modality:
                _inputs[m] = _inputs[m].to(self._device)
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
    

    def _proof_show(self, model, loader, attr):
        model.eval()
        targets_all =[]
        sensor_tf_tsne = []
        if attr == "init":
            for i, (_, inputs, targets) in enumerate(loader):
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                with torch.no_grad():
                    sensor_tf = model.feature_extractor(inputs)["mire"]
                    sensor_tf = sensor_tf.view((-1,self._num_segments) + sensor_tf.size()[1:])
                    sensor_tf_avg = sensor_tf.cpu().numpy()
                    sensor_tf_avg = np.mean(sensor_tf_avg,axis=1)
                target1 = targets.numpy()
                targets_all.append(target1)
                sensor_tf_tsne.append(sensor_tf_avg)

            targets_all = np.hstack(targets_all)

            sensor_tf_tsne = np.concatenate(sensor_tf_tsne,axis=0)
            sensor_tf_tsne = np.vstack(sensor_tf_tsne)

            sensorTf = TSNE(perplexity=40,early_exaggeration=15,n_iter=1000,learning_rate=500,method='exact',init='pca').fit_transform(sensor_tf_tsne)
            np.save("./Tsne_proof_raw/actual_after_tsne_4.npy", sensorTf)
            np.save("./Tsne_proof_raw/actual_targets_4.npy", targets_all)
            classes_num = self._total_classes
            tsne_plot(sensorTf,targets_all,classes_num, attr)

        elif attr == "incremental_origin":
            index = np.random.choice(range(self._known_classes),size= 400,replace=True)
            proto_features = np.array(self._protos)[index]
            std_features = np.array(self._std)[index]
            proto_targets = index
            proto_time = np.zeros([proto_features.shape[0],8,proto_features.shape[1]])
            for i in range(proto_features.shape[0]):
                for j in range(8):
                        if len(self._modality) > 1:
                            proto_time[i,j,:] = proto_features[i,:] + np.random.normal(0,1,proto_features[i,:].shape) * std_features[i,:] * 2
                        else:
                            proto_time[i,j,:] = proto_features[i] + np.random.normal(0,1,proto_features[i].shape) * self._radius * 2
            proto_time = np.mean(proto_time,axis=1)
            proto_generation = TSNE(perplexity=40,early_exaggeration=15,n_iter=1000,learning_rate=500,method='exact',init='pca').fit_transform(proto_time)
            classes_num = self._total_classes
            tsne_plot(proto_generation, proto_targets, classes_num, attr)
        elif attr == "incremental_ours":
            index = np.random.choice(range(self._known_classes),size= 200,replace=True)
            proto_features_1 = np.array(self.cluster_centers)[2*index]
            proto_features_2 = np.array(self.cluster_centers)[2*index + 1]
            std_features_1 = np.array(self.cluster_std)[index*2]
            std_features_2 = np.array(self.cluster_std)[index*2 + 1]
            proto_targets = np.repeat(index, 2)
            mask = ((proto_targets == 1) | (proto_targets == 2))
            
            proto_time = np.zeros([proto_features_1.shape[0],8,proto_features_1.shape[1] * 2])
            for i in range(proto_features_1.shape[0]):
                for j in range(8):
                        ###########################  row-major order ，所以可以直接进行拼起来然后再view   ################################################################
                        proto_time[i,j,:proto_features_1.shape[1]] = proto_features_1[i,:] + np.random.normal(0,1,proto_features_1[i,:].shape) * std_features_1[i,:]
                        proto_time[i,j,proto_features_2.shape[1]:] = proto_features_2[i,:] + np.random.normal(0,1,proto_features_2[i,:].shape) * std_features_2[i,:]
            proto_multi = proto_time.reshape(-1,8,proto_features_1.shape[1])
            proto_multi = np.mean(proto_multi,axis=1)
            # proto_multi = proto_multi[mask]
            # proto_targets = proto_targets[mask]
            ##############################################################inter-class mixup###################################################################################
            proto_inter_mix_1 = proto_time[:,:,:4096]
            proto_inter_mix_2 = proto_time[:,:,4096:]
            proto_inter_mix_1 = np.mean(proto_inter_mix_1, axis=1)
            proto_inter_mix_2 = np.mean(proto_inter_mix_2, axis=1)
            beta = dist.Beta(0.5, 0.5)
            lambda_value = beta.sample().item()
            inter_mix_proto =  lambda_value * proto_inter_mix_1 + (1 - lambda_value) * proto_inter_mix_2
            inter_mix_proto = np.concatenate((inter_mix_proto, proto_multi),axis=0)
            inter_targets = np.concatenate((index,proto_targets),axis=0)
            proto_gen_ours = TSNE(perplexity=40,early_exaggeration=15,n_iter=1000,learning_rate=500,method='exact',init='pca').fit_transform(inter_mix_proto)
            classes_num = self._total_classes
            tsne_plot(proto_gen_ours,inter_targets,classes_num,attr)
            
def mixup_proto(self, newTensor, oldTensor, newTarget, oldTarget):

    # 生成Beta分布的lambda值
    beta_dist = dist.Beta(0.5, 0.5)  # 选择合适的Beta分布参数
    
    lambda_value = beta_dist.sample().item()
    test = self._total_classes
    target_b_onehot = F.one_hot(oldTarget, num_classes = self._total_classes).float()
    target_a_onehot = F.one_hot(newTarget, num_classes = self._total_classes).float()
    #a -- new 
    # 使用Mixup方法对张量和目标进行线性组合

    mixed_target = lambda_value * target_a_onehot + (1 - lambda_value) * target_b_onehot
    result = lambda_value * newTensor + (1 - lambda_value) * oldTensor

    return mixed_target, result

def tsne_plot(x, colors ,class_num, attr):

    #palette = np.array(sns.color_palette("bright", task_num))
    mycolors = ["indian red","windows blue", "amber","faded green", "dusty purple","red","coral","orange","gold","green","aqua","dodger blue","dark blue","plum","pink","greyish","tan","yellow","wheat","black","navy","olive","indigo","brown","sage","olive","cyan","salmon","orchid","blue","lime","amber"]
    #palette = np.array(sns.color_palette("Paired",n_colors = task_num))
    
    palette = np.array(sns.xkcd_palette(mycolors))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    """plt.xlim((-300, 300))
    plt.ylim((-300, 300))
    plt.axis('square')"""
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=20, c=palette[colors.astype(np.int8)])  # type: ignore

    txts = []
    for i in range(class_num):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=10)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    plt.savefig('./tsne/proto_gen' + "_" + attr + str(class_num) + '.png', dpi=400)
    return f, ax, txts