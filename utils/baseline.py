from model import TBN
import copy
from torch import nn
from torch.nn.init import normal_, constant_
from context_gating import Context_Gating
from multimodal_gating import Multimodal_Gated_Unit
from ops.basic_ops import ConsensusModule
import numpy as np
import torch
from torch.nn import init
from torch.autograd import Function

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1) #output size is 1*1
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output
    
class TimeSpectrumAttention(nn.Module):
    def __init__(self,channel,kernel_size=7,reduction=1):
        super().__init__()
      
        self.sigmoid=nn.Sigmoid()
        self.pool_dim2 = nn.AdaptiveAvgPool2d((1, 7))
        self.pool_dim3 = nn.AdaptiveAvgPool2d((7, 1))
        self.se1=nn.Sequential(
            nn.Conv2d(7,4,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(4,7,1,bias=False)
        )
        self.se2=nn.Sequential(
            nn.Conv2d(7,4,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(4,7,1,bias=False)
        )
    def forward(self, x) :
        
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        x_dim2 = self.pool_dim2(max_result).squeeze(dim=1).squeeze(dim=1)
        x_dim3 = self.pool_dim3(max_result).squeeze(dim=1).squeeze(dim=2)
        x_dim2_1 = self.pool_dim2(avg_result).squeeze(dim=1).squeeze(dim=1)
        x_dim3_1 = self.pool_dim3(avg_result).squeeze(dim=1).squeeze(dim=2)
        max_out_1 = self.se1(x_dim2.unsqueeze(2).unsqueeze(3))
        max_out_2 = self.se2(x_dim3.unsqueeze(2).unsqueeze(3))
        avg_out_1 = self.se1(x_dim2_1.unsqueeze(2).unsqueeze(3))
        avg_out_2 = self.se2(x_dim3_1.unsqueeze(2).unsqueeze(3))
        dim1 = (max_out_1 + avg_out_1).squeeze(dim=2).squeeze(dim=2)
        dim2 = (max_out_2 + avg_out_2).squeeze(dim=2).squeeze(dim=2)
        dim1 = self.sigmoid(dim1.unsqueeze(1).unsqueeze(1))
        dim2 = self.sigmoid(dim2.unsqueeze(2).unsqueeze(1))
        out = {"dim1":dim1, "dim2":dim2}
        return out

class VSAM_Block(nn.Module):
    def __init__(self, modality, channel=512,reduction=16):
        super().__init__()
        self.modality = modality
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa_1=SpatialAttention(kernel_size=7)
        self.avgpool_3 = nn.AvgPool2d(kernel_size=7, stride=7)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.tsa = TimeSpectrumAttention(channel=channel,kernel_size=7, reduction=4)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, inputs):
        #####Fusion#####
        if len(self.modality) > 1:
            if len(self.modality) == 4:
                vision = inputs[:2]
                sensor = inputs[2:]
                base_out = torch.cat(inputs, dim=1)
                out_vision = torch.cat(vision, dim=1)
                out_sensor = torch.cat(sensor, dim=1)
            else:
                base_out = torch.cat(inputs, dim=1)
        else:
            base_out = inputs[0]
        ####==============================================================================####
        ########### 7 * 7 #####################
        #res_v = out_vision
        #res_s = out_sensor
        out_v = out_vision * self.sa_1(out_vision)
        out_s = out_sensor * (self.tsa(out_sensor)['dim1']+self.tsa(out_sensor)['dim2'])
        out_1 = torch.cat((out_v,out_s),dim=1) 
        ca = self.ca(out_1)
        out_1=out_1*ca + base_out
        
        base_out_3 = self.avgpool_3(base_out)
        base = base_out_3
                
        out_1 = self.pooling(out_1) + base 
        #out_2 = self.pooling(out_2)
        out_1 = out_1.squeeze(dim=2)
        out_1 = out_1.squeeze(dim=2)
        
        """out_2 = out_2.squeeze(dim=2)
        out_2 = out_2.squeeze(dim=2)
        out_3 = out_3.squeeze(dim=2)
        out_3 = out_3.squeeze(dim=2)"""

        base = base.squeeze(dim=2)
        base = base.squeeze(dim=2)
        out = {"mire": out_1, "base" : base}
        return out

class SC1_Block(nn.Module):
    def __init__(self, modality, channel=512,reduction=16):
        super().__init__()
        self.modality = modality
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa_1=SpatialAttention(kernel_size=7)
        self.avgpool_3 = nn.AvgPool2d(kernel_size=7, stride=7)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        #self.tsa = TimeSpectrumAttention(channel=channel,kernel_size=7, reduction=4)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, inputs):
        #####Fusion#####
        if len(self.modality) > 1:
            if len(self.modality) == 4:
                vision = inputs[:2]
                sensor = inputs[2:]
                base_out = torch.cat(inputs, dim=1)
                out_vision = torch.cat(vision, dim=1)
                out_sensor = torch.cat(sensor, dim=1)
            else:
                base_out = torch.cat(inputs, dim=1)
        else:
            base_out = inputs[0]
        ####==============================================================================####
        ########### v_sa + ca ####################
        out_v = out_vision * self.sa_1(out_vision)
        out_s = out_sensor
        out_1 = torch.cat((out_v,out_s),dim=1) 
        ca = self.ca(out_1)
        out_1=out_1*ca + base_out
        #############original #########################
        base_out_3 = self.avgpool_3(base_out)
        base = base_out_3
        ################################################
        out_1 = self.pooling(out_1) + base 
        out_1 = out_1.squeeze(dim=2)
        out_1 = out_1.squeeze(dim=2)
        base = base.squeeze(dim=2)
        base = base.squeeze(dim=2)
        out = {"mire": out_1, "base" : base}
        return out

class SC2_Block(nn.Module):
    def __init__(self, modality, channel=512,reduction=16):
        super().__init__()
        self.modality = modality
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa_1=SpatialAttention(kernel_size=7)
        self.avgpool_3 = nn.AvgPool2d(kernel_size=7, stride=7)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        #self.tsa = TimeSpectrumAttention(channel=channel,kernel_size=7, reduction=4)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, inputs):
        #####Fusion#####
        if len(self.modality) > 1:
            if len(self.modality) == 4:
                base_out = torch.cat(inputs, dim=1)
            else:
                base_out = torch.cat(inputs, dim=1)
        else:
            base_out = inputs[0]
        ####==============================================================================####
        ########### concatenate & only CA #####################
        ca = self.ca(base_out)
        out_1=base_out * ca + base_out
        #############original #########################
        base_out_3 = self.avgpool_3(base_out)
        base = base_out_3
        ################################################
        out_1 = self.pooling(out_1) + base 
        out_1 = out_1.squeeze(dim=2)
        out_1 = out_1.squeeze(dim=2)
        base = base.squeeze(dim=2)
        base = base.squeeze(dim=2)
        out = {"mire": out_1, "base" : base}
        return out

class SC3_Block(nn.Module):
    def __init__(self, modality, channel=512,reduction=16):
        super().__init__()
        self.modality = modality
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa_1=SpatialAttention(kernel_size=7)
        self.sa_2=SpatialAttention(kernel_size=7)
        self.avgpool_3 = nn.AvgPool2d(kernel_size=7, stride=7)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        #self.tsa = TimeSpectrumAttention(channel=channel,kernel_size=7, reduction=4)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, inputs):
        #####Fusion#####
        if len(self.modality) > 1:
            if len(self.modality) == 4:
                vision = inputs[:2]
                sensor = inputs[2:]
                base_out = torch.cat(inputs, dim=1)
                out_vision = torch.cat(vision, dim=1)
                out_sensor = torch.cat(sensor, dim=1)
            else:
                base_out = torch.cat(inputs, dim=1)
        else:
            base_out = inputs[0]
        ####==============================================================================####
        ########### 2_sa + ca #####################
        out_v = out_vision * self.sa_1(out_vision)
        out_s = out_sensor * self.sa_2(out_sensor)
        out_1 = torch.cat((out_v,out_s),dim=1)
        ca = self.ca(out_1)
        out_1=out_1*ca + base_out
        #############original #########################
        base_out_3 = self.avgpool_3(base_out)
        base = base_out_3
        ################################################
        out_1 = self.pooling(out_1) + base 
        out_1 = out_1.squeeze(dim=2)
        out_1 = out_1.squeeze(dim=2)
        base = base.squeeze(dim=2)
        base = base.squeeze(dim=2)
        out = {"mire": out_1, "base" : base}
        return out

class Sensor_Block(nn.Module):
    def __init__(self, modality, channel=512,reduction=16):
        super().__init__()
        self.modality = modality
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa_1=SpatialAttention(kernel_size=7)
        self.avgpool_3 = nn.AvgPool2d(kernel_size=7, stride=7)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.tsa = TimeSpectrumAttention(channel=channel,kernel_size=7, reduction=4)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, inputs):
        #####Fusion#####
        if len(self.modality) > 1:
            if len(self.modality) == 4:
                vision = inputs[:2]
                sensor = inputs[2:]
                base_out = torch.cat(inputs, dim=1)
                out_vision = torch.cat(vision, dim=1)
                out_sensor = torch.cat(sensor, dim=1)
            else:
                base_out = torch.cat(inputs, dim=1)
        else:
            base_out = inputs[0]
        ####==============================================================================####
        ########### sensor_FOR_tSNE #####################
        out_v = out_vision * self.sa_1(out_vision)
        out_s = out_sensor * (self.tsa(out_sensor)['dim1']+self.tsa(out_sensor)['dim2'])
        out_1 = torch.cat((out_v,out_s),dim=1) 
        ca = self.ca(out_1)
        out_1=out_1*ca + base_out
        
        base_out_3 = self.avgpool_3(base_out)
        base = base_out_3
        out_1 = self.pooling(out_1) + base 
        #out_2 = self.pooling(out_2)
        out_1 = out_1.squeeze(dim=2)
        out_1 = out_1.squeeze(dim=2)

        base = base.squeeze(dim=2)
        base = base.squeeze(dim=2)
        sensor_base = self.pooling(out_sensor)
        sensor_tf = self.pooling(out_s)
        sensor_base = sensor_base.squeeze(dim=2)
        sensor_base = sensor_base.squeeze(dim=2)
        sensor_tf = sensor_tf.squeeze(dim=2)
        sensor_tf = sensor_tf.squeeze(dim=2)
        out = {"mire": out_1, "base" : base, "sensor_base": sensor_base,"sensor_tf": sensor_tf}
        return out

class Only_Sensor_Block(nn.Module):
    def __init__(self, modality, channel=512,reduction=16):
        super().__init__()
        self.modality = modality
        self.avgpool_3 = nn.AvgPool2d(kernel_size=7, stride=7)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.tsa = TimeSpectrumAttention(channel=channel,kernel_size=7, reduction=4)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, inputs):
        #####Fusion#####
        if len(self.modality) > 1:
            if len(self.modality) == 4:
                vision = inputs[:2]
                sensor = inputs[2:]
                base_out = torch.cat(inputs, dim=1)
            else:
                base_out = torch.cat(inputs, dim=1)
        else:
            base_out = inputs[0]
        ####==============================================================================####
        ########### Only_sensor_FOR_tSNE #####################
        
        out_s = base_out * (self.tsa(base_out)['dim1']+self.tsa(base_out)['dim2'])
        out_s = out_s + base_out
        
        sensor_base = self.pooling(base_out)
        sensor_tf = self.pooling(out_s)
        sensor_base = sensor_base.squeeze(dim=2)
        sensor_base = sensor_base.squeeze(dim=2)
        sensor_tf = sensor_tf.squeeze(dim=2)
        sensor_tf = sensor_tf.squeeze(dim=2)
        out = {"mire": sensor_tf, "base" : sensor_base}
        return out

class Balance_Block(nn.Module):
    def __init__(self, modality, channel=512,reduction=16):
        super().__init__()
        self.modality = modality
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa_1=SpatialAttention(kernel_size=7)
        self.avgpool_3 = nn.AvgPool2d(kernel_size=7, stride=7)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.tsa = TimeSpectrumAttention(channel=channel,kernel_size=7, reduction=4)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, inputs):
        #####Fusion#####
        if len(self.modality) > 1:
            if len(self.modality) == 4:
                vision = inputs[:2]
                sensor = inputs[2:]
                base_out = torch.cat(inputs, dim=1)
                out_vision = torch.cat(vision, dim=1)
                out_sensor = torch.cat(sensor, dim=1)
            else:
                base_out = torch.cat(inputs, dim=1)
        else:
            base_out = inputs[0]
        ####==============================================================================####
        ########### sensor_FOR_tSNE #####################
        out_v = out_vision * self.sa_1(out_vision)
        out_s = out_sensor * (self.tsa(out_sensor)['dim1']+self.tsa(out_sensor)['dim2'])
        out_1 = torch.cat((out_v,out_s),dim=1) 
        ca = self.ca(out_1)
        out_1=out_1*ca + base_out
        
        base_out_3 = self.avgpool_3(base_out)
        base = base_out_3
                
        out_1 = self.pooling(out_1) + base 
        #out_2 = self.pooling(out_2)
        out_1 = out_1.squeeze(dim=2)
        out_1 = out_1.squeeze(dim=2)

        base = base.squeeze(dim=2)
        base = base.squeeze(dim=2)
        
        out = {"mire":out_1,"base":base,"v": out_vision, "s" : out_sensor, "vision": out_v,"sensor": out_s}
        return out

class Classification_Network(nn.Module):
    def __init__(self, feature_dim, modality, num_class,
                 consensus_type, before_softmax, num_segments):
        super().__init__()
        self.num_class = num_class
        self.modality = modality
        self.reshape = True
        self.consensus = ConsensusModule(consensus_type)
        self.before_softmax = before_softmax
        self.num_segments = num_segments

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        if len(self.modality) > 1:
            self._add_classification_layer(feature_dim) #multi scale pooling
        else:  # Single modality
            self._add_classification_layer(feature_dim)

    def _add_classification_layer(self, input_dim):

        std = 0.001
        if isinstance(self.num_class, (list, tuple)):  # Multi-task

            self.fc_verb = nn.Linear(input_dim, self.num_class[0])
            self.fc_noun = nn.Linear(input_dim, self.num_class[1])
            normal_(self.fc_verb.weight, 0, std)
            constant_(self.fc_verb.bias, 0)
            normal_(self.fc_noun.weight, 0, std)
            constant_(self.fc_noun.bias, 0)
        else:
            self.fc_action = nn.Linear(input_dim, self.num_class)
            normal_(self.fc_action.weight, 0, std)
            constant_(self.fc_action.bias, 0)
            self.weight = self.fc_action.weight
            self.bias = self.fc_action.bias
            ###################################################################################################################
            self.RGB_fc_action = nn.Linear(1024, self.num_class)
            normal_(self.RGB_fc_action.weight, 0, std)
            constant_(self.RGB_fc_action.bias, 0)
            self.RGB_weight = self.RGB_fc_action.weight
            self.RGB_bias = self.RGB_fc_action.bias
            ###################################################################################################################
            self.acc_fc_action = nn.Linear(1024, self.num_class)
            normal_(self.acc_fc_action.weight, 0, std)
            constant_(self.acc_fc_action.bias, 0)
            self.acc_weight = self.acc_fc_action.weight
            self.acc_bias = self.acc_fc_action.bias
            ###################################################################################################################
            self.flow_fc_action = nn.Linear(1024, self.num_class)
            normal_(self.flow_fc_action.weight, 0, std)
            constant_(self.flow_fc_action.bias, 0)
            self.flow_weight = self.flow_fc_action.weight
            self.flow_bias = self.flow_fc_action.bias
            ###################################################################################################################
            self.gyro_fc_action = nn.Linear(1024, self.num_class)
            normal_(self.gyro_fc_action.weight, 0, std)
            constant_(self.gyro_fc_action.bias, 0)
            self.gyro_weight = self.gyro_fc_action.weight
            self.gyro_bias = self.gyro_fc_action.bias
            ###################################################################################################################
            
    def forward(self, inputs):

        # Snippet-level predictions and temporal aggregation with consensus
        if isinstance(self.num_class, (list, tuple)):  # Multi-task
            # Verb
            base_out_verb = self.fc_verb(inputs)
            if not self.before_softmax:
                base_out_verb = self.softmax(base_out_verb)
            if self.reshape:
                base_out_verb = base_out_verb.view((-1, self.num_segments) + base_out_verb.size()[1:])
            output_verb = self.consensus(base_out_verb)

            # Noun
            base_out_noun = self.fc_noun(inputs)
            if not self.before_softmax:
                base_out_noun = self.softmax(base_out_noun)
            if self.reshape:
                base_out_noun = base_out_noun.view((-1, self.num_segments) + base_out_noun.size()[1:])
            output_noun = self.consensus(base_out_noun)

            output = (output_verb.squeeze(1), output_noun.squeeze(1))

        else:
            base_out = self.fc_action(inputs)
            if not self.before_softmax:
                base_out = self.softmax(base_out)
            if self.reshape:
                #s = (-1, self.num_segments) + base_out.size()[1:]
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output_pre = base_out
            output = self.consensus(base_out)
            output = output.squeeze(1)
#################get RGB logits##################################################################################################
            #a = inputs[:,:1024]
            RGB_base_out = self.RGB_fc_action(inputs[:,:1024])
            if not self.before_softmax:
                RGB_base_out = self.softmax(RGB_base_out)
            if self.reshape:
                RGB_base_out = RGB_base_out.view((-1, self.num_segments) + RGB_base_out.size()[1:])

            RGB_output = self.consensus(RGB_base_out)
            RGB_output = RGB_output.squeeze(1)
            return {'logits': output,'logits_pre': output_pre}

class Baseline(nn.Module):
    def __init__(self, num_segments, modality, base_model='BNInception',
                 new_length=None, consensus_type='avg', before_softmax=True,
                 dropout=0.2, midfusion='context_gating',):
        super().__init__()

        self.num_segments = num_segments
        self.modality = modality
        self.base_model = base_model
        self.new_length = new_length
        self.dropout = dropout
        self.before_softmax = before_softmax
        self.consensus_type = consensus_type
        self.midfusion = midfusion
        
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")
        
        self.feature_extract_network = TBN(self.num_segments, self.modality,
                                           self.base_model, self.new_length, 
                                           self.dropout)
        # self.vsam = VSAM_Block(self.modality,channel=1024*len(self.modality),reduction=16)
        #self.spatial_channel = SC2_Block(self.modality,channel=1024*len(self.modality),reduction=16)
        #self.am_ablation = SC3_Block(self.modality,channel=1024*len(self.modality),reduction=16)
        # self.sensor_only = Sensor_Block(self.modality,channel=1024*len(self.modality),reduction=16)
        #self.sensor_only = Only_Sensor_Block(self.modality,channel=1024*len(self.modality),reduction=16)
        self.balance = Balance_Block(self.modality,channel=1024*len(self.modality),reduction=16)
        self.feature_extractor = nn.Sequential(
            self.feature_extract_network,
            self.balance
        )
        self.fc = None
        self.fc_3 = None
        self.fc_7 = None
        self.softmax = nn.Softmax(dim=1)
        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.feature_extract_network.new_length, 
                   consensus_type, self.dropout)))

    @property
    def feature_dim(self):
        if len(self.modality) > 1:
            return 4096
        else:
            return 1024


    def extract_vector(self, x):
        return self.feature_extractor(x)["mire"]

    def forward(self, x):
        x = self.feature_extractor(x)
        out = self.fc(x)
        out.update(x)
        return out

    def update_fc(self, nb_classes, known_classes):
        fc = Classification_Network(1024*len(self.modality), self.modality, nb_classes, self.consensus_type, 
                                    self.before_softmax, self.num_segments)
        fc_7 = Classification_Network(1024*49*len(self.modality),self.modality,nb_classes,self.consensus_type,
                                    self.before_softmax,self.num_segments)
        fc_3 = Classification_Network(1024*9*len(self.modality),self.modality,nb_classes,self.consensus_type,
                                    self.before_softmax,self.num_segments)

        if self.fc is not None:
            nb_output = self.fc.num_class
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.fc_action.weight.data[:nb_output] = weight
            fc.fc_action.bias.data[:nb_output] = bias
            fc.fc_action.weight.data[:known_classes] = self.fc.weight
        if self.fc_3 is not None:
            nb_output = self.fc_3.num_class
            weight = copy.deepcopy(self.fc_3.weight.data)
            bias = copy.deepcopy(self.fc_3.bias.data)
            fc_3.fc_action.weight.data[:nb_output] = weight
            fc_3.fc_action.bias.data[:nb_output] = bias
            fc_3.fc_action.weight.data[:known_classes] = self.fc_3.weight
        if self.fc_7 is not None:
            nb_output = self.fc_7.num_class
            weight = copy.deepcopy(self.fc_7.weight.data)
            bias = copy.deepcopy(self.fc_7.bias.data)
            fc_7.fc_action.weight.data[:nb_output] = weight
            fc_7.fc_action.bias.data[:nb_output] = bias
            fc_7.fc_action.weight.data[:known_classes] = self.fc_7.weight
        

        del self.fc
        del self.fc_3
        del self.fc_7

        self.fc = fc
        self.fc_3 = fc_3
        self.fc_7 = fc_7

###############################################################################
    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
