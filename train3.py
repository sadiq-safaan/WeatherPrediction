#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code implements a DoubleEncoder MLP model to predict rainfall in mm/hr and the probability that it exceeds a 
1mm/hr threshold.  The model has five heads in the sequence 2 -> 1 -> 2.

The first two heads are the Rain Head and the Cloud Head which take in the following features each:

    Rain Head: ['goes_rain_rate_mm_hr','goes_rain_rate_max_50km','hrrr_rain_rate_mm_hr','hrrr_age_min',
                'grrm_lag10/20/40','grrmax_lag10/20/40','hrrm_lag','grrm_dif','grrm_acc']
    
    Cloud Head: ['goes_cod','goes_cod_max_50km','goes_cloud_top_height_m',
                 'goes_cod_gradient_per_km','goes_age_min','hrrr_cloud_liquid_water_kg_m2','hrrr_age_min','is_day',
                 'goes_miss_cod', 'goes_miss_cod50km', 'goes_miss_cth', 'gcod_lag5/10/25','gcod50_lag5/10/25',
                 'gcodg_lag5/10/25', 'gcth_lag5/10/25','hclw_lag','gcth_dif','gcth_acc']
    
The outputs of these two heads are 16 dimensional embeddings which are concatenated along with the following meta
features.

    Meta Features: ['hrrr_temperature_c','hrrr_pressure_hpa','hrrr_humidity_pct','day_c','day_s','time_s','time_c',
                    'lon_cos','lon_sin','lat_cos','lat_sin','hh_lag','ht_lag','hp_lag']

This concatenated vector is then passed to a Fusion Head which produces a final vector which is passed to the 
Probability and Regression heads for final computations of outputs.


@author: Safaan Sadiq
"""

import pandas as pd
import numpy as np
import pvlib as pv

import argparse
from preprocess3 import preprocess
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

epoch_num=15

# Double Encoder Architecture
class DoubleEncoder(nn.Module):
    def __init__(self,cl_dim,r_dim,m_dim,dropout=0.3):
        super().__init__()
        
        self.dropout=nn.Dropout(p=dropout)
        
        self.cloud_head=nn.Sequential(nn.Linear(cl_dim,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU())
        
        self.rain_head=nn.Sequential(nn.Linear(r_dim,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU())
        
        self.fusion_head=nn.Sequential(nn.Linear(32+m_dim,32), nn.ReLU(), nn.Linear(32,32), nn.ReLU())
        
        self.final_reg=nn.Sequential(nn.Linear(32,32), nn.ReLU(), nn.Linear(32,32), nn.ReLU(), nn.Linear(32,1))
        self.final_prob=nn.Sequential(nn.Linear(32,32), nn.ReLU(), nn.Linear(32,32), nn.ReLU(), nn.Linear(32,1))
        
        #self.final_reg=nn.Linear(32,1)
        #self.final_prob=nn.Linear(32,1)
        
    def forward(self,cl_vec,r_vec,m_vec):
        
        # First two heads
        cl_emb=self.dropout(self.cloud_head(cl_vec))
        r_emb=self.dropout(self.rain_head(r_vec))
        
        # Concatenate embeddings with meta features
        mix=torch.cat([cl_emb,r_emb,m_vec],dim=1)
        
        result_vec=self.dropout(self.fusion_head(mix))
        
        rain_reg=F.softplus(self.final_reg(result_vec))
        #rain_prob=torch.sigmoid(self.final_prob(result_vec))
        rain_prob_logit=self.final_prob(result_vec)
        
        return rain_reg, rain_prob_logit
    

# Preprocessing and Feature Engineering (see preprocess2.py for more details)
df,c_cols, r_cols, cl_cols, m_cols=preprocess('train_rows.parquet')
#m_cols=['day_s','day_c']

# Scale features (RobustScaler instead of StandardScalar due to skew in GOES Cloud Data)
scaler_r=RobustScaler()
scaler_c=RobustScaler()


# Scale Data
R_train=torch.tensor(scaler_r.fit_transform(df[r_cols]), dtype=torch.float32)
C_train=torch.tensor(scaler_c.fit_transform(df[cl_cols]), dtype=torch.float32)
M_train=torch.tensor(df[m_cols].values, dtype=torch.float32)

targ_train_1=torch.tensor(df[['target_rain_rate_mm_hr_t+1m','target_rain_probability_t+1m']].values, dtype=torch.float32)
targ_train_3=torch.tensor(df[['target_rain_rate_mm_hr_t+3m','target_rain_probability_t+3m']].values, dtype=torch.float32)
targ_train_5=torch.tensor(df[['target_rain_rate_mm_hr_t+5m','target_rain_probability_t+5m']].values, dtype=torch.float32)

# Dimensions for Initialization
r_dim=len(r_cols)
cl_dim=len(cl_cols)
m_dim=len(m_cols)

# Initialize Models
model_1=DoubleEncoder(cl_dim, r_dim, m_dim)
model_3=DoubleEncoder(cl_dim, r_dim, m_dim)
model_5=DoubleEncoder(cl_dim, r_dim, m_dim)

# Initialize Datasets and batch with DataLoader
data1=TensorDataset(C_train,R_train,M_train,targ_train_1)
loader1=DataLoader(data1, batch_size=128, shuffle=True)
data3=TensorDataset(C_train,R_train,M_train,targ_train_3)
loader3=DataLoader(data3, batch_size=128, shuffle=True)
data5=TensorDataset(C_train,R_train,M_train,targ_train_5)
loader5=DataLoader(data5, batch_size=128, shuffle=True)


# Create Hybrid Loss Function
ms=nn.MSELoss()
bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(14.0))

bce_eval=nn.BCEWithLogitsLoss() # For evaluation

def custom_loss(reg_p, reg_r, prob_p,prob_r):
    return 0.3*ms(reg_p,reg_r)+0.7*bce(prob_p,prob_r)


# Set up model training loop
def model_train(model,optimizer,scheduler,loader):
    model.train()
    model.to(device)
    for epoch in range(epoch_num):
        total_loss=0.0
        msloss=0.0
        bceloss=0.0
        for C, R, M, T in loader:
            C=C.to(device)
            R=R.to(device)
            M=M.to(device)
            T=T.to(device)
            
            # Zero Grad
            optimizer.zero_grad()
        
            # Forward Pass
            r,p=model(C,R,M)

            # Loss
            loss=custom_loss(r,T[:,0:1],p,T[:,1:2])
            
            # Backward Pass
            loss.backward()
            
            # Step
            optimizer.step()
            
            total_loss += loss.item()
            msloss += ms(r,T[:,0:1])
            bceloss += bce_eval(p,T[:,1:2])
            
        scheduler.step()
        
        avg_loss=total_loss/len(loader)
        m_loss=msloss/len(loader)
        b_loss=bceloss/len(loader)
        print(f'Epoch {epoch+1}/{epoch_num} | Loss: {avg_loss:.4f} | MSE Loss: {m_loss:.4f} | BCE (Unweighted) Loss: {b_loss:.4f}')

# Set up training
optimizer1=torch.optim.AdamW(model_1.parameters(), lr=3e-4, weight_decay=1e-4)
optimizer3=torch.optim.AdamW(model_3.parameters(), lr=3e-4, weight_decay=1e-4)
optimizer5=torch.optim.AdamW(model_5.parameters(), lr=3e-4, weight_decay=1e-4)

scheduler1=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=epoch_num)
scheduler3=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=epoch_num)
scheduler5=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer5, T_max=epoch_num)

# Train Models
model_train(model_1, optimizer1, scheduler1, loader1)
model_train(model_3, optimizer3, scheduler3, loader3)
model_train(model_5, optimizer5, scheduler5, loader5)

del C_train, R_train, M_train, targ_train_1, targ_train_3, targ_train_5, df, loader1, loader3, loader5, data1, data3, data5

###############################################################################


# Setting up Validation Dataset

df,c_cols, r_cols, cl_cols, m_cols=preprocess('val_rows.parquet')
#m_cols=['day_s','day_c']

R_val=torch.tensor(scaler_r.transform(df[r_cols]), dtype=torch.float32)
C_val=torch.tensor(scaler_c.transform(df[cl_cols]), dtype=torch.float32)
M_val=torch.tensor(df[m_cols].values, dtype=torch.float32)

targ_val_1=torch.tensor(df[['target_rain_rate_mm_hr_t+1m','target_rain_probability_t+1m']].values, dtype=torch.float32)
targ_val_3=torch.tensor(df[['target_rain_rate_mm_hr_t+3m','target_rain_probability_t+3m']].values, dtype=torch.float32)
targ_val_5=torch.tensor(df[['target_rain_rate_mm_hr_t+5m','target_rain_probability_t+5m']].values, dtype=torch.float32)


# Initialize Datasets and batch with DataLoader
data1=TensorDataset(C_val,R_val,M_val,targ_val_1)
loader1=DataLoader(data1, batch_size=128, shuffle=False)
data3=TensorDataset(C_val,R_val,M_val,targ_val_3)
loader3=DataLoader(data3, batch_size=128, shuffle=False)
data5=TensorDataset(C_val,R_val,M_val,targ_val_5)
loader5=DataLoader(data5, batch_size=128, shuffle=False)

# Evaluation Loop
def model_eval(model,loader):
    model.eval()
    
    all_probs=[]
    all_actuals=[]
    all_rates=[]
    all_rate_actuals=[]
    
    bce_loss=0.0
    msloss=0.0
    
    with torch.no_grad():
        for C, R, M, T in loader:
            C=C.to(device)
            R=R.to(device)
            M=M.to(device)
            T=T.to(device)
            
            # Forward Pass
            r,p_logit=model(C,R,M)
            probs = torch.sigmoid(p_logit).cpu().numpy().flatten()
            actuals = T[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_actuals.append(actuals)
            all_rates.append(r.cpu().numpy().flatten())
            all_rate_actuals.append(T[:, 0].cpu().numpy())
            
            # Validation Metrics
            msloss += ms(r,T[:,0:1])
            bce_loss += bce_eval(p_logit,T[:,1:2])
            
            
    all_probs = np.concatenate(all_probs)
    all_actuals = np.concatenate(all_actuals)
    all_rates = np.concatenate(all_rates)
    all_rate_act = np.concatenate(all_rate_actuals)
    
    rain_mask = all_rate_act >= 1.0
    
    print("BCE:        ", (bce_loss/len(loader)).item())
    print("MSE:        ", (msloss/len(loader)).item())
    print("AUC:        ", roc_auc_score(all_actuals, all_probs))
    print('PR AUC:     ', average_precision_score(all_actuals,all_probs))
    print("Brier score:", brier_score_loss(all_actuals, all_probs))
    print("RMSE (all): ", np.sqrt(np.mean((all_rates - all_rate_act)**2)))
    print("MAE  (all): ", np.mean(np.abs(all_rates - all_rate_act)))
    print("RMSE (rain only):", np.sqrt(np.mean((all_rates[rain_mask] - all_rate_act[rain_mask])**2)))
    print("MAE  (rain only):", np.mean(np.abs(all_rates[rain_mask] - all_rate_act[rain_mask])))
    print(confusion_matrix(all_actuals, (all_probs>=0.5).astype(int)))

# Model Validation
model_eval(model_1,loader1)
model_eval(model_3,loader3)
model_eval(model_5,loader5)  

# Save Models
torch.save(model_1.state_dict(), 'model_1.pt')
torch.save(model_3.state_dict(), 'model_3.pt')
torch.save(model_5.state_dict(), 'model_5.pt')

# Save Scalers
joblib.dump(scaler_r,'sg.joblib')
joblib.dump(scaler_c,'sh.joblib')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


