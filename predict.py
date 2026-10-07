#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code runs inference to predict rain rate in mm/hr and probability that it exceeds 1mm/hr.  It makes use
of GOES and HRRR data as a minute level time series.  Inputs and outputs are in parquet format.

To be used in conjunction with train3.py and preprocess3.py.  

@author: Safaan Sadiq
"""

import pandas as pd
import numpy as np
import pvlib as pv

import argparse
from preprocess3 import preprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 


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

def evaluate(df,c_cols,r_cols,cl_cols,m_cols):
    
    # Dimensions for initialization
    cl_dim=len(cl_cols)
    r_dim=len(r_cols)
    m_dim=len(m_cols)

    # Initialize Models
    model_1=DoubleEncoder(cl_dim, r_dim, m_dim)
    model_3=DoubleEncoder(cl_dim, r_dim, m_dim)
    model_5=DoubleEncoder(cl_dim, r_dim, m_dim)
    
    # Load trained models
    model_1.load_state_dict(torch.load('model_1.pt', map_location=device))
    model_3.load_state_dict(torch.load('model_3.pt', map_location=device))
    model_5.load_state_dict(torch.load('model_5.pt', map_location=device))
    
    # Load saved scalers
    scaler_r=joblib.load('sg.joblib')
    scaler_c=joblib.load('sh.joblib')
    
    # Scale Data
    C=torch.tensor(scaler_c.transform(df[cl_cols]), dtype=torch.float32)
    R=torch.tensor(scaler_r.transform(df[r_cols]), dtype=torch.float32)
    M=torch.tensor(df[m_cols].values, dtype=torch.float32)
    
    # Prepare Data into batches with DataLoader
    data=TensorDataset(C,R,M)
    loader=DataLoader(data,batch_size=128, shuffle=False)
    
    # Set models in eval mode
    model_1.eval()
    model_3.eval()
    model_5.eval()
    
    
    all_rates_1=[]
    all_rates_3=[]
    all_rates_5=[]
    
    all_probs_1=[]
    all_probs_3=[]
    all_probs_5=[]
    
    
    with torch.no_grad():
        for C, R, M in loader:
            C = C.to(device)
            R = R.to(device)
            M = M.to(device)
            
            # Run Models
            rate1, logit1 = model_1(C, R, M)
            rate3, logit3 = model_3(C, R, M)
            rate5, logit5 = model_5(C, R, M)
 
            # Append inferences
            all_rates_1.append(rate1.cpu().numpy().flatten())
            all_rates_3.append(rate3.cpu().numpy().flatten())
            all_rates_5.append(rate5.cpu().numpy().flatten())
            all_probs_1.append(torch.sigmoid(logit1).cpu().numpy().flatten())
            all_probs_3.append(torch.sigmoid(logit3).cpu().numpy().flatten())
            all_probs_5.append(torch.sigmoid(logit5).cpu().numpy().flatten())
            
    # Append data to data frame
    df['r1'] = np.concatenate(all_rates_1)
    df['p1'] = np.concatenate(all_probs_1)
    df['r3'] = np.concatenate(all_rates_3)
    df['p3'] = np.concatenate(all_probs_3)
    df['r5'] = np.concatenate(all_rates_5)
    df['p5'] = np.concatenate(all_probs_5)
    
    
    # Reshape into submission format
    base = df[['site_id', 'timestamp']].copy()

    df1 = base.copy(); df1['horizon_minute'] = 1; df1['rain_probability'] = df['p1'].values; df1['rain_rate_mm_hr'] = df['r1'].values
    df3 = base.copy(); df3['horizon_minute'] = 3; df3['rain_probability'] = df['p3'].values; df3['rain_rate_mm_hr'] = df['r3'].values
    df5 = base.copy(); df5['horizon_minute'] = 5; df5['rain_probability'] = df['p5'].values; df5['rain_rate_mm_hr'] = df['r5'].values

    result = pd.concat([df1, df3, df5]).sort_values(['site_id', 'timestamp', 'horizon_minute']).reset_index(drop=True)
    return result

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    
    df,c_cols, r_cols, cl_cols, m_cols=preprocess(args.input)
    
    df2=evaluate(df,c_cols,r_cols,cl_cols,m_cols)
    
    df2.to_parquet(args.output)
    