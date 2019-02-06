#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:20:40 2019

@author: wsw
"""


# using sine predict cosine with LSTM 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np




class Net(nn.Module):
  
  def __init__(self):
    super(Net,self).__init__()
    self.lstm = nn.LSTM(
                        input_size=1,
                        hidden_size=32,
                        num_layers=1,
                        batch_first=True                  
        )
    self.rnn = nn.RNN(
                        input_size=1,
                        hidden_size=64,
                        num_layers=1,
                        batch_first=True
        )
    self.predict = nn.Linear(64,1)
    
  def forward(self,xs,h_state):
    r_out, h_state = self.rnn(xs,h_state)
    
    '''
    outs = []
    for time_step in range(r_out.size(1)):
      outs.append(self.predict(r_out[:,time_step,:]))
    # predicts concat in time_step direction
    # B*time_step*Out
    outs = torch.stack(outs,dim=1)
    '''
    outs = self.predict(r_out)
    return outs
    

def train():
  epochs = 100
  time_step = 10
  # define model
  # define initial hidden_state = None
  h_state = None
  rnn = Net().cuda()
  # define loss function
  loss_func = nn.MSELoss().cuda()
  optimizer = optim.Adam(rnn.parameters(),lr=1e-2)
  
  plt.figure(figsize=(24,5))
  plt.ion()
  for epoch in range(epochs):
    start,end = epoch*np.pi,(epoch+2)*np.pi
    # generate a sequential
    seqs = torch.linspace(start,end,time_step)
    # make data
    # input shape NxTime_stepxInput_size
    xs = torch.sin(seqs[np.newaxis,:,np.newaxis]).cuda()
    # print(xs.requires_grad,xs.size())
    ys = torch.cos(seqs[np.newaxis,:,np.newaxis]).cuda()
    # predicts
    predicts = rnn(xs,h_state)
    # print(predicts.requires_grad)
    # compute loss
    loss = loss_func(predicts,ys)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print('Epoch:[{:03d}]-Loss:{:.3f}'.format(epoch+1,loss.item()))
    # plot figure
    
    plt.plot(seqs.numpy(),ys.squeeze().cpu().numpy(),'r-')
    plt.plot(seqs.numpy(),predicts.squeeze().detach().cpu().numpy(),'g-')

    plt.draw()
    plt.pause(0.05)
    
  plt.ioff()
  plt.show()
    
    
    
if __name__ == '__main__':
  train()
  
  
  
      
      




