#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:18:32 2019

@author: wangshanmin
"""
import sys
sys.path.append('../deploy')
import wx
import cv2
import argparse
import face_model
from utils import load_image

class interface(wx.Frame):
    def __init__(self,parent, id):
        wx.Frame.__init__(self,parent = None, id = -1, title = '人脸识别系统',size = (1280, 960))
        self.interface_img = 'interface.jpeg'
        self.initpos = 200
        self.minpane = 200
        self.InitUI()

        

      
    def InitUI(self):
      
      swindow = wx.SplitterWindow(self,-1)
      self.left = wx.Panel(parent=swindow, style = wx.SUNKEN_BORDER)
      self.right = wx.Panel(parent=swindow, style = wx.SUNKEN_BORDER)
      self.left.SetBackgroundColour('white')
      self.right.SetBackgroundColour('white')
      self.left.Hide()
      self.right.Hide()
      swindow.SplitVertically(self.left,self.right,self.initpos)###初始位置
      swindow.SetMinimumPaneSize(self.minpane)
        
      
      register_Button = wx.Button(self.left, -1, u'注册', pos=(40,50), size=(100, 40))
      register_Button.SetForegroundColour('white')
      register_Button.SetBackgroundColour('#0a74f7')
#        self.Bind(wx.EVT_BUTTON, self.takePhoto, idButton)
#        
      match_Button = wx.Button(self.left, -1, u'人脸匹配', pos=(40,130), size=(100, 40))
      match_Button.SetForegroundColour('white')
      match_Button.SetBackgroundColour('#0a74f7')
#        self.Bind(wx.EVT_BUTTON, self.takePhoto, idButton)
#        
      recognition_Button = wx.Button(self.left, -1, u'实时识别', pos=(40,210), size=(100, 40))
      recognition_Button.SetForegroundColour('white')
      recognition_Button.SetBackgroundColour('#0a74f7')  
      
        
      
class login(wx.Frame):
    def __init__(self,parent, id, model):
      wx.Frame.__init__(self,parent, id, '用户注册',pos =(600,600), size = (400,400), style = wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)|wx.STAY_ON_TOP)
      self.path = 'img_save/'
      self.id_file = 'id_file.txt'
      self.id_embedding = 'id_embedding.txt'
      self.model = model

      self.panel = wx.Panel(self,-1) 
      
      self.themeColor = '#0a74f7'

      font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)

      accountLabel = wx.StaticText(self.panel, -1, u'姓   名', pos=(50,33), size  = (100, 100))
      accountLabel.SetForegroundColour(self.themeColor)
      accountLabel.SetFont(font)
        
      self.accountInput = wx.TextCtrl(self.panel, -1, u'', pos=(150, 30), size=(150, -1))
      self.accountInput.SetForegroundColour('gray')
      self.accountInput.SetFont(font)
        
      idButton = wx.Button(self.panel, -1, u'拍   照', pos=(30, 100), size=(100, 40))
      idButton.SetForegroundColour('white')
      idButton.SetBackgroundColour(self.themeColor)
      self.Bind(wx.EVT_BUTTON, self.takePhoto, idButton)  
      
      idButton = wx.Button(self.panel, -1, u'从文件夹选择', pos=(30, 180), size=(100, 40))
      idButton.SetForegroundColour('white')
      idButton.SetBackgroundColour(self.themeColor)
#      self.Bind(wx.EVT_BUTTON, self.takePhoto, idButton)  
      
      ##照片选择　　拍照或从文件夹选择
#      take_photo = wx.RadioButton(self.panel, -1, '拍照',pos=(80, 100), size=(65, 25))
#      take_photo.SetForegroundColour('white')
#      take_photo.SetBackgroundColour(self.themeColor)
#      self.Bind(wx.EVT_BUTTON, self.takePhoto, take_photo)
#      select_photo = wx.RadioButton(self.panel, -1, '从文件夹选择',pos=(80, 140), size=(125, 25))
#      select_photo.SetForegroundColour('white')
#      select_photo.SetBackgroundColour(self.themeColor)
#      self.Bind(wx.EVT_BUTTON, self.selectphoto, select_photo)

      idButton = wx.Button(self.panel, -1, u'确　定', pos=(80, 330), size=(80, 30))
      idButton.SetForegroundColour('white')
      idButton.SetBackgroundColour(self.themeColor)
      self.Bind(wx.EVT_BUTTON, self.MakeSure, idButton) 
      
      idButton = wx.Button(self.panel, -1, u'取消', pos=(280, 330), size=(80, 30))
      idButton.SetForegroundColour('white')
      idButton.SetBackgroundColour(self.themeColor)
      self.Bind(wx.EVT_BUTTON, self.destroy, idButton)
      
    def takePhoto(self, event):
        cap = cv2.VideoCapture(0)
        
        account = self.accountInput.GetValue()
#        assert(account == None)
        self.name = self.path + account+'.jpeg'
        while(cap.isOpened()):
          flag, im_rd = cap.read()
          h,w,c = im_rd.shape
          cv2.imshow('img', im_rd)
#          try:
          c = cv2.waitKey(1)
          if c == ord('s'):  
              face_sum, bbox_sum = self.model.get_input(cv2.resize(im_rd, (112, 112)))
              box = bbox_sum[0]
              cv2.rectangle(im_rd, (int(box[0]/112*w), int(box[1]/112*h)),(int(box[2]/112*w), int(box[3]/112*h)),(55, 255, 155),2)
#            img_save = im_rd[int(box[1]/112*h):int(box[3]/112*h), int(box[0]/112*w):int(box[2]/112*w)]
          
              cv2.imwrite(self.name, cv2.resize(im_rd,(200,150)))
              register_img = wx.Image(self.name ,wx.BITMAP_TYPE_JPEG).ConvertToBitmap()
              wx.StaticBitmap(self.panel,-1,register_img, pos = (150,100), size = (200,150))
          if c == ord('q'):
              break
#          except:
#            print('No face dected') 
        cap.release()
        cv2.destroyAllWindows()
        
    def MakeSure(self, event):###１．账户名不重复，２　保存embedding  3跳转界面
      account = self.accountInput.GetValue()
      fid = open(self.id_file,'r')
      names = fid.readlines()
      fid.close()
      if names is not None:
        for name in names:
          name = name.strip()
          if name == account:
            font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)
            accountLabel = wx.StaticText(self.panel, -1, u'帐号已经存在', pos=(330, 50), size  = (100, 100))
            accountLabel.SetForegroundColour(self.themeColor)
            accountLabel.SetFont(font)
            return 0
      fid = open(self.id_file,'a')
      fid.write(account + '\n')
      fid.close()
      fid = open(self.id_embedding,'a')
      images = cv2.resize(cv2.imread(self.name),(112,112))
      aligned_sum, bbox_sum = self.model.get_input(images)
      embedding = self.model.get_feature(aligned_sum[0])
      fid.write(account)
      for emb in embedding:
        fid.write(' '+str(emb))
      fid.write('\n')
      fid.close()
      self.Destroy()
      
    def destroy(self):
      self.Destroy()
      
class App(wx.App):
  def __init__(self, redirect = True, filename = None):
    wx.App.__init__(self, redirect, filename)
    self.model = self.load_model()
    
  def load_model(self):
  
    parser = argparse.ArgumentParser(description='face model test')
# general
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='/home/wangshanmin/insightface/models/model,00', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    args = parser.parse_args()
    return face_model.FaceModel(args)
    
  def InitUI(self):
                                            
    Frame = interface(parent = None, id = -1)
        
    self.frame_register = login(None, -1, self.model)
    self.frame_register.Center()
    self.frame_register.Show(True)
    Frame.Show()
    
if __name__ == '__main__':
  App = App()
  App.InitUI()
  App.MainLoop()