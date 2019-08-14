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
import random
import numpy as np
import _thread
from PIL import Image, ImageDraw, ImageFont

class interface(wx.Frame):
    def __init__(self,parent, id, model):
        wx.Frame.__init__(self,parent = None, id = -1, title = '人脸识别系统',size = (1280, 960))
        self.interface_img = 'interface.jpeg'
        self.path = 'img_save/'
        self.initpos = 200
        self.minpane = 400
        self.interface_size = (600, 480)
        self.themeColor = '#0a74f7'
        self.interface_pos = (100,100)
        self.id_file = 'id_file.txt'
        self.id_embedding = 'id_embedding.txt'
        self.model =  model
        self.colour = (148,175, 255)
        self.InitUI()

        

      
    def InitUI(self):
    
      self.window_split_left = wx.SplitterWindow(self)

#      
      self.left = wx.Panel(parent = self.window_split_left, style = wx.SUNKEN_BORDER)
      self.left.SetBackgroundColour(self.colour)  
      self.middle = wx.Panel(parent = self.window_split_left, style = wx.SUNKEN_BORDER)
      self.middle.SetBackgroundColour(self.colour) 
#      self.left.Hide()
#      self.middle.Hide()
      self.window_split_left.SplitVertically(self.left,self.middle)###初始位置
#      self.window_split_left.SetMinimumPaneSize(self.minpane) 
      self.window_split_left.SetSashGravity(0.2)
      
#      self.windows = self.window_split_left.GetWindow1()
      self.window_split_right = wx.SplitterWindow(self)      

##      
      self.right = wx.Panel(parent = self.window_split_right, style = wx.SUNKEN_BORDER)
#      self.right.SetBackgroundColour((25,25,112)) 
      self.middle_right = wx.Panel(parent = self.window_split_right, style = wx.SUNKEN_BORDER)
      self.middle_right.SetBackgroundColour(self.colour) 
#      self.right.Hide()
#      self.middle_.Hide()
      self.window_split_right.SplitVertically(self.middle_right,self.right)###初始位置
#      self.window_split_right.SetMinimumPaneSize(self.minpane) 
      self.window_split_right.SetSashGravity(1)
      self.window_split_right.Unsplit(self.right)
      

      sizer = wx.BoxSizer(wx.HORIZONTAL)
      sizer.Add(self.window_split_left, 3, flag=wx.EXPAND) #自动缩放
      sizer.Add(self.window_split_right, 1, flag=wx.EXPAND) #自动缩放
      self.SetSizer(sizer)
    
#      
#      self.right = wx.Panel(parent = swindow, style = wx.SUNKEN_BORDER)
#      self.right .SetBackgroundColour('white')
#      self.middle_r = wx.Panel(parent = swindow, style = wx.SUNKEN_BORDER)
#      self.middle_r.SetBackgroundColour('white')
#      self.right.Hide()
#      self.middle_l.Hide()
#      swindow.SplitVertically(self.middle_r,self.right,self.initpos+400 )###初始位置
#      swindow.SetMinimumPaneSize(self.minpane) 
      
      register_Button = wx.Button(self.left, -1, u'注册', pos=(40,50), size=(100, 40))
#      register_Button.SetForegroundColour('white')
      register_Button.SetBackgroundColour(self.colour)
      self.Bind(wx.EVT_BUTTON, self.register, register_Button)
#        
      match_Button = wx.Button(self.left, -1, u'人脸匹配', pos=(40,130), size=(100, 40))
#      match_Button.SetForegroundColour('white')
      match_Button.SetBackgroundColour(self.colour)
#      self.Bind(wx.EVT_BUTTON, self.takePhoto, idButton)
#        
      recognition_Button = wx.Button(self.left, -1, u'实时识别', pos=(40,210), size=(100, 40))
#      recognition_Button.SetForegroundColour('white')
      recognition_Button.SetBackgroundColour(self.colour)  
      self.Bind(wx.EVT_BUTTON, self.rt_recognition, recognition_Button)
      
      
#      start_Button = wx.Button(self.middle,-1, u'开始', pos = (160, 600), size=(60, 40))
#      start_Button.SetBackgroundColour((25,25,112))
##      self.Bind(wx.EVT_BUTTON, self.show_video, start_Button)
#      
#      end_Button = wx.Button(self.middle,-1, u'结束', pos = (450, 600), size=(60, 40))
#      end_Button.SetBackgroundColour((25,25,112))
##      self.Bind(wx.EVT_BUTTON, self.show_video, end_Button)
      
      
          
      
      #打开界面显示封面图像
      image = cv2.imread(self.interface_img)
      image = cv2.resize(image, self.interface_size)
      cv2.imwrite(self.interface_img, image)
      interface_img = wx.Image(load_image(self.interface_img), wx.BITMAP_TYPE_ANY).ConvertToBitmap()
      wx.StaticBitmap(self.middle, -1, interface_img, pos=self.interface_pos, size=self.interface_size)
    
    def register(self,event):
      ####弹出界面
#      frame_register = login(None, -1, self.model)
#      frame_register.row_register()
#      frame_register.Center()
#      frame_register.Show(True)
      #＃#直接在中间界面显示
      font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)
      accountLabel = wx.StaticText(self.middle, -1, u'姓   名', pos=(180,710), size  = (100, 100))
      accountLabel.SetForegroundColour('white' )
      accountLabel.SetFont(font)
        
      self.accountInput = wx.TextCtrl(self.middle, -1, u'', pos=(300, 700), size=(150, -1))
      self.accountInput.SetForegroundColour('gray')
      self.accountInput.SetFont(font)
      
      take_Button = wx.Button(self.middle, -1, u'拍   照', pos=(180, 780), size=(100, 30))
      take_Button.SetForegroundColour('white')
      take_Button.SetBackgroundColour(self.colour)
      self.Bind(wx.EVT_BUTTON, self.takephoto, take_Button)
      
      take_Button = wx.Button(self.middle, -1, u'确定', pos=(180, 850), size=(100, 30))
      take_Button.SetForegroundColour('white')
      take_Button.SetBackgroundColour(self.colour)
      self.Bind(wx.EVT_BUTTON, self.MakeSure, take_Button)
      
      take_Button = wx.Button(self.middle, -1, u'取消', pos=(350, 850), size=(100, 30))
      take_Button.SetForegroundColour('white')
      take_Button.SetBackgroundColour(self.colour)
      self.Bind(wx.EVT_BUTTON, self.destroy, take_Button)
      
      select_Button = wx.Button(self.middle, -1, u'从文件夹选择', pos=(350, 780), size=(100, 30))
      select_Button.SetForegroundColour('white')
      select_Button.SetBackgroundColour(self.colour)
#      self.Bind(wx.EVT_BUTTON, self.takephoto, idButton)
      
    def takephoto(self, event):
#      _thread.start_new_thread(self.takePhoto(),(2,))
        self.takePhoto()
        
    def takePhoto(self):
        cap = cv2.VideoCapture(0)
        account = self.accountInput.GetValue()
#        assert(account == None)
        self.name = self.path + account+'.jpeg'
#        interface_img = wx.Image(load_image(self.interface_img), wx.BITMAP_TYPE_ANY).ConvertToBitmap()
#        wx.StaticBitmap(self.middle, -1, interfqace_img, pos=(20, 20), size=(720, 540))
        while(cap.isOpened()):
          flag, im_rd = cap.read()
          cv2.imshow('img', im_rd)
          h,w,c = im_rd.shape
  
          if cv2.waitKey(1) == ord('q'):
            img = cv2.resize(im_rd, (112, 112))
#            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
              aligned,bbox = self.model.get_input(img)
              self.img = aligned[0]
              box = bbox[0]
              cv2.rectangle(img, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])),(55, 255, 155),2)
              image = cv2.resize(img,self.interface_size)
              cv2.imwrite(self.name, image)
              wxbmp = wx.BitmapFromBuffer(600, 480, image)
              wx.StaticBitmap(self.middle, -1 , wxbmp, pos=self.interface_pos, size=self.interface_size)
            except:
              font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)
              accountLabel = wx.StaticText(self.middle, -1, u'没有检测到人脸', pos=(480, 780), size  = (100, 100))
              accountLabel.SetForegroundColour(self.themeColor)
              accountLabel.SetFont(font)
            break
        cap.release()
        cv2.destroyAllWindows()
        embedding = self.model.get_feature(self.img)
        fid = open(self.id_embedding,'a')
        fid.write(account)
        for emb in embedding:
          fid.write(' '+str(emb))
        fid.write('\n')
        fid.close()
#      self.Destroy()
        

        
      
    def MakeSure(self, event):###１．账户名不重复，２　保存embedding  3跳转界面
      exist = 0
      account = self.accountInput.GetValue()
      fid = open(self.id_file,'r')
      names = fid.readlines()
      fid.close()
      if names is not None:
        for name in names:
          name = name.strip()
          if name == account:
            font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)
            accountLabel = wx.StaticText(self.middle, -1, u'帐号已经存在', pos=(450, 700), size  = (100, 100))
            accountLabel.SetForegroundColour(self.themeColor)
            accountLabel.SetFont(font)
            exist = 1
      if exist == 0:
        fid = open(self.id_file,'a')
        fid.write(account + '\n')
        fid.close()

      
      
    def destroy(self):
      self.Destroy()
      image = cv2.imread(self.interface_img)
      image = cv2.resize(image, (720,540))
      cv2.imwrite(self.interface_img, image)
      interface_img = wx.Image(load_image(self.interface_img), wx.BITMAP_TYPE_ANY).ConvertToBitmap()
      wx.StaticBitmap(self.middle, -1, interface_img,pos=self.interface_pos, size=self.interface_size)
        
          
        
    def rt_recognition(self, event):
        id_embd = {}
        fid = open(self.id_embedding,'r')
        embeddings  = fid.readlines()
        for emb in embeddings:
            emb = emb.strip()
            emb = emb.split(' ')
            id_embd[emb[0]] = list(map(float,emb[1:]))
        id_infor = []
        fid.close()
        cap = cv2.VideoCapture(0)
        while(cap.isOpened()):
          flag, im_rd = cap.read()
          img = cv2.resize(im_rd, (112, 112))
          try:
            dis = []
            name_list = []
            aligned,bbox_total = self.model.get_input(img)

            for i in range(len(aligned)):
              feature = self.model.get_feature(aligned[i])
              for id, embd in id_embd.items():
#                dis.append(np.dot(embd, feature)/(np.linalg.norm(embd) * np.linalg.norm(feature)))
                dis.append(np.sum(np.square(embd - feature)))
                id_infor.append(id)
#              index = np.argmax(dis)              
#              if dis[index] >0.3:
#                name = id_infor[index]
              index = np.argmin(dis)
              if dis[index] < 1.27:
                  name = id_infor[index]
              else:
                  name = 'unknown'
                
              name_list.append(name)
            img_PIL = Image.fromarray(cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_PIL)
            for m in range(len(name_list)): 
              name = name_list[m]
              font = ImageFont.truetype('NotoSansCJK-Black.ttc', 20, encoding="utf-8")
              h,w,c = im_rd.shape
              bbox = bbox_total[m]
              bbox_new = [bbox[0] * w / 112, bbox[1] *h /112, bbox[2]*w/112, bbox[3]*h/112 ]
              draw.rectangle((int(bbox_new[0]),int(bbox_new[1]),int(bbox_new[2]), int(bbox_new[3])), None, (0,0,255), 2)
              draw.text((int(bbox_new[0]), int(bbox_new[1]-40)), name, (55, 255, 155), font= font)
            im_changed = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
            cv2.imshow('img', im_changed)
          except:
            print('No face is detected!')
          if cv2.waitKey(1) == ord('q'):
            break
        cv2.destroyAllWindows()
      
#class login(wx.Frame):
#    def __init__(self,parent, id, model):
#      wx.Frame.__init__(self,parent, id, '用户注册',pos =(600,600), size = (400,400), style = wx.DEFAULT_FRAME_STYLE^(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)|wx.STAY_ON_TOP)
#      self.path = 'img_save/'
#      self.id_file = 'id_file.txt'
#      self.id_embedding = 'id_embedding.txt'
#      self.model = model
#
#      self.panel = wx.Panel(self,-1) 
#      
#      self.themeColor = '#0a74f7'
#      
#    def row_register(self):
#
#      font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)
#      self.panel.SetBackgroundColour((25,25,112))
#      accountLabel = wx.StaticText(self.panel, -1, u'姓   名', pos=(50,33), size  = (100, 100))
#      accountLabel.SetForegroundColour('white' )
#      accountLabel.SetFont(font)
#        
#      self.accountInput = wx.TextCtrl(self.panel, -1, u'', pos=(150, 30), size=(150, -1))
#      self.accountInput.SetForegroundColour('gray')
#      self.accountInput.SetFont(font)
#        
#      idButton = wx.Button(self.panel, -1, u'拍   照', pos=(30, 100), size=(100, 40))
#      idButton.SetForegroundColour('white')
#      idButton.SetBackgroundColour(self.themeColor)
#      self.Bind(wx.EVT_BUTTON, self.takephoto, idButton)  
#      
#      idButton = wx.Button(self.panel, -1, u'从文件夹选择', pos=(30, 180), size=(100, 40))
#      idButton.SetForegroundColour('white')
#      idButton.SetBackgroundColour(self.themeColor)
##      self.Bind(wx.EVT_BUTTON, self.takePhoto, idButton)  
#      
#      ##照片选择　　拍照或从文件夹选择
##      take_photo = wx.RadioButton(self.panel, -1, '拍照',pos=(80, 100), size=(65, 25))
##      take_photo.SetForegroundColour('white')
##      take_photo.SetBackgroundColour(self.themeColor)
##      self.Bind(wx.EVT_BUTTON, self.takePhoto, take_photo)
##      select_photo = wx.RadioButton(self.panel, -1, '从文件夹选择',pos=(80, 140), size=(125, 25))
##      select_photo.SetForegroundColour('white')
##      select_photo.SetBackgroundColour(self.themeColor)
##      self.Bind(wx.EVT_BUTTON, self.selectphoto, select_photo)
#
#      idButton = wx.Button(self.panel, -1, u'确　定', pos=(80, 330), size=(80, 30))
#      idButton.SetForegroundColour('white')
#      idButton.SetBackgroundColour(self.themeColor)
#      self.Bind(wx.EVT_BUTTON, self.MakeSure, idButton) 
#      
#      idButton = wx.Button(self.panel, -1, u'取消', pos=(280, 330), size=(80, 30))
#      idButton.SetForegroundColour('white')
#      idButton.SetBackgroundColour(self.themeColor)
#      self.Bind(wx.EVT_BUTTON, self.destroy, idButton)
#    
#    def takephoto(self,event):
#      self.takePhoto()
#      
#    def takePhoto(self):
#        cap = cv2.VideoCapture(0)
#        
#        account = self.accountInput.GetValue()
##        assert(account == None)
#        self.name = self.path + account+'.jpeg'
#        while(cap.isOpened()):
#          flag, im_rd = cap.read()
#          h,w,c = im_rd.shape
#          cv2.imshow('img', im_rd)
##          try:
#          c = cv2.waitKey(1)
#          if c == ord('s'):  
#              face_sum, bbox_sum = self.model.get_input(cv2.resize(im_rd, (112, 112)))
#              box = bbox_sum[0]
#              cv2.rectangle(im_rd, (int(box[0]/112*w), int(box[1]/112*h)),(int(box[2]/112*w), int(box[3]/112*h)),(55, 255, 155),2)
##            img_save = im_rd[int(box[1]/112*h):int(box[3]/112*h), int(box[0]/112*w):int(box[2]/112*w)]
#          
#              cv2.imwrite(self.name, cv2.resize(im_rd,(200,150)))
#              register_img = wx.Image(self.name ,wx.BITMAP_TYPE_JPEG).ConvertToBitmap()
#              wx.StaticBitmap(self.panel,-1,register_img, pos = (150,100), size = (200,150))
#          if c == ord('q'):
#              break
##          except:
##            print('No face dected') 
#        cap.release()
#        cv2.destroyAllWindows()
#        
#    def MakeSure(self, event):###１．账户名不重复，２　保存embedding  3跳转界面
#      account = self.accountInput.GetValue()
#      fid = open(self.id_file,'r')
#      names = fid.readlines()
#      fid.close()
#      if names is not None:
#        for name in names:
#          name = name.strip()
#          if name == account:
#            font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)
#            accountLabel = wx.StaticText(self.panel, -1, u'帐号已经存在', pos=(330, 50), size  = (100, 100))
#            accountLabel.SetForegroundColour(self.themeColor)
#            accountLabel.SetFont(font)
#            return 0
#      fid = open(self.id_file,'a')
#      fid.write(account + '\n')
#      fid.close()
#      fid = open(self.id_embedding,'a')
#      images = cv2.resize(cv2.imread(self.name),(112,112))
#      aligned_sum, bbox_sum = self.model.get_input(images)
#      embedding = self.model.get_feature(aligned_sum[0])
#      fid.write(account)
#      for emb in embedding:
#        fid.write(' '+str(emb))
#      fid.write('\n')
#      fid.close()
##      self.Destroy()
#      
#    def destroy(self):
#      self.Destroy()
      

      
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
                                            
    Frame = interface(parent = None, id = -1, model = self.model)
        
    Frame.Show()
    
if __name__ == '__main__':
  App = App()
  App.InitUI()
  App.MainLoop()