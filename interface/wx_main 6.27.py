#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:18:32 2019

@author: wangshanmin
"""
#
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
import os
from PIL import Image, ImageDraw, ImageFont
import sklearn

class interface(wx.Frame):
    def __init__(self,parent, id, model):
        wx.Frame.__init__(self,parent = None, id = -1, title = '人脸识别系统',size = (1280, 960))
        self.interface_img = 'interface.jpeg'
        self.path = 'img_save/'
        self.initpos = 200
        self.minpane = 400
        self.interface_size = (600, 480)
        self.themeColor = '#0a74f7'
        self.interface_pos = (90,100)
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

      self.window_split_left.SplitVertically(self.left,self.middle)###初始位置

      self.window_split_left.SetSashGravity(0.2)

      self.window_split_right = wx.SplitterWindow(self)      

##      
      self.right = wx.Panel(parent = self.window_split_right, style = wx.SUNKEN_BORDER)

      self.middle_right = wx.Panel(parent = self.window_split_right, style = wx.SUNKEN_BORDER)
      self.middle_right.SetBackgroundColour(self.colour) 

      self.window_split_right.SplitVertically(self.middle_right,self.right)###初始位置
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
#      self.Bind(wx.EVT_BUTTON, self.verification, match_Button)
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
      self.interface_button = wx.BitmapButton(self.middle, -1, interface_img, pos=self.interface_pos, size=self.interface_size)
    
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
      self.Bind(wx.EVT_BUTTON, self.select_Button, select_Button)
      
    def takephoto(self, event):
        _thread.start_new_thread(self.takePhoto,(event,))

        
    def takePhoto(self,event):
        cap = cv2.VideoCapture(0)
        try:
            self.accountLabel.Destroy()
        except:
            print()
        while(True):
          flag, im_rd = cap.read()
          cv2.imshow('img', im_rd)
            #####
               ####
          if cv2.waitKey(1) & 0xFF == ord('q'):
                break

          # try:
          #       im_rd = cv2.resize(im_rd, self.interface_size)
          #
          #       im_rd = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
          #       pic = wx.Bitmap.FromBuffer(self.interface_size[0], self.interface_size[1], im_rd)
          #       self.interface_button.SetBitmap(pic)
          # except:
          #     print()
        cap.release()
        _thread.exit()
        cv2.destroyAllWindows()
        self.save_id(im_rd)



    def save_id(self, image_row):
      h, w, c = image_row.shape
      image = cv2.resize(image_row, (112, 112))
      account = self.accountInput.GetValue()
#        assert(account == None)
      font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)
      self.path_id = os.path.join(self.path, account)
      if not os.path.exists(self.path_id):
        os.makedirs(self.path_id)
      file = os.listdir(self.path_id)
      count = len(file) + 1
      self.save_name = self.path_id + '/' + str(count) + '.jpeg'
      try:
        aligned,bbox = self.model.get_input(image)

        if len(bbox) > 1:###检测到多个人脸
            self.accountLabel = wx.StaticText(self.middle, -1, u'检测到多个人脸', pos=(480, 780), size=(100, 100))
            self.accountLabel.SetForegroundColour(self.themeColor)
            self.accountLabel.SetFont(font)
        else:###检测到一个人脸
            self.img = aligned[0]
            box = bbox[0]
            cv2.rectangle(image_row, (int(box[0]* w / 112), int(box[1] * h / 112)), (int(box[2] * w / 112), int(box[3] * h / 112)), (55, 255, 155), 2)


            image = cv2.resize(image_row,self.interface_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.save_name, image)
            wxbmp = wx.BitmapFromBuffer(self.interface_size[0], self.interface_size[1], image)
            wx.BitmapButton(self.middle, -1 , wxbmp, pos=self.interface_pos, size=self.interface_size)

            ##保存特征
            feature = self.model.get_feature(self.img)
            fid = open(self.id_embedding, 'a')
            for emb in feature:
                fid.write(str(emb) + ' ')
            fid.write(account)
            fid.write('\n')
            fid.close()
      except:  ####没有检测到人脸
            self.accountLabel = wx.StaticText(self.middle, -1, u'没有检测到人脸', pos=(480, 780), size=(100, 100))
            self.accountLabel.SetForegroundColour(self.themeColor)
            self.accountLabel.SetFont(font)
      # _thread.exit()

        
    def select_Button(self, event):
      wildcard = 'All files(*.*)|*.*'
      dlg = wx.FileDialog(None,'select',os.getcwd(),'',wildcard)
#      dlg = wx.DirDialog(self,u"选择文件夹",style=wx.DD_DEFAULT_STYLE)
      if dlg.ShowModal() == wx.ID_OK:
        file = dlg.GetPath()
        image = cv2.imread(file)
        self.save_id(image)
        

      
        
      
    def MakeSure(self, event):###１．账户名不重复，２　保存embedding  3跳转界面
      exist = 0
      account = self.accountInput.GetValue()
      try:
        self.accountLabel.Destroy()
        self.accountInput.Clear()
      except:
        print()
      if not os.path.exists(self.id_file):
          os.mknod(self.id_file)
      fid = open(self.id_file,'r')
      names = fid.readlines()
      fid.close()
      font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)
      # accountLabel = wx.StaticText(self.middle, -1, u'帐号已经存在', pos=(450, 700), size  = (100, 100))
      if names is not None:
        for name in names:
          name = name.strip()
          if name == account:
            exist = 1###当前用户名已经存在
      if exist == 0:
        # accountLabel.SetForegroundColour(self.colour)
        fid = open(self.id_file,'a')
        fid.write(account + '\n')
        fid.close()
      else:
        self.accountLabel = wx.StaticText(self.middle, -1, u'帐号已经存在', pos=(450, 700), size  = (100, 100))
        self.accountLabel.SetFont(font)
      interface_img = wx.Image(load_image(self.interface_img), wx.BITMAP_TYPE_ANY).ConvertToBitmap()
      wx.BitmapButton(self.middle, -1, interface_img, pos=self.interface_pos, size=self.interface_size)

      
      
    def destroy(self, event):
        try:
            self.accountLabel.Destroy()
            self.accountInput.Clear()

        except:
            print()

          
    def rt_recognition(self, event):
       fid = open(self.id_embedding,'r')
       embeddings  = fid.readlines()
       id = []
       embed = []
       for emb in embeddings:
           emb = emb.strip()
           emb = emb.split(' ')
           id.append(emb.pop(-1))
           embed.append(list(map(float,emb)))
       fid.close()
       cap = cv2.VideoCapture(0)
       count = 0
       while(cap.isOpened()):
           flag, im_rd = cap.read()
           img_PIL = Image.fromarray(cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB))
           draw = ImageDraw.Draw(img_PIL)
           font = ImageFont.truetype('NotoSansCJK-Black.ttc', 20, encoding="utf-8")
           h,w, c = im_rd.shape
           try:
                img  = cv2.resize(im_rd,(112,112))
                aligned, bbox  = self.model.get_input(img)
                for i in range(len(aligned)):
                    img_align = aligned[i]
                    box = bbox[i]
                    feature = self.model.get_feature(img_align)

                    ###using MSE distance
                    dis_list = np.mean(np.square(np.subtract(feature, embed)), axis = 1)
                    min_dis = np.argmin(dis_list)
                    if dis_list[min_dis] < 1.27:
                        name = id[min_dis]
                    else:
                        name = 'unknown'

                    bbox_new = [box[0] * w / 112, box[1] * h / 112, box[2] * w / 112, box[3] * h / 112]
                    # cv2.rectangle(im_rd, (int(bbox_new[0]),int(bbox_new[1])),(int(bbox_new[2]), int(bbox_new[3])), (0,0,255), 2)
                    draw.rectangle((int(bbox_new[0]), int(bbox_new[1]), int(bbox_new[2]), int(bbox_new[3])), None,
                                   (0, 0, 255), 2)
                    draw.text((int(bbox_new[0]), int(bbox_new[1]-40)), name, (55, 255, 155), font= font)
                im_changed = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
                cv2.imshow('img',im_changed)
                if count % 20 == 0:
                    self.similar_pthotos(im_rd, id, dis_list, 1)
           except:
                print('No face detected!!!')
           count += 1
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break

       cap.release()
       cv2.destroyAllWindows()


    def similar_pthotos(self, im_rd, id_list, dis_list, target_num):
        for i in range(target_num):
            target_location = np.argmin(dis_list)
            id = id_list[target_location]
            dis_list = np.delete(dis_list, target_location)
            id_list = np.delete(id_list, target_location)
            target_id = os.path.join(self.path, r'%s'%id)
            similar_img = os.path.join(target_id,random.choice(os.listdir(target_id)))

            ###show 采集到的图片
            # im_rd = cv2.resize(im_rd, (100,100))
            # im_rd = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
            # wxbmp_row = wx.BitmapFromBuffer(100,100, im_rd)
            # wx.BitmapButton(self.middle_right, -1, wxbmp_row, pos=(20, 50), size=(100, 100))
            #
            img_similar = cv2.resize(cv2.imread(similar_img),(100,100))
            # img_similar = cv2.cvtColor(img_similar, cv2.COLOR_BGR2RGB)
            wxbmp = wx.BitmapFromBuffer(100, 100, img_similar)
            wx.BitmapButton(self.middle_right, -1, wxbmp, pos=(180, 50), size=(100, 100))









      
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
    parser.add_argument('--model', default='/home/wangshanmin/academy/insightface/models/model,00', help='path to load model.')
    parser.add_argument('--ga-model', default='/home/wangshanmin/academy/insightface/models/model,00', help='path to load model.')
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