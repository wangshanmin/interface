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
        self.colour = (148, 175, 255)
        self.InitUI()

    def InitUI(self):
        self.window_split_left = wx.SplitterWindow(self)

        #将一个window_split_left 切割成两个panel: self.left 和 self.middle
        self.left = wx.Panel(parent=self.window_split_left, style=wx.SUNKEN_BORDER)
        self.left.SetBackgroundColour(self.colour)
        self.middle = wx.Panel(parent=self.window_split_left, style=wx.SUNKEN_BORDER)
        self.middle.SetBackgroundColour(self.colour)

        self.window_split_left.SplitVertically(self.left, self.middle)  ###初始位置

        self.window_split_left.SetSashGravity(0.2)

        ###将window_split_right 分割成两个panel,self.right和self.middle_right
        self.window_split_right = wx.SplitterWindow(self)

        ##middle panel 再切割成middle_right 和
        self.right = wx.Panel(parent=self.window_split_right, style=wx.SUNKEN_BORDER)

        self.middle_right = wx.Panel(parent=self.window_split_right, style=wx.SUNKEN_BORDER)
        self.middle_right.SetBackgroundColour(self.colour)

        self.window_split_right.SplitVertically(self.middle_right, self.right)  ###初始位置
        self.window_split_right.SetSashGravity(1)
        ###self.right
        self.window_split_right.Unsplit(self.right)
        ###剩下的三个窗口水平排列
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.window_split_left, 3, flag=wx.EXPAND)  # 自动缩放
        sizer.Add(self.window_split_right, 1, flag=wx.EXPAND)  # 自动缩放
        self.SetSizer(sizer)

        # 打开界面显示封面图像
        image = cv2.imread(self.interface_img)
        image = cv2.resize(image, self.interface_size)
        cv2.imwrite(self.interface_img, image)
        interface_img = wx.Image(load_image(self.interface_img), wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.interface_button = wx.BitmapButton(self.middle, -1, interface_img, pos=self.interface_pos,
                                                size=self.interface_size)

        ###注册按钮
        self.register_Button = wx.Button(self.left, -1, u'注册', pos=(40, 50), size=(100, 40))
        self.register_Button.SetBackgroundColour(self.colour)
        self.Bind(wx.EVT_BUTTON, self.register, self.register_Button)
        #人脸匹配按钮
        self.match_Button = wx.Button(self.left, -1, u'人脸匹配', pos=(40, 130), size=(100, 40))
        self.match_Button.SetBackgroundColour(self.colour)
        ###实时识别按钮
        self.recognition_Button = wx.Button(self.left, -1, u'实时识别', pos=(40, 210), size=(100, 40))
        self.recognition_Button.SetBackgroundColour(self.colour)
        self.Bind(wx.EVT_BUTTON, self.rt_recognition, self.recognition_Button)

    ####注册界面
    def register(self, event):
        ####弹出界面
        #      frame_register = login(None, -1, self.model)
        #      frame_register.row_register()
        #      frame_register.Center()
        #      frame_register.Show(True)
        # ＃#直接在中间界面显示
        ##显示姓名
        font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)
        self.accountLabel = wx.StaticText(self.middle, -1, u'姓   名', pos=(180, 710), size=(100, 100))
        self.accountLabel.SetForegroundColour('white')
        self.accountLabel.SetFont(font)
        ##输入姓名文本框
        self.accountInput = wx.TextCtrl(self.middle, -1, u'', pos=(300, 700), size=(150, -1))
        self.accountInput.SetForegroundColour('gray')
        self.accountInput.SetFont(font)
        ###拍照
        self.take_Button = wx.Button(self.middle, -1, u'拍   照', pos=(180, 780), size=(100, 30))
        self.take_Button.SetForegroundColour('white')
        self.take_Button.SetBackgroundColour(self.colour)
        self.Bind(wx.EVT_BUTTON, self.takephoto, self.take_Button)
        ##确定
        self.sure_Button = wx.Button(self.middle, -1, u'确定', pos=(180, 850), size=(100, 30))
        self.sure_Button.SetForegroundColour('white')
        self.sure_Button.SetBackgroundColour(self.colour)
        self.Bind(wx.EVT_BUTTON, self.MakeSure, self.sure_Button)
        ###取消
        self.delay_Button = wx.Button(self.middle, -1, u'取消', pos=(350, 850), size=(100, 30))
        self.delay_Button.SetForegroundColour('white')
        self.delay_Button.SetBackgroundColour(self.colour)
        self.Bind(wx.EVT_BUTTON, self.destroy, self.delay_Button)
        ###从文件夹选择
        self.select_Button = wx.Button(self.middle, -1, u'从文件夹选择', pos=(350, 780), size=(100, 30))
        self.select_Button.SetForegroundColour('white')
        self.select_Button.SetBackgroundColour(self.colour)
        self.Bind(wx.EVT_BUTTON, self.select_Button, self.select_Button)

    def takephoto(self, event):
        _thread.start_new_thread(self.takePhoto, (event,))

    def takePhoto(self):
        cap = cv2.VideoCapture(0)
        try:
            self.accountLabel.Destroy()
        except:
            print()
        while (True):
            flag, im_rd = cap.read()
            cv2.imshow('img', im_rd)
            #####
            ####
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
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
            aligned, bbox = self.model.get_input(image)

            if len(bbox) > 1:  ###检测到多个人脸
                self.accountLabel = wx.StaticText(self.middle, -1, u'检测到多个人脸', pos=(480, 780), size=(100, 100))
                self.accountLabel.SetForegroundColour(self.themeColor)
                self.accountLabel.SetFont(font)
            else:  ###检测到一个人脸
                self.img = aligned[0]
                box = bbox[0]
                cv2.rectangle(image_row, (int(box[0] * w / 112), int(box[1] * h / 112)),
                              (int(box[2] * w / 112), int(box[3] * h / 112)), (55, 255, 155), 2)

                image = cv2.resize(image_row, self.interface_size)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(self.save_name, image)
                wxbmp = wx.BitmapFromBuffer(self.interface_size[0], self.interface_size[1], image)
                wx.BitmapButton(self.middle, -1, wxbmp, pos=self.interface_pos, size=self.interface_size)

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
        dlg = wx.FileDialog(None, 'select', os.getcwd(), '', wildcard)
        #      dlg = wx.DirDialog(self,u"选择文件夹",style=wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            file = dlg.GetPath()
            image = cv2.imread(file)
            self.save_id(image)

    def MakeSure(self, event):  ###１．账户名不重复，２　保存embedding  3跳转界面
        exist = 0
        account = self.accountInput.GetValue()
        try:
            self.accountLabel.Destroy()
            self.accountInput.Clear()
        except:
            print()
        if not os.path.exists(self.id_file):
            os.mknod(self.id_file)
        fid = open(self.id_file, 'r')
        names = fid.readlines()
        fid.close()
        font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)
        # accountLabel = wx.StaticText(self.middle, -1, u'帐号已经存在', pos=(450, 700), size  = (100, 100))
        if names is not None:
            for name in names:
                name = name.strip()
                if name == account:
                    exist = 1  ###当前用户名已经存在
        if exist == 0:
            # accountLabel.SetForegroundColour(self.colour)
            fid = open(self.id_file, 'a')
            fid.write(account + '\n')
            fid.close()
        else:
            self.accountLabel = wx.StaticText(self.middle, -1, u'帐号已经存在', pos=(450, 700), size=(100, 100))
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
        fid = open(self.id_embedding, 'r')
        embeddings = fid.readlines()
        id = []
        embed = []
        for emb in embeddings:
            emb = emb.strip()
            emb = emb.split(' ')
            id.append(emb.pop(-1))
            embed.append(list(map(float, emb)))
        fid.close()
        cap = cv2.VideoCapture(0)
        count = 0
        while (cap.isOpened()):
            flag, im_rd = cap.read()
            img_PIL = Image.fromarray(cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_PIL)
            font = ImageFont.truetype('NotoSansCJK-Black.ttc', 20, encoding="utf-8")
            h, w, c = im_rd.shape
            try:
                img = cv2.resize(im_rd, (112, 112))
                aligned, bbox = self.model.get_input(img)
                for i in range(len(aligned)):
                    img_align = aligned[i]
                    box = bbox[i]
                    feature = self.model.get_feature(img_align)

                    ###using MSE distance
                    dis_list = np.mean(np.square(np.subtract(feature, embed)), axis=1)
                    min_dis = np.argmin(dis_list)
                    if dis_list[min_dis] < 1.27:
                        name = id[min_dis]
                    else:
                        name = 'unknown'

                    bbox_new = [box[0] * w / 112, box[1] * h / 112, box[2] * w / 112, box[3] * h / 112]
                    # cv2.rectangle(im_rd, (int(bbox_new[0]),int(bbox_new[1])),(int(bbox_new[2]), int(bbox_new[3])), (0,0,255), 2)
                    draw.rectangle((int(bbox_new[0]), int(bbox_new[1]), int(bbox_new[2]), int(bbox_new[3])), None,
                                   (0, 0, 255), 2)
                    draw.text((int(bbox_new[0]), int(bbox_new[1] - 40)), name, (55, 255, 155), font=font)
                im_changed = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
                cv2.imshow('img', im_changed)
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
            target_id = os.path.join(self.path, r'%s' % id)
            similar_img = os.path.join(target_id, random.choice(os.listdir(target_id)))

            ###show 采集到的图片
            # im_rd = cv2.resize(im_rd, (100,100))
            # im_rd = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
            # wxbmp_row = wx.BitmapFromBuffer(100,100, im_rd)
            # wx.BitmapButton(self.middle_right, -1, wxbmp_row, pos=(20, 50), size=(100, 100))
            #
            img_similar = cv2.resize(cv2.imread(similar_img), (100, 100))
            # img_similar = cv2.cvtColor(img_similar, cv2.COLOR_BGR2RGB)
            wxbmp = wx.BitmapFromBuffer(100, 100, img_similar)
            wx.BitmapButton(self.middle_right, -1, wxbmp, pos=(180, 50), size=(100, 100))


class App(wx.App):
    def __init__(self, redirect=True, filename=None):
        wx.App.__init__(self, redirect, filename)
        self.model = self.load_model()

    def load_model(self):
        parser = argparse.ArgumentParser(description='face model test')
        # general
        parser.add_argument('--image-size', default='112,112', help='')
        parser.add_argument('--model', default='/home/wangshanmin/academy/insightface/models/model,00',
                            help='path to load model.')
        parser.add_argument('--ga-model', default='',
                            help='path to load model.')
        parser.add_argument('--gpu', default=0, type=int, help='gpu id')
        parser.add_argument('--det', default=0, type=int,
                            help='mtcnn option, 1 means using R+O, 0 means detect from begining')
        parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
        args = parser.parse_args()
        return face_model.FaceModel(args)

    def InitUI(self):
        Frame = interface(parent=None, id=-1, model=self.model)

        Frame.Show()


if __name__ == '__main__':
    App = App()
    App.InitUI()
    App.MainLoop()
