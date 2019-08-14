import sys
sys.path.append('../deploy')
import wx
import cv2
import argparse
import face_model
from utils import load_image
class interface(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self,parent = None, id = -1)
        self.path = 'img_save'
        self.interface_img = 'interface.jpeg'
        self.initpos = 200
        self.minpane = 200
        self.InitUI()
        self.model = self.load_model()
        
    def load_model(self):
  
        parser = argparse.ArgumentParser(description='face model test')
# general
        parser.add_argument('--image-size', default='112,112', help='')
        parser.add_argument('--model', default='/home/wangshanmin/academy/insightface/models/model,00', help='path to load model.')
        parser.add_argument('--ga-model', default='', help='path to load model.')
        parser.add_argument('--gpu', default=0, type=int, help='gpu id')
        parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
        parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
        args = parser.parse_args()
        return face_model.FaceModel(args)
      
    def InitUI(self):
     ####分割窗口
        Frame = interface(parent = None, id = -1, title = '人脸识别系统', size = (1280, 960)) 
        swindow = wx.SplitterWindow(self,-1)
        self.left = wx.Panel(parent=swindow, style = wx.SUNKEN_BORDER)
        self.right = wx.Panel(parent=swindow, style = wx.SUNKEN_BORDER)
        self.left.SetBackgroundColour('white')
        self.right.SetBackgroundColour('white')
        self.left.Hide()
        self.right.Hide()
        swindow.SplitVertically(self.left,self.right,self.initpos)###初始位置
        swindow.SetMinimumPaneSize(self.minpane)
        
        
        

      ##添加菜单
#        image = cv2.imread(self.interface_img)
#        image = cv2.resize(image, (720, 480))
#        cv2.imwrite(self.interface_img, image)
#        interface_img = wx.Image(self.interface_img ,wx.BITMAP_TYPE_JPEG).ConvertToBitmap()
#        wx.StaticBitmap(self.panel,-1,interface_img, pos = (250,250), size = (720, 480))
#        menuBar = wx.MenuBar()##定义一个菜单
#        menu = wx.Menu()##定义按钮
#        register = menu.Append(-1, '注册','register')
#        test = menu.Append(-1, '测试','Test')
#        menuBar.Append(menu, '&File')
#        self.SetMenuBar(menuBar)
#        self.Bind(wx.EVT_MENU, self.register, register)
##        self.Bind(wx.EVT_MENU, self.test, test)
#        self.Center()
#        self.Show()
      
      ####按钮方式
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
#        self.Bind(wx.EVT_BUTTON, self.takePhoto, idButton)
        
        
        self.frame_register = login(None, -1)
        self.frame_register.Center()
        self.frame_register.Show(True)
        
        Frame.Show()
        
    def OnQuit(self):
        self.Close()
        
        
class login(wx.Frame):
    def __init__(self,parent, id):
      wx.Frame.__init__(self,parent, id, '用户注册',pos =(400,400), size = (400,400), style  = wx.FRAME_FLOAT_ON_PARENT)
      panel = wx.Panel(self,-1)
        
    def register(self):
#        

        self.Center
        self.themeColor = '#0a74f7'

        font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)

        accountLabel = wx.StaticText(panel, -1, u'姓   名', pos=(400, 400), size  = (200, 100))
        accountLabel.SetForegroundColour(self.themeColor)
        accountLabel.SetFont(font)
        
        self.accountInput = wx.TextCtrl(self.right, -1, u'', pos=(500, 400), size=(150, -1))
        self.accountInput.SetForegroundColour('gray')
        self.accountInput.SetFont(font)
        
        idButton = wx.Button(self.right, -1, u'拍   照', pos=(400, 500), size=(50, 50))
        idButton.SetForegroundColour('white')
        idButton.SetBackgroundColour(self.themeColor)
        self.Bind(wx.EVT_BUTTON, self.takePhoto, idButton)
#        
#    def takePhoto(self, event):
#      cap = cv2.VideoCapture(0)
#      self.panel = wx.Panel(self, -1)
#      self.account = self.accountInput.GetValue()
#      name = self.path + '/'+self.account+'.png'
#      while(cap.isOpened()):
#        flag, im_rd = cap.read()
#        h,w,c = im_rd.shape
#        cv2.imshow('face', im_rd) 
#        try:
#          c = cv2.waitKey(1)
#          if c  == ord('s'):  
#            face_sum, bbox_sum = self.model.get_input(cv2.resize(im_rd, (112, 112)))
##            if len(face_sum) > 1:###采集到多人
##              accountLabel = wx.StaticText(self.panel, -1, u'采集到多个人脸', pos=(300, 28), size  = (75, 75))
##              accountLabel.SetForegroundColour(self.themeColor)
##              accountLabel.SetFont(font)
##            if len(face_sum) < 1:#＃##没有采集到人
##              accountLabel = wx.StaticText(self.panel, -1, u'没有采集到人脸', pos=(300, 28), size  = (75, 75))
##              accountLabel.SetForegroundColour(self.themeColor)
##              acountLabel.SetFont(font)
##            else:
#            box = bbox_sum[0]
#            cv2.rectangle(im_rd, (int(box[0]/112*w), int(box[1]/112*h)),(int(box[2]/112*w), int(box[3]/112*h)),(55, 255, 155),1)
##            img_save = im_rd[int(box[1]/112*h):int(box[3]/112*h), int(box[0]/112*w):int(box[2]/112*w)]
#            cv2.imwrite(name, cv2.resize(im_rd,(320,240)))
#            register_img = wx.Image(name ,wx.BITMAP_TYPE_PNG).ConvertToBitmap()
#            wx.StaticBitmap(self.panel,-1,register_img, pos = (200,200), size = (320,240))
#          if c == ord('q'):
#            cap.release()
#            cv2.destroyAllWindows()
#        except:
#          print('No face dected')
      
#class LoginPanel(wx.Panel):
#
#    def __init__(self):
#        self.panel = wx.Panel(None, -1, size = (1000, 800))
#    
#    def Login(self):
#      
#      ###显示姓名
#      font = wx.Font(12, wx.DEFAULT, wx.NORMAL, wx.BOLD, True)
#      accountLabel = wx.StaticText(self.panel, -1, u'姓   名', pos=(70, 28), size  = (75, 75), style = None, name = '姓名')
#      accountLabel.SetForegroundColour(self.themeColor)
#      accountLabel.SetFont(font)
#      self.accountInput = wx.TextCtrl(self.panel, -1, u'', pos=(145, 25), size=(150, -1))
#      self.accountInput.SetForegroundColour('gray')
#      self.accountInput.SetFont(font)
#      
#      ###拍照按钮
#      idButton = wx.Button(self.panel, -1, u'拍   照', pos=(70, 75), size=(50, 40))
#      idButton.SetForegroundColour('white')
#      idButton.SetBackgroundColour(self.themeColor)
##       self.Bind(wx.EVT_BUTTON, self.takePhoto, idButton)

      
    
def main():
    app = wx.App()
    
    app.MainLoop()
if __name__ == '__main__':
    main()