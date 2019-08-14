# -*- encoding:utf-8 -*-

system_version='v0.1.0.0531_alpha'

"""===========================================================================

                             IMPORT FILES

==========================================================================="""
import wx
import os
import time
import cv2
import scipy.misc
import os
import _thread


"""===========================================================================

                              VARIABLES

==========================================================================="""
#data_path    = 'E:\ICDM_PREDICTION/test_img/video'
data_path    = 'E:\ICDM_PREDICTION/test_img/501_mm'
img_test     = 'E:\ICDM_PREDICTION/test_img/120/1.jpg'
image_cover  = 'E:\ICDM_PREDICTION/test_img/200/12.jpg'


"""===========================================================================

                             FUNCTIONS

==========================================================================="""
class Frame(wx.Frame):
	def __init__(self,parent,title):
		wx.Frame.__init__(self,parent,title=title,size=(1100,550))
		self.panel = wx.Panel(self)
		self.Center()

		self.image_cover = wx.Image(COVER, wx.BITMAP_TYPE_ANY).Scale(350,300)

		self.bmp         = wx.StaticBitmap(self.panel, -1, wx.Bitmap(self.image_cover))
		self.bmp2        = wx.StaticBitmap(self.panel , pos = (500,0))


		start_button = wx.Button(self.panel,label='Start' , pos = (370,100))
		close_button = wx.Button(self.panel,label='Close' , pos = (370,200))

		self.Bind(wx.EVT_BUTTON,self.showing,start_button)
		self.Bind(wx.EVT_BUTTON,self.closing,close_button)



	def _showing(self,event):
		trig = 'True'
		f = open('trig.txt', 'w')
		f.write(trig)
		f.close()
		all_index = os.listdir(data_path)
		data_len  = len(all_index)
		data1     = []
		
		for i in range(500):
			image = scipy.misc.imread(data_path + '/' + all_index[i], 'rb')
			image_path = data_path + '/' + all_index[i]
			
			height , width = image.shape
			pic   = wx.Bitmap.FromBuffer(width,height,image)
		
			self.bmp2.SetBitmap(pic)
			time.sleep(0.08)
			if i%5 == 0:
				f    = open('trig.txt', 'r')
				trig = f.read()
				f.close()
				if trig == 'False':
					break
		
		self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
		_thread.exit()



	def showing(self,event):
		_thread.start_new_thread(self._showing, (event,))


	def closing(self,event):
		trig = 'False'
		f = open('trig.txt', 'w')
		f.write(trig)
		f.close()


class App(wx.App):
	def OnInit(self):
		self.frame = Frame(parent=None,title="thread test")
		self.frame.Show(True)
		return True


def main():
	app = App()
	app.MainLoop()

if __name__ == '__main__':
     main()



