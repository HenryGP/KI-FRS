#!/usr/bin/env python
"""
	GUI Controller class:
	This is the triggering class for the KI-FRS application.
	Initializes the Kinect interface services and handles all user's input.
	The class was generated automatically using wxGlade 0.6.8
"""

from recognizer import Recognizer
import kinect
import preprocessor
import wx
import thread
import gettext
import cv2 as cv

class Frame(wx.Frame):
	def __init__(self, *args, **kwds):
		"""
			Initialization of the graphic components
		"""
		# begin wxGlade: Frame.__init__
		wx.Frame.__init__(self, *args, **kwds)
		self.Controller_statusbar = self.CreateStatusBar(1, 0)
		self.notebook_15 = wx.Notebook(self, wx.ID_ANY, style=0)
		self.tab1 = wx.Panel(self.notebook_15, wx.ID_ANY)
		self.label_5 = wx.StaticText(self.tab1, wx.ID_ANY, _(" Type:"))
		self.combo_box_1 = wx.ComboBox(self.tab1, wx.ID_ANY, choices=[_("Training"), _("Test")], style=wx.CB_DROPDOWN | wx.CB_READONLY)
		self.label_1 = wx.StaticText(self.tab1, wx.ID_ANY, _(" ID: "))
		self.text_ctrl_1 = wx.TextCtrl(self.tab1, wx.ID_ANY, "")
		self.button_3 = wx.Button(self.tab1, wx.ID_ANY, _("New"))
		self.button_4 = wx.Button(self.tab1, wx.ID_ANY, _("Add"))
		self.button_1 = wx.Button(self.tab1, wx.ID_ANY, _("Pre-process Database"), style=wx.BU_EXACTFIT)
		self.tab2 = wx.Panel(self.notebook_15, wx.ID_ANY)
		self.label_2 = wx.StaticText(self.tab2, wx.ID_ANY, _(" Model:"))
		self.combo_box_2 = wx.ComboBox(self.tab2, wx.ID_ANY, choices=[_("Eigenfaces"), _("Fisherfaces")], style=wx.CB_DROPDOWN | wx.CB_READONLY)
		self.label_2_copy = wx.StaticText(self.tab2, wx.ID_ANY, _(" Source:"))
		self.combo_box_2_copy = wx.ComboBox(self.tab2, wx.ID_ANY, choices=[_("Automatic"), _("B&W"), _("Depth Image")], style=wx.CB_DROPDOWN | wx.CB_READONLY)
		self.button_7 = wx.Button(self.tab2, wx.ID_ANY, _("Train"))
		self.button_8 = wx.Button(self.tab2, wx.ID_ANY, _("Test"))
		self.static_line_1 = wx.StaticLine(self.tab2, wx.ID_ANY)
		self.label_3 = wx.StaticText(self.tab2, wx.ID_ANY, _(" Recognize:"))
		self.button_10 = wx.Button(self.tab2, wx.ID_ANY, _("Identify"))
		self.__set_properties()
		self.__do_layout()
		self.Bind(wx.EVT_BUTTON, self.new_form, self.button_3)
		self.Bind(wx.EVT_BUTTON, self.add_database, self.button_4)
		self.Bind(wx.EVT_BUTTON, self.preprocess_database, self.button_1)
		self.Bind(wx.EVT_BUTTON, self.train_model, self.button_7)
		self.Bind(wx.EVT_BUTTON, self.test_model, self.button_8)
		self.Bind(wx.EVT_BUTTON, self.identify, self.button_10)
		self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.tab_changed, self.notebook_15)
		self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
		# end wxGlade

	def __set_properties(self):
		"""Setting up special properties on the graphic components"""
		# begin wxGlade: Frame.__set_properties
		self.SetTitle(_("Controller"))
		self.SetSize((188, 293))
		self.SetFocus()
		self.Controller_statusbar.SetStatusWidths([-1])
		# statusbar fields
		Controller_statusbar_fields = [_("Press NEW to start sampling")]
		for i in range(len(Controller_statusbar_fields)):
		    self.Controller_statusbar.SetStatusText(Controller_statusbar_fields[i], i)
		self.label_5.SetFont(wx.Font(11, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Droid Sans"))
		self.combo_box_1.SetMinSize((80,27))
		self.combo_box_1.SetSelection(0)
		self.label_1.SetFont(wx.Font(11, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Droid Sans"))
		self.text_ctrl_1.SetMinSize((170, 27))
		self.label_2.SetFont(wx.Font(11, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Droid Sans"))
		self.combo_box_2.SetMinSize((170, 27))
		self.combo_box_2.SetSelection(0)
		self.label_2_copy.SetFont(wx.Font(11, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Droid Sans"))
		self.combo_box_2_copy.SetMinSize((170, 27))
		self.combo_box_2_copy.SetSelection(0)
		self.label_3.SetFont(wx.Font(11, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "Droid Sans"))
		# end wxGlade

	def __do_layout(self):
		"""Distributes the created components in the frame"""
		# begin wxGlade: Frame.__do_layout
		sizer_1 = wx.BoxSizer(wx.VERTICAL)
		sizer_3 = wx.BoxSizer(wx.VERTICAL)
		sizer_5 = wx.BoxSizer(wx.HORIZONTAL)
		sizer_4 = wx.BoxSizer(wx.VERTICAL)
		sizer_2 = wx.BoxSizer(wx.VERTICAL)
		grid_sizer_2 = wx.GridSizer(1, 2, 0, 0)
		grid_sizer_1 = wx.GridSizer(2, 1, 0, 0)
		grid_sizer_3 = wx.GridSizer(1, 2, 0, 0)
		grid_sizer_3.Add(self.label_5, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 0)
		grid_sizer_3.Add(self.combo_box_1, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 0)
		sizer_4.Add(grid_sizer_3, 1, wx.ALIGN_CENTER_VERTICAL, 6)
		grid_sizer_1.Add(self.label_1, 0, wx.ALIGN_CENTER_VERTICAL, 0)
		grid_sizer_1.Add(self.text_ctrl_1, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL, 0)
		sizer_4.Add(grid_sizer_1, 1, wx.EXPAND, 0)
		grid_sizer_2.Add(self.button_3, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL, 0)
		grid_sizer_2.Add(self.button_4, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL, 0)
		sizer_4.Add(grid_sizer_2, 1, wx.ALIGN_BOTTOM, 0)
		sizer_2.Add(self.button_1, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL, 1)
		sizer_2.Add((0, 5), 0, 0, 0)
		sizer_4.Add(sizer_2, 1, wx.EXPAND, 0)
		self.tab1.SetSizer(sizer_4)
		sizer_3.Add((0, 5), 0, 0, 0)
		sizer_3.Add(self.label_2, 0, 0, 0), _("Depth Image")
		sizer_3.Add(self.combo_box_2, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL, 0)
		sizer_3.Add(self.label_2_copy, 0, 0, 0)
		sizer_3.Add((0, 5), 0, 0, 0)
		sizer_3.Add(self.combo_box_2_copy, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL, 0)
		sizer_3.Add((0, 5), 0, 0, 0)
		sizer_5.Add(self.button_7, 0, 0, 0)
		sizer_5.Add(self.button_8, 0, 0, 0)
		sizer_3.Add(sizer_5, 0, wx.EXPAND, 0)
		sizer_3.Add((0, 5), 0, 0, 0)
		sizer_3.Add(self.static_line_1, 0, wx.EXPAND, 0)
		sizer_3.Add(self.label_3, 0, 0, 0)
		sizer_3.Add(self.button_10, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL, 0)
		self.tab2.SetSizer(sizer_3)
		self.notebook_15.AddPage(self.tab1, _("Sampler"))
		self.notebook_15.AddPage(self.tab2, _("Model"))
		sizer_1.Add(self.notebook_15, 1, wx.EXPAND, 0)
		self.SetSizer(sizer_1)
		self.Layout()
		# end wxGlade

	def OnCloseWindow(self, event):
		"""Event when closing the main window"""
		kinect.keep_running = False #makes the Kinect interface go to "close" sequence		
		self.Destroy()

	def new_form(self, event):
		"""
			Cleans up the text field for new sampling
		"""
		if self.combo_box_1.GetSelection()==0:
			mode = "tr"
		else:
			mode = "ts"
		kinect.new_sampling(mode)
		self.text_ctrl_1.SetValue("")
		self.Controller_statusbar.SetStatusText("Ready for sampling")
		event.Skip()

	def add_database(self, event):  
		"""Adds the captured data from the Kinect device into the data directory"""
		if self.combo_box_1.GetSelection()==0:
			kinect.run_mode = "tr"
		else:
			kinect.run_mode = "ts"
		
		name = self.text_ctrl_1.GetValue()
		if name=="":
			self.Controller_statusbar.SetStatusText("Name field can't be blank")
		else:
			kinect.file_saving(name)
			self.Controller_statusbar.SetStatusText("Sample added to database")
		event.Skip()

	def preprocess_database(self, event):
		"""
			Triggers preprocessor actions for the entire training or testing set
		"""
		if self.combo_box_1.GetSelection()==0:
			mode = "tr"
		else:
			mode = "ts"
		ret = preprocessor.normalize_images(mode)
		self.Controller_statusbar.SetStatusText("Processed "+str(ret)+" samples")
		event.Skip()

	def train_model(self, event):  
		"""Trains the Face Recognition model"""
		if self.combo_box_2.GetSelection()==0:
			mode = 1
		else:
			mode = 2
		selection = self.combo_box_2_copy.GetSelection()
		if selection==0:
			self.Controller_statusbar.SetStatusText("Mode only for on-line recognition")
		elif selection==1: #Black and white image
			rec = Recognizer(mode,"nbw")
			rec.tr()
			self.Controller_statusbar.SetStatusText("Model successfully trained")
		elif selection==2:#Depth image
			rec=Recognizer(mode,"ndepth")
			rec.tr()
			self.Controller_statusbar.SetStatusText("Model successfully trained")
		event.Skip()

	def test_model(self, event): 
		"""Tests the Face Recognition model"""
		if self.combo_box_2.GetSelection()==0:
			mode = 1
		else:
			mode = 2
		selection = self.combo_box_2_copy.GetSelection()
		if selection==0:
			self.Controller_statusbar.SetStatusText("Mode only for on-line recognition")
		elif selection==1: #Black and white image
			rec = Recognizer(mode,"nbw")
			ret = rec.ts()
			print "ERROR: ",ret
			self.Controller_statusbar.SetStatusText("Classification error: %.2f"%(ret))
		elif selection==2:#Depth image
			rec = Recognizer(mode,"ndepth")
			ret = rec.ts()
			print "ERROR: ",ret
			self.Controller_statusbar.SetStatusText("Classification error: %.2f"%(ret))
		event.Skip()

	def identify(self, event):  
		"""
			On-line face recognition
		"""
		rgb_img,depth_img,depth_mtx = kinect.get_sample()
		r_bw,r_depth,r_mtx=preprocessor.normalize_sample(rgb_img,depth_img,depth_mtx)
		if r_bw==None:
			event.Skip()
			self.Controller_statusbar.SetStatusText("There where no faces detected")
			return
		#0: eigenfaces (1), 1: fisherfaces(2)
		if self.combo_box_2.GetSelection()==0:
			model = 1
		else:
			model = 2
		#0: automatic("auto"), 1:"nbw", 2:"ndepth"
		if self.combo_box_2_copy.GetSelection()==0:
			mode = "auto"
		elif self.combo_box_2_copy.GetSelection()==1:
			mode = "nbw"
		else:
			mode = "ndepth"
		rec = Recognizer(model)
		rec.predict(mode,r_bw,r_depth)				
		event.Skip()

	def tab_changed(self, event):  
		event.Skip()

if __name__ == "__main__":
	"""Initializes the application"""
	gettext.install("GUI_panel") # replace with the appropriate catalog name
	GUI_panel = wx.PySimpleApp(0)
	wx.InitAllImageHandlers()
	Controller = Frame(None, wx.ID_ANY, "",style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
	GUI_panel.SetTopWindow(Controller)
	Controller.Show() #Shows floating panel
	kinect.start("tr") #Initializes the Kinect interface