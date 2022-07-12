import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

import os
import pydicom as dicom
import numpy as np
import cv2 as cv
from scipy import ndimage as ndi
from skimage.measure import regionprops_table
from skimage import feature, io
import tkinter as tk
from tkinter import filedialog
import qimage2ndarray
import pandas as pd
from PIL import Image
from csv import reader

qtCreatorFile = "C:\Temp_Data\PyQt-GUI\Seed-xray-analysis.ui" # Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        #Drop down menu action setup
        self.actionOpen_DICOM_file.triggered.connect(self.OpenDICOM)
        self.actionZoom_in.triggered.connect(self.ZoomIn)
        self.actionZoom_out.triggered.connect(self.ZoomOut)
        self.actionZoom_reset.triggered.connect(self.ZoomReset)
        self.actionIncrease_Contrast.triggered.connect(self.IncreaseContrast)
        self.actionReduce_Contrast.triggered.connect(self.ReduceContrast)
        self.actionReset_Contrast.triggered.connect(self.ResetContrast)
        
        # Image processing toolbox action setup
        self.image_resolution_spinBox.valueChanged.connect(self.UpdateDataResolution)
        self.increase_contrast_pushButton.clicked.connect(self.IncreaseContrast)
        self.decrease_contrast_pushButton.clicked.connect(self.ReduceContrast)
        self.image_resolution_pushButton.clicked.connect(self.UpdateDataResolution) 
        self.small_objects_spinBox.valueChanged.connect(self.CleanupImage)
        self.large_objects_spinBox.valueChanged.connect(self.CleanupImage)
        self.image_cleanup_pushButton.clicked.connect(self.CleanupImage)
        self.background_spinBox.valueChanged.connect(self.DefineSearchArea)
        self.foreground_spinBox.valueChanged.connect(self.DefineSearchArea)
        self.bg_fg_pushButton.clicked.connect(self.DefineSearchArea)   
        self.run_edge_detection_pushButton.clicked.connect(self.DetectSeeds)
        self.calc_props_pushButton.clicked.connect(self.CalcRegionProps)
        self.min_intensity_spinBox.valueChanged.connect(self.UpdateIntensity)
        self.max_intensity_spinBox.valueChanged.connect(self.UpdateIntensity)
        self.viability_pushButton.clicked.connect(self.ClassifyViable)
        self.viability_spinBox.valueChanged.connect(self.ClassifyViable)
        self.save_image_pushButton.clicked.connect(self.SaveImage)
        
        #presets menu setup
        self.actionAcacia_aulacophylla.triggered.connect(self.AcaciaAulacophyllaPreset)
        self.actionAcacia_andrewsii.triggered.connect(self.AcaciaAndrewsiiPreset)
        self.actionAcacia_aneura.triggered.connect(self.AcaciaAneuraPreset)
        self.actionAcacia_acuaria.triggered.connect(self.AcaciaAcuariaPreset)        
        
    def OpenDICOM(self):
        # User select file for analysis
        root = tk.Tk()
        root.withdraw()
        self.file_path = filedialog.askopenfilename(title = 'Choose DICOM file', filetypes = [('DICOM files', '*.dcm')])
        file_name = os.path.split(self.file_path)[1] 
        self.status_bar.setText('File ' + str(file_name) + ' loaded')
        
        # Import DICOM file, format as image array
        DICOM_data = dicom.read_file(self.file_path)
        # Load dimensions based on the number of rows, columns)
        ConstPixelDims = (int(DICOM_data.Rows), int(DICOM_data.Columns))
        ConstPixelSpacing = (float(DICOM_data.PixelSpacing[0]), float(DICOM_data.PixelSpacing[1]))
        x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
        x = np.delete(x,-1)
        y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
        y = np.delete(y,-1)
        self.DICOMimage = DICOM_data.pixel_array
        # normalize and scale it to 8-bit grayscale:
        self.image = (-(self.DICOMimage -np.percentile(self.DICOMimage,50))*(255/(np.percentile(self.DICOMimage,50)-np.percentile(self.DICOMimage,0.001))))
        self.image[self.image<0] = 0
        self.image[self.image>255] = 255
        self.data = np.uint8(self.image)
        self.scale_factor = 100 # percent of original size
        self.zoom_factor = 100
        self.data_scale = 100
        self.contrast_factor = 3 # Ratio to original scaled contrast
        width = int(self.image.shape[1] * self.scale_factor / 100)
        height = int(self.image.shape[0] * self.scale_factor / 100)
        dim = (width, height)
        self.image_display = cv.resize(self.image, dim, interpolation = cv.INTER_AREA)
        self.UpdateImageDisplay()
        
    def UpdateImageDisplay(self):
        image_profile = qimage2ndarray.array2qimage(self.image_display)
        self.image_profile = image_profile.scaled(self.image_display.shape[1],self.image_display.shape[0], aspectRatioMode=QtCore.Qt.KeepAspectRatio, 
                                    transformMode=QtCore.Qt.SmoothTransformation) # To scale image for example and keep its Aspect Ratio    
        self.image_view.setGeometry(0, 0, self.image_display.shape[0], self.image_display.shape[1])
        self.image_view.setPixmap(QtGui.QPixmap.fromImage(self.image_profile))
        self.image_view.setMinimumSize(self.image_display.shape[1], self.image_display.shape[0])
        self.scrollAreaWidget_image_view.setMinimumSize(self.image_display.shape[1], self.image_display.shape[0])

    def UpdateDataResolution(self):
        self.data = self.image
        self.data_scale = self.image_resolution_spinBox.value()
        TEMPimage = ((self.data-np.median(self.data)-10) * self.contrast_factor + np.median(self.data)+10)
        TEMPimage[TEMPimage<0] = 0
        TEMPimage[TEMPimage>255] = 255
        self.TEMPimage = np.uint8(TEMPimage)
        data_width = int(self.TEMPimage.shape[1] * self.zoom_factor / 100 * self.data_scale / 100)
        data_height = int(self.TEMPimage.shape[0] * self.zoom_factor / 100 * self.data_scale / 100)
        data_dim = (data_width, data_height)
        self.scaled_data = cv.resize(self.TEMPimage, data_dim, interpolation = cv.INTER_AREA)
        self.image_display = self.scaled_data
        self.UpdateImageDisplay()
        self.status_bar.setText('Raw data scaled to ' + str(self.data_scale) + '% of original DICOM file')

    def CleanupImage(self):       
        #Binary threshold algorithm, OTsu algorithm to choose optimum threshold value
        ret, thresh = cv.threshold(self.scaled_data,np.median(self.scaled_data)+5,255,cv.THRESH_BINARY)
        # Remove small objects from image
        thresh_bool = thresh.astype(bool)
        label_seeds, nb_labels = ndi.label(thresh_bool)
        sizes = np.bincount(label_seeds.ravel())
        mask_sizes = sizes < self.large_objects_spinBox.value()
        mask_sizes[0] = 0
        thresh = mask_sizes[label_seeds]
        thresh = thresh + 1
        thresh = thresh - 1
        thresh = thresh * 255
        thresh = np.uint8(thresh)
        # Remove small objects from image
        thresh_bool = thresh.astype(bool)
        label_seeds, nb_labels = ndi.label(thresh_bool)
        sizes = np.bincount(label_seeds.ravel())
        mask_sizes = sizes > self.small_objects_spinBox.value()
        mask_sizes[0] = 0
        thresh = mask_sizes[label_seeds]
        thresh = thresh + 1
        thresh = thresh - 1
        thresh = thresh * 255
        # Re-number labels with noise removed
        thresh_bool = thresh.astype(bool)
        label_seeds, nb_labels = ndi.label(thresh_bool)
        self.data_scale = 100
        self.threshold_data = np.uint8(thresh)        
        self.image_display = self.threshold_data
        self.data = self.threshold_data
        self.UpdateImageDisplay()
        self.status_bar.setText('Median intensity is for data is ' + str(np.median(self.scaled_data))
                                + '. Small and large objects removed as per defined pixel size')
    
    def DefineSearchArea(self):
        #Binary threshold algorithm, OTsu algorithm to choose optimum threshold value
        ret, thresh = cv.threshold(self.scaled_data,np.median(self.scaled_data)+5,255,cv.THRESH_BINARY)
        # Remove small objects from image
        thresh_bool = thresh.astype(bool)
        label_seeds, nb_labels = ndi.label(thresh_bool)
        sizes = np.bincount(label_seeds.ravel())
        mask_sizes = sizes < self.large_objects_spinBox.value()
        mask_sizes[0] = 0
        thresh = mask_sizes[label_seeds]
        thresh = thresh + 1
        thresh = thresh - 1
        thresh = thresh * 255
        thresh = np.uint8(thresh)
        # Remove small objects from image
        thresh_bool = thresh.astype(bool)
        label_seeds, nb_labels = ndi.label(thresh_bool)
        sizes = np.bincount(label_seeds.ravel())
        mask_sizes = sizes > self.small_objects_spinBox.value()
        mask_sizes[0] = 0
        thresh = mask_sizes[label_seeds]
        thresh = thresh + 1
        thresh = thresh - 1
        thresh = thresh * 255
        # Re-number labels with noise removed
        thresh_bool = thresh.astype(bool)
        label_seeds, nb_labels = ndi.label(thresh_bool)
        self.data_scale = 100
        self.threshold_data = np.uint8(thresh) 
        
        kernel_size = self.background_spinBox.value()
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        self.background = cv.dilate(self.threshold_data,kernel,iterations=1)
        self.background = np.int8(self.background)        
       
        dist_transform = cv.distanceTransform(self.threshold_data,cv.DIST_L2,3)
        ret, self.foreground = cv.threshold(dist_transform,(self.foreground_spinBox.value())*dist_transform.max(),255,0)
        self.foreground = np.int8(self.foreground)

        self.search_area = cv.subtract(self.background,self.foreground)
        self.search_area = np.uint8(self.search_area)
        self.data = self.search_area
        self.image_display = self.search_area
        self.UpdateImageDisplay()
        
    def DetectSeeds(self):
                # Marker labelling
        ret, self.markers = cv.connectedComponents(self.foreground)
        # Add one to all labels so that sure background is not 0, but 1
        self.markers = self.markers+1
        # Now, mark the region of unknown with zero
        self.markers[self.search_area==255] = 0 
        scaled_data_stack = np.stack((self.scaled_data, self.scaled_data, self.scaled_data), axis = 2)
        self.markers = cv.watershed(scaled_data_stack,self.markers)
        
        self.data = self.image
        self.data_scale = self.image_resolution_spinBox.value()
        TEMPimage = ((self.data-np.median(self.data)-10) * self.contrast_factor + np.median(self.data)+10)
        TEMPimage[TEMPimage<0] = 0
        TEMPimage[TEMPimage>255] = 255
        self.TEMPimage = np.uint8(TEMPimage)
        data_width = int(self.TEMPimage.shape[1] * self.zoom_factor / 100 * self.data_scale / 100)
        data_height = int(self.TEMPimage.shape[0] * self.zoom_factor / 100 * self.data_scale / 100)
        data_dim = (data_width, data_height)
        self.scaled_data = cv.resize(self.TEMPimage, data_dim, interpolation = cv.INTER_AREA)
        self.image_display = self.scaled_data
        self.image_display = np.stack((self.image_display, self.image_display, self.image_display ), axis = 2)
        self.image_display[self.markers == -1] = [255,0,0]     
        self.UpdateImageDisplay()      
        self.status_bar.setText(str(np.max(self.markers)-1) + ' seeds detected.')

    def CalcRegionProps(self):
        def percentile(regionmask, intensity_image):
            return np.percentile(intensity_image, q=(90))
        props = pd.DataFrame(regionprops_table(self.markers,intensity_image=self.scaled_data, properties =
                                               ('label','centroid','filled_area','max_intensity','mean_intensity',
                                                'min_intensity','perimeter'), extra_properties=[percentile]))
        print(props)
        self.props = props.drop(props.index[[0]])        
        min_int = props['mean_intensity'].quantile(0.1)
        median_int = props['mean_intensity'].quantile(0.5)
        max_int = props['mean_intensity'].quantile(0.9)
        self.min_intensity_spinBox.setValue(int(min_int))
        self.viability_spinBox.setValue(int(median_int))
        self.max_intensity_spinBox.setValue(int(max_int))
#        print(props)
        mean_intensity = self.markers
        for index, row in self.props.iterrows():
            mean_intensity = np.where(mean_intensity == int(row['label']), row['mean_intensity'], mean_intensity)
        min_intensity = self.min_intensity_spinBox.value()
        max_intensity = self.max_intensity_spinBox.value()
        range_intensity = max_intensity - min_intensity
        mean_intensity = np.where(mean_intensity > 1, (mean_intensity-min_intensity)*(255/range_intensity), mean_intensity)
        self.image_display = np.stack((mean_intensity, mean_intensity, mean_intensity), axis = 2)
        self.image_display[self.markers == -1] = [255,0,0]     
        self.UpdateImageDisplay()
        
    def UpdateIntensity(self):
        mean_intensity = self.markers
        for index, row in self.props.iterrows():
            mean_intensity = np.where(mean_intensity == int(row['label']), row['mean_intensity'], mean_intensity)
        min_intensity = self.min_intensity_spinBox.value()
        max_intensity = self.max_intensity_spinBox.value()
        range_intensity = max_intensity - min_intensity
        mean_intensity = (mean_intensity-min_intensity)*(255/range_intensity)
        self.image_display = np.stack((mean_intensity, mean_intensity, mean_intensity), axis = 2)
        self.image_display[self.markers == -1] = [255,0,0]     
        self.UpdateImageDisplay()
    '''
    def ClassifyViable(self):
        viable_cutoff = self.viability_spinBox.value()
        mean_intensity = self.markers
        for index, row in self.props.iterrows():
            mean_intensity = np.where(mean_intensity == np.int(row['label']), row['mean_intensity'], mean_intensity)
        kernel_size=3
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        mean_intensity = cv.dilate(mean_intensity,kernel,iterations=1)      
        seed_borders = np.where(self.markers == -1, mean_intensity, 0)
        seed_borders = cv.dilate(seed_borders,kernel,iterations=1)
        image_display_G = np.where(seed_borders > viable_cutoff, 150, self.scaled_data)
        image_display_R = np.where((seed_borders <= viable_cutoff) & (seed_borders > 1), 150, self.scaled_data)
        image_display_B = self.scaled_data
        self.image_display = np.stack((image_display_R, image_display_G, image_display_B), axis = 2)       
        self.UpdateImageDisplay()
        temp = self.props.apply(lambda x: True if x['mean_intensity'] > viable_cutoff else False, axis = 1)
        viable = len(temp[temp == True].index)
        temp = self.props.apply(lambda x: True if x['mean_intensity'] <= viable_cutoff else False, axis = 1)
        nonviable = len(temp[temp == True].index)
        self.status_bar.setText(str(np.max(self.markers)-1) + ' seeds detected. ' + str(viable) + ' seeds are viable, ' + str(nonviable) + ' seeds are nonviable')
    '''
    def ClassifyViable(self):
        viable_cutoff = self.viability_spinBox.value()
        percentile = self.markers
        for index, row in self.props.iterrows():
            percentile = np.where(percentile == int(row['label']), row['percentile'], percentile)
        kernel_size=3
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        percentile = cv.dilate(percentile,kernel,iterations=1)      
        seed_borders = np.where(self.markers == -1, percentile, 0)
        seed_borders = cv.dilate(seed_borders,kernel,iterations=1)
        image_display_G = np.where(seed_borders > viable_cutoff, 150, self.scaled_data)
        image_display_R = np.where((seed_borders <= viable_cutoff) & (seed_borders > 1), 150, self.scaled_data)
        image_display_B = self.scaled_data
        self.image_display = np.stack((image_display_R, image_display_G, image_display_B), axis = 2)       
        self.UpdateImageDisplay()
        temp = self.props.apply(lambda x: True if x['percentile'] > viable_cutoff else False, axis = 1)
        viable = len(temp[temp == True].index)
        temp = self.props.apply(lambda x: True if x['percentile'] <= viable_cutoff else False, axis = 1)
        nonviable = len(temp[temp == True].index)
        self.status_bar.setText(str(np.max(self.markers)-1) + ' seeds detected. ' + str(viable) + ' seeds are viable, ' + str(nonviable) + ' seeds are nonviable')

    def SaveImage(self):
        im = Image.fromarray(self.image_display)
        save_path = os.path.splitext(self.file_path)[0]
        im.save(str(save_path) + '.jpg')
        self.status_bar.setText('Current image saved to ' + str(save_path) + '.jpg')        
       
    def ZoomIn(self):
        TEMPimage = ((self.data-np.median(self.data)-10) * self.contrast_factor + np.median(self.data)+10)
        TEMPimage[TEMPimage<0] = 0
        TEMPimage[TEMPimage>255] = 255
        self.image_display = np.uint8(TEMPimage)
        self.scale_factor = self.scale_factor * 1.1
        self.zoom_factor = self.scale_factor
        width = int(self.data.shape[1] * self.zoom_factor / 100 * self.data_scale / 100)
        height = int(self.data.shape[0] * self.zoom_factor / 100 * self.data_scale / 100)
        dim = (width, height)
        self.image_display = cv.resize(self.data, dim, interpolation = cv.INTER_AREA)      
        self.UpdateImageDisplay()

    def ZoomOut(self):
        TEMPimage = ((self.data-np.median(self.data)-10) * self.contrast_factor + np.median(self.data)+10)
        TEMPimage[TEMPimage<0] = 0
        TEMPimage[TEMPimage>255] = 255
        self.image_display = np.uint8(TEMPimage)
        self.scale_factor = self.scale_factor / 1.1
        self.zoom_factor = self.scale_factor
        width = int(self.data.shape[1] * self.zoom_factor / 100 * self.data_scale / 100)
        height = int(self.data.shape[0] * self.zoom_factor / 100 * self.data_scale / 100)
        dim = (width, height)
        self.image_display = cv.resize(self.data, dim, interpolation = cv.INTER_AREA)      
        self.UpdateImageDisplay()

    def ZoomReset(self):
        self.scale_factor = 100 # percent of original size
        self.zoom_factor = 100
        TEMPimage = ((self.image-np.median(self.image)-10) * self.contrast_factor + np.median(self.image)+10)
        TEMPimage[TEMPimage<0] = 0
        TEMPimage[TEMPimage>255] = 255
        self.image_display = np.uint8(TEMPimage)
#        self.image = np.uint8(-(self.DICOMimage -np.max(self.DICOMimage ))*(255/(np.max(self.DICOMimage )-np.min(self.DICOMimage ))))
        width = int(self.image.shape[1] * self.zoom_factor / 100 * self.data_scale / 100)
        height = int(self.image.shape[0] * self.zoom_factor / 100 * self.data_scale / 100)
        dim = (width, height)
        self.image_display = cv.resize(self.image_display, dim, interpolation = cv.INTER_AREA)
        self.UpdateImageDisplay()           

    def IncreaseContrast(self):
        self.contrast_factor  = self.contrast_factor  * 1.25
        TEMPimage = ((self.image-np.median(self.image)-10) * self.contrast_factor + np.median(self.image)+10)
        TEMPimage[TEMPimage<0] = 0
        TEMPimage[TEMPimage>255] = 255
        self.image_display = np.uint8(TEMPimage)
        width = int(self.image.shape[1] * self.scale_factor / 100 * self.data_scale / 100)
        height = int(self.image.shape[0] * self.scale_factor / 100 * self.data_scale / 100)
        dim = (width, height)
        self.image_display = cv.resize(self.image_display, dim, interpolation = cv.INTER_AREA)      
        self.UpdateImageDisplay()          

    def ReduceContrast(self):
        self.contrast_factor = self.contrast_factor  / 1.25
        TEMPimage = ((self.image-np.median(self.image)-10) * self.contrast_factor + np.median(self.image)+10)
        TEMPimage[TEMPimage<0] = 0
        TEMPimage[TEMPimage>255] = 255
        self.image_display = np.uint8(TEMPimage)
        width = int(self.image.shape[1] * self.scale_factor / 100 * self.data_scale / 100)
        height = int(self.image.shape[0] * self.scale_factor / 100 * self.data_scale / 100)
        dim = (width, height)
        self.image_display = cv.resize(self.image_display, dim, interpolation = cv.INTER_AREA)      
        self.UpdateImageDisplay()

    def ResetContrast(self):
        self.image = np.uint8(-(self.DICOMimage -np.max(self.DICOMimage ))*(255/(np.max(self.DICOMimage )-np.min(self.DICOMimage ))))
        self.contrast_factor = 1 # Ratio to original scaled contrast
        width = int(self.image.shape[1] * self.scale_factor / 100 * self.data_scale / 100)
        height = int(self.image.shape[0] * self.scale_factor / 100 * self.data_scale / 100)
        dim = (width, height)
        self.image_display = cv.resize(self.image, dim, interpolation = cv.INTER_AREA)
        self.UpdateImageDisplay()
        
        
    
    def AcaciaAulacophyllaPreset(self):
        self.rowset = 0
        self.SetPreset()
    
    def AcaciaAndrewsiiPreset(self):
       self.rowset = 1
       self.SetPreset()
       
    def AcaciaAneuraPreset(self):
        self.rowset = 2
        self.SetPreset()
    
    def AcaciaAcuariaPreset(self):
        self.rowset = 3
        self.SetPreset()    
    
    def SetPreset(self):
        with open('AcaciaData.csv', 'r') as obj:
            data = []
            csv_reader = reader(obj)
            for row in csv_reader:
                data.append(row)
            #species = str(data[self.rowset][0])
            #magnification = int(data[self.rowset][1])          
            
            #contrast setting
            
            if int(data[self.rowset][2]) > 0:
                for i in range(int(data[self.rowset][2])):
                    self.IncreaseContrast()
            if int(data[self.rowset][2]) < 0:
                for i in range(abs(int(data[self.rowset][2]))):
                    self.ReduceContrast()    
            self.image_resolution_spinBox.setValue(int(data[self.rowset][3]))
            self.UpdateDataResolution()
            self.small_objects_spinBox.setValue(int(data[self.rowset][4]))
            self.large_objects_spinBox.setValue(int(data[self.rowset][5]))
            self.CleanupImage()
            self.background_spinBox.setValue(int(data[self.rowset][6]))
            self.foreground_spinBox.setValue(float(data[self.rowset][7]))
            self.DefineSearchArea()
            self.DetectSeeds()
            self.CalcRegionProps()
            self.min_intensity_spinBox.setValue(int(data[self.rowset][8]))
            self.max_intensity_spinBox.setValue(int(data[self.rowset][9]))
            self.UpdateIntensity()
            self.viability_spinBox.setValue(int(data[self.rowset][10]))
            self.ClassifyViable()
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
