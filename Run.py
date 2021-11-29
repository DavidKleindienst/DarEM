#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:34:50 2021

@author: dkleindienst
"""


import sys, os, re
import shutil
from PyQt5.QtWidgets import (
    QWidget,
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QMainWindow,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QPlainTextEdit,
    QListWidget,
    QFileDialog,
    QAbstractItemView
    
)
from PyQt5.QtCore import Qt

from utils.makeTFRecords import maketfRecords

#Default Values
downscale_targetsize=2048
split_targetsize=1080
overlap=0
eval_probability=0.15
min_score=0.15

def processTfRecordFilename(filename):
    
    return filename

class SubWindow(QMainWindow):
    def __init__(self,parent):
        super().__init__(parent)
        self.parent = parent
        
    def closeEvent(self,event):
        self.parent.show()
        super().closeEvent(event)

class ConfigureWindow(SubWindow):
    def __init__(self,parent):
        super().__init__(parent)
        txt=QLabel('st')
        self.setCentralWidget(txt)
    

class ImportWindow(SubWindow):
    def __init__(self,parent):
        super().__init__(parent)
        self.makeFrom='Darea'
        self.configs=[]
        self.tfRecord=''
        layoutMain = QVBoxLayout()
        layoutFrom=QHBoxLayout()
        
        layoutFrom.addWidget(QLabel('Import dataset from'))
        self.combo=QComboBox()
        self.combo.addItem("Darea project file")
        self.combo.addItem("Folder with labelImg annotated images")
        self.combo.addItem("Folder with manually annotated images")
        self.combo.currentIndexChanged.connect(self.changeInputType)
        layoutFrom.addWidget(self.combo)
        layout=QHBoxLayout()
        layoutMain.addLayout(layoutFrom)
        self.listbox=QListWidget()
        self.listbox.setSelectionMode(QAbstractItemView.ExtendedSelection)

        layout.addWidget(self.listbox)
        layout2=QVBoxLayout()
        self.addButton=QPushButton("+")
        self.addButton.clicked.connect(self.addConfig)
        removeButton=QPushButton("-")
        removeButton.clicked.connect(self.remove)
        layout2.addWidget(self.addButton)
        layout2.addWidget(removeButton)
        layout.addLayout(layout2)
        layoutMain.addLayout(layout)
        
        l_save=QHBoxLayout()
        l_save.addWidget(QLabel('Save dataset as'))
        self.saveFileButton=QPushButton("Select File...")
        self.saveFileButton.clicked.connect(self.selectTfrecord)
        l_save.addWidget(self.saveFileButton)
        layoutMain.addLayout(l_save)
        
        l_run=QHBoxLayout()
        startButton=QPushButton('Make')
        startButton.clicked.connect(self.makeRecords)
        l_run.addWidget(startButton)
        cancelButton=QPushButton('Cancel')
        cancelButton.clicked.connect(self.close)
        l_run.addWidget(cancelButton)
        layoutMain.addLayout(l_run)
        self.progress=QLabel('')
        self.progress.setAlignment(Qt.AlignCenter)
        layoutMain.addWidget(self.progress)
        
        widget=QWidget()
        widget.setLayout(layoutMain)
        self.setCentralWidget(widget)
    def changeInputType(self,idx):
        self.listbox.clear()
        self.addButton.clicked.disconnect()
        if idx==0:
            self.makeFrom='Darea'
            self.addButton.clicked.connect(self.addConfig)
        elif idx==1:
            self.makeFrom='labelImg'
            self.addButton.clicked.connect(self.addFolder)
        elif idx==2:
            self.makeFrom='folder'
            self.addButton.clicked.connect(self.addFolder)
        else:
            raise ValueError('This option is unknown (this is a bug)')
        
    def addConfig(self):
        paths = QFileDialog.getOpenFileNames(self, 'Select project files',filter='Darea project file (*.dat)')[0]
        for p in paths:
            if p not in self.configs:
                self.configs.append(p)
        self.displayPaths()
    def addFolder(self):
        paths = QFileDialog.getExistingDirectory(self,"Select Directory")
        if paths:
            self.configs.append(paths)
        self.displayPaths()
    def remove(self):
        idx=self.listbox.selectedIndexes()

        idx = [i.row() for i in idx]
        idx.sort(reverse=True)
        for i in idx:
            self.configs.pop(i)
        self.displayPaths()
    
    def displayPaths(self):
        self.listbox.clear()
        self.listbox.addItems(self.configs)
        
    def selectTfrecord(self):
        filename = QFileDialog.getSaveFileName(self,'Select File', filter='*.tfrecord')[0]
        if filename:
            filename.replace('_eval.tfrecord', '.tfrecord')
            filename.replace('_train.tfrecord', '.tfrecord')
            #Remove -?????-of-????? suffix if it exists
            p=re.compile('.tfrecord-\d+-of-\d+')
            m=p.search(filename)
            if m and len(filename)==m.span(0)[1]:
                filename = filename[0:m.span(0)[0]+len('.tfrecord')]
            self.tfRecord=filename
            self.saveFileButton.setText(filename)
        
    def makeRecords(self):
        self.progress.setText('Processing...')
        app.processEvents()
        outputFolder=os.path.splitext(self.tfRecord)[0]  
        
        maketfRecords(self.makeFrom,self.configs, self.tfRecord, outputFolder,
                        downscale_targetSize=downscale_targetsize,
                        split_targetSize=split_targetsize, overlap=overlap,
                        eval_probability=eval_probability, 
                        progressHandle=self.progress, app=app)
        shutil.rmtree(outputFolder)

class TrainingWindow(SubWindow):
    def __init__(self,parent):
        super().__init__(parent)
        networks=getNetworks()
        self.tfRecord=''
        self.tfRecordEval=''
        layoutMain = QVBoxLayout()
        
        l_networks=QHBoxLayout()
        l_networks.addWidget(QLabel('Neural Network'))
        network_select = QComboBox()
        for n in networks:
            network_select.addItem(n)
        l_networks.addWidget(network_select)
        layoutMain.addLayout(l_networks)
        
        l_tfRec=QHBoxLayout()
        l_tfRec.addWidget(QLabel('Dataset'))
        self.sel_tfRec=QPushButton("Select .tfrecord Dataset...")
        self.sel_tfRec.clicked.connect(self.selectTfrecord)
        l_tfRec.addWidget(self.sel_tfRec)
        layoutMain.addLayout(l_tfRec)
        
        widget=QWidget()
        widget.setLayout(layoutMain)
        self.setCentralWidget(widget)
        
    def selectTfrecord(self):
     filename = QFileDialog.getOpenFileNames(self,'Select File', filter='Tfrecord dataset (*.tfrecord)')[0][0]
     print(filename)
     if filename:
        
        #If suffix ?????-of-????? exists
        #Exchange first number by ?????
        p=re.compile('.tfrecord-\d+-of-\d+')
        m=p.search(filename)
        if m and len(filename)==m.span(0)[1]:
            nr_digits = int((m.span(0)[1] - m.span(0)[0] - len('.tfrecord--of-'))/2)
            number_index = -2*nr_digits-len('-of-')
            filename=filename[:number_index] + '?' * nr_digits + filename[number_index+nr_digits:]
            #filename = filename[0:m.span(0)[0]+len('.tfrecord')]
            
        
        self.tfRecord=filename
        self.sel_tfRec.setText(filename)

             
        

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Helper for automated imaging")
        layout=QVBoxLayout()
        buttonConfigure=QPushButton("Configure Prediction for SerialEM")
        buttonConfigure.clicked.connect(lambda x: self.open_window(ConfigureWindow))
        layout.addWidget(buttonConfigure)
        buttonImport=QPushButton("Import Images for Training")
        buttonImport.clicked.connect(lambda x: self.open_window(ImportWindow))
        layout.addWidget(buttonImport)
        buttonTraining = QPushButton("Configure Training")
        buttonTraining.clicked.connect(lambda x: self.open_window(TrainingWindow))
        layout.addWidget(buttonTraining)
        
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
    def open_window(self, window):
        self.w = window(self)
        self.w.show()
        self.hide()
        
        
def getNetworks():
    folder=os.path.dirname(os.path.realpath(__file__))
    modelfolder = os.path.join(folder,'models')
    networks = [f for f in os.listdir(modelfolder)
                if not f.startswith('.') and
                os.path.isdir(os.path.join(modelfolder,f)) and
                os.path.isfile(os.path.join(modelfolder,f,'pipeline_default.config'))]
    networks.sort()
    return networks

app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
    
