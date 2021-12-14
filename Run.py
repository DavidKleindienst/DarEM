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
    QLineEdit,
    QListWidget,
    QFileDialog,
    QAbstractItemView
    
)
from PyQt5.QtCore import Qt

from utils.makeTFRecords import maketfRecords
from utils.utils import getCheckpoints, getNetworks

#Default Values
downscale_targetsize=2048
split_targetsize=1080
overlap=0
eval_probability=0.15
min_score=0.15

MODELFOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models')

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
        buttonTraining = QPushButton("Perform Training")
        buttonTraining.clicked.connect(lambda x: self.open_window(TrainingWindow))
        layout.addWidget(buttonTraining)
        
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
    def open_window(self, window):
        self.window = window(self)
        self.window.show()
        self.hide()


class SubWindow(QMainWindow):
    '''Functionality for showing MainWindow after closing Subwindow'''
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
        self.setWindowTitle("Import training data")
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
        self.combo.setToolTip('Select what kind of data you want to import')
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
            filename=processTfRecordFilename(filename)
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
        self.setWindowTitle('Perform Training')
        networks=getNetworks(MODELFOLDER)
        self.tfRecord=''
        self.tfRecordEval=''
        #Still need to write UI for following!!
        self.label_map_path = 'label_map.pbtxt'
        self.num_steps = 120 #250000
        
        #ToDo: Deal with 0 or 1 network types
        # Deal with 0 checkpoints
        
        layoutMain = QVBoxLayout()
        
        l_tfRec=QHBoxLayout()
        l_tfRec.addWidget(QLabel('Dataset'))
        self.sel_tfRec=QPushButton("Select .tfrecord Dataset...")
        self.sel_tfRec.clicked.connect(self.selectTfrecord)
        l_tfRec.addWidget(self.sel_tfRec)
        layoutMain.addLayout(l_tfRec)
        
        l_networks=QHBoxLayout()
        l_networks.addWidget(QLabel('Neural Network Type'))
        self.network_select = QComboBox()
        for n in networks:
            self.network_select.addItem(n)
        self.network_select.currentTextChanged.connect(self.changeNetwork)
        l_networks.addWidget(self.network_select)
        layoutMain.addLayout(l_networks)
        
        l_continue = QHBoxLayout()
        l_continue.addWidget(QLabel('Continue Training From'))
        checkpoints = getCheckpoints(MODELFOLDER,self.network_select.currentText())
        self.continue_from = QComboBox()
        self.continue_from.addItems(checkpoints)
        l_continue.addWidget(self.continue_from)
        layoutMain.addLayout(l_continue)
        
        l_model_dir = QHBoxLayout()
        l_model_dir.addWidget(QLabel('Network name'))
        self.nameEdit = QLineEdit()
        l_model_dir.addWidget(self.nameEdit)
        layoutMain.addLayout(l_model_dir)
        
        l_run=QHBoxLayout()
        startButton=QPushButton('Start')
        startButton.clicked.connect(self.startTraining)
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
      
    def changeNetwork(self,text):
        self.continue_from.clear()
        self.continue_from.addItems(MODELFOLDER,getCheckpoints(text))
    def selectTfrecord(self):
     filename = QFileDialog.getOpenFileNames(self,'Select File', filter='Tfrecord dataset (*.tfrecord)')[0][0]
     if filename:
        self.sel_tfRec.setText(processTfRecordFilename(filename))
        #If suffix XXXXX-of-XXXXX exists
        #Exchange first number by ?????
        p=re.compile('.tfrecord-\d+-of-\d+')
        m=p.search(filename)
        if m and len(filename)==m.span(0)[1]:
            nr_digits = int((m.span(0)[1] - m.span(0)[0] - len('.tfrecord--of-'))/2)
            number_index = -2*nr_digits-len('-of-')
            filename = filename[:number_index] + '?' * nr_digits + filename[number_index+nr_digits:]
            
        if '_train.tfrecord' in filename:
            self.tfRecord = filename
            self.tfRecordEval = filename.replace('_train.tfrecord', '_eval.tfrecord')
        elif '_eval.tfrecord' in filename:
            self.tfRecordEval = filename
            self.tfRecord = filename.replace('_eval.tfrecord', '_train.tfrecord')
        else:
            self.tfRecord = filename
            self.tfRecordEval = ''
            

    def startTraining(self):
        from model_main_tf2 import main as train
        from absl import flags
        
        if not self.tfRecord:
            self.progress.setText('You need to select a dataset!')
            return
        if not self.nameEdit.text():
            self.progress.setText('You have to name your network!')
            return

        self.progress.setText('Preparing...')
        app.processEvents()
        model_dir = os.path.join(MODELFOLDER, self.network_select.currentText(),
                                 self.nameEdit.text())
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        
        default_config_file = os.path.join(MODELFOLDER,
                                           self.network_select.currentText(),
                                           'pipeline_default.config') 
        pipeline_file = os.path.join(model_dir, 'pipeline.config')
        self.makePipelineConfig(default_config_file, pipeline_file)
        
        self.progress.setText('Performing Training...')
        app.processEvents()
        
        #flags will be read by train even when not explicitly passed
        flags.FLAGS(['model_main_tf2.py', 
                   '--pipeline_config_path', pipeline_file,
                   '--model_dir', model_dir])
        train()
        if self.tfRecordEval:
            # Run evaluation (Evaluation is triggered by including
            # checkpoint_dir variable)
            self.progress.setText('Performing Evaluation...')
            app.processEvents()
            
            flags.FLAGS(['model_main_tf2.py', 
                   '--pipeline_config_path', pipeline_file,
                   '--model_dir', model_dir, '--checkpoint_dir', model_dir])
            train()
        self.progress.setText('Training completed!')
        app.processEvents()

    def makePipelineConfig(self,default_config_file, config_file):
        #Reads the default_config for the selected model and
        #Changes Values to make it ready for training
        #Then saves modified pipeline.config under config_file path
        
        checkpoint = os.path.join(MODELFOLDER,self.network_select.currentText(),
                                  self.continue_from.currentText())
        
        
        with open(default_config_file,'r') as f:
            text = f.read()
            
        text = text.replace('label_map_path: "PATH_TO_LABELMAP"',
                     f'label_map_path: "{self.label_map_path}"')
        text = text.replace('input_path: "INPUT_PATH.tfrecord"',
                     f'input_path: "{self.tfRecord}"')
        if self.tfRecordEval:
            text = text.replace('input_path: "EVAL_PATH.tfrecord"',
                         f'input_path: "{self.tfRecordEval}"')
        text = text.replace('fine_tune_checkpoint: "PATH_TO_CHECKPOINT"',
                     f'fine_tune_checkpoint: "{checkpoint}"')
        
        #This would be better with regex in case the val is changed in the default
        text = text.replace('total_steps: 250000',
                     f'total_steps: {self.num_steps}')
        text = text.replace('num_steps: 250000',
                     f'num_steps: {self.num_steps}')
        #Warmup steps need to be less than total steps
        p = re.compile('warmup_steps: (\d+)')
        m = p.search(text)
        if m and int(m.group(1)) > self.num_steps:
            text = text.replace(f'warmup_steps: {m.group(1)}',
                                f'warmup_steps: {int(self.num_steps/2)}')
        
        with open(config_file,'w') as f:
            f.write(text)

def processTfRecordFilename(filename):
    #Removes _eval and _train extensions as well as 
    #the XXXXX-of-XXXXX
    filename.replace('_eval.tfrecord', '.tfrecord')
    filename.replace('_train.tfrecord', '.tfrecord')
    #Remove -XXXXX-of-XXXXX suffix if it exists (irrespective of the number of digits)
    p=re.compile('.tfrecord-\d+-of-\d+')
    m=p.search(filename)
    if m and len(filename)==m.span(0)[1]:
        filename = filename[0:m.span(0)[0]+len('.tfrecord')]
    return filename             
        


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
    
