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
    QAbstractItemView,
    QMessageBox
)
from PyQt5.QtCore import Qt
from absl import flags

from modifiedObjectDetection.model_main_tf2 import main as trainModel
from utils.makeTFRecords import maketfRecords
from utils.utils import getNetworkList

#Default Values
downscale_targetsize=2048
split_targetsize=1080
overlap=0
eval_probability=0.15
min_score=0.15
checkpoint_every=2000

MODELFOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)),'models')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Helper for automated imaging")
        layout=QVBoxLayout()
        buttonImport=QPushButton("Import Images for Training")
        buttonImport.clicked.connect(lambda x: self.open_window(ImportWindow))
        layout.addWidget(buttonImport)
        buttonTraining = QPushButton("Perform Training")
        buttonTraining.clicked.connect(lambda x: self.open_window(TrainingWindow))
        layout.addWidget(buttonTraining)
        buttonBoard=QPushButton("See Training result in TensorBoard")
        buttonBoard.clicked.connect(lambda x: self.open_window(TensorBoardWindow))
        layout.addWidget(buttonBoard)
        
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


class ImportWindow(SubWindow):
    def __init__(self,parent):
        super().__init__(parent)
        self.setWindowTitle("Import training data")
        self.makeFrom='Darea'
        self.configs=[]
        self.tfRecord=''
        self.eval_probability=eval_probability
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
        
        l_feat=QHBoxLayout()
        l_feat.addWidget(QLabel('Name of feature (e.g. Active Zone)'))
        self.feat_edit = QLineEdit('')
        l_feat.addWidget(self.feat_edit)
        layoutMain.addLayout(l_feat)
        
        l_eval=QHBoxLayout()
        l_eval.addWidget(QLabel('Ratio of Images used for Evaluation'))
        eval_edit = QLineEdit(str(self.eval_probability))
        eval_edit.editingFinished.connect(lambda: self.changeEvalProb(eval_edit))
        l_eval.addWidget(eval_edit)
        layoutMain.addLayout(l_eval)
        
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
    def changeEvalProb(self, edit):
        try: 
            nr=float(edit.text())
        
        except:
            edit.setText(str(self.eval_probability)) 
            return
        
        if nr<0:
            pass
        elif nr<=1:
            self.eval_probability = nr
            return
        elif nr>1 and nr<=100:
            self.eval_probability = nr/100

        edit.setText(str(self.eval_probability))    
    
    def changeInputType(self,idx):
        self.listbox.clear()
        self.addButton.clicked.disconnect()
        if idx==0:
            self.makeFrom='Darea'
            self.addButton.clicked.connect(self.addConfig)
        elif idx==1:
            self.makeFrom='XML'
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
                        class_name=self.feat_edit.text(),
                        eval_probability=self.eval_probability, 
                        progressHandle=self.progress, app=app)
        shutil.rmtree(outputFolder) #Delete folder with temp .jpg images

class TrainingWindow(SubWindow):
    def __init__(self,parent):
        super().__init__(parent)
        self.networks = getNetworkList(MODELFOLDER)
        if len(self.networks) == 0:
            QMessageBox.about(self, "No networks found",
                              "Training can only be started from an existing network.\n" \
                              "Please refer to the manual!")
            self.close()
            return

        self.setWindowTitle('Perform Training')
        self.tfRecord=''
        self.tfRecordEval=''
        #Still need to write UI for following!!
        self.label_map_path = 'label_map.pbtxt'
        self.num_steps = 250000

        layoutMain = QVBoxLayout()
        
        l_tfRec=QHBoxLayout()
        l_tfRec.addWidget(QLabel('Dataset'))
        self.sel_tfRec=QPushButton("Select .tfrecord Dataset...")
        self.sel_tfRec.clicked.connect(self.selectTfrecord)
        l_tfRec.addWidget(self.sel_tfRec)
        layoutMain.addLayout(l_tfRec)
        
        
        l_continue = QHBoxLayout()
        continue_TT = 'Select the network from which to continue from\n.' 
        continue_label = QLabel('Continue Training From')
        continue_label.setToolTip(continue_TT)
        l_continue.addWidget(continue_label)
        self.continue_from = QComboBox()
        self.continue_from.addItems([n[0] for n in self.networks])
        self.continue_from.setToolTip(continue_TT)
        l_continue.addWidget(self.continue_from)
        layoutMain.addLayout(l_continue)
        
        l_trainsteps = QHBoxLayout()
        trainsteps_TT = 'Select number of training steps'
        trainsteps_label = QLabel('Number of training steps')
        trainsteps_label.setToolTip(trainsteps_TT)
        steps_edit = QLineEdit(str(self.num_steps))
        steps_edit.setToolTip(trainsteps_TT)
        steps_edit.editingFinished.connect(lambda: self.changeSteps(steps_edit))
        l_trainsteps.addWidget(trainsteps_label)
        l_trainsteps.addWidget(steps_edit)
        layoutMain.addLayout(l_trainsteps)
        
        l_model_dir = QHBoxLayout()
        nameTT = 'Choose a name for your new network'
        name_label = QLabel('New Network name:')
        name_label.setToolTip(nameTT)
        l_model_dir.addWidget(name_label)
        self.nameEdit = QLineEdit()
        self.nameEdit.setToolTip(nameTT)
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
        
    def changeSteps(self, edit):
        if edit.text().isnumeric():
            self.num_steps = int(edit.text())
        else:
            edit.setText(str(self.num_steps))
    
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
        if not self.tfRecord:
            self.progress.setText('You need to select a dataset!')
            return
        if not self.nameEdit.text():
            self.progress.setText('You have to name your network!')
            return

        self.progress.setText('Preparing...')
        app.processEvents()

        chosen_network = self.networks[self.continue_from.currentIndex()]

        model_name = self.nameEdit.text()
        model_dir = os.path.join(MODELFOLDER, chosen_network[1], model_name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        
        default_config_file = os.path.join(MODELFOLDER,chosen_network[1],
                                           'pipeline_default.config') 
        pipeline_file = os.path.join(model_dir, 'pipeline.config')
        label_map_file = os.path.join(model_dir, 'label_map.pbtxt')
        
        #TODO: get proper class_names
        self.makeLabelMap(label_map_file,classes=['PSD']) 
        
        self.makePipelineConfig(default_config_file, pipeline_file, label_map_file)
        
        self.progress.setText('Performing Training (This may take a long time up to several days)...')
        app.processEvents()
        checkpoint_every_n = min(checkpoint_every,self.num_steps )
        
        flags.FLAGS.unparse_flags() #Clear FLAGS. Is necessary when running twice.
        
        #flags will be read by trainModel even when not explicitly passed
        flags.FLAGS(['model_main_tf2.py', 
                   '--pipeline_config_path', pipeline_file,
                   '--model_dir', model_dir,
                   '--checkpoint_every_n', str(checkpoint_every_n),
                   '--checkpoints_max_to_keep', str(round(self.num_steps/checkpoint_every_n)+1)])
        trainModel()
        if self.tfRecordEval:
            # Run evaluation (Evaluation is triggered by including
            # checkpoint_dir variable)
            self.progress.setText('Performing Evaluation...')
            app.processEvents()
            flags.FLAGS.unparse_flags()
            flags.FLAGS(['model_main_tf2.py', 
                   '--pipeline_config_path', pipeline_file,
                   '--model_dir', model_dir, '--checkpoint_dir', model_dir])
            trainModel()
        self.progress.setText('Training completed!')
        app.processEvents()
        
    def makeLabelMap(self,label_map_path,classes):
        with open(label_map_path,'w') as f:
            for i,c in enumerate(classes):
                if i>0:
                    f.write('\n')
                f.write("item{\n" \
                        f"\tid: {i+1}\n" \
                        f"\tname: '{c}'\n" \
                        "}")

    def makePipelineConfig(self,default_config_file, config_file, label_map_path):
        #Reads the default_config for the selected model and
        #Changes Values to make it ready for training
        #Then saves modified pipeline.config under config_file path
        chosen_network = self.networks[self.continue_from.currentIndex()]
        checkpoint = os.path.join(MODELFOLDER,chosen_network[1],chosen_network[2])
        
        with open(default_config_file,'r') as f:
            text = f.read()
        
        # Better would be to use regex, but difficult because both identifiers are called input_path
        text = text.replace('input_path: "INPUT_PATH.tfrecord"',
                     f'input_path: "{self.tfRecord}"')
        if self.tfRecordEval:
            text = text.replace('input_path: "EVAL_PATH.tfrecord"',
                         f'input_path: "{self.tfRecordEval}"')
            
        # text = text.replace('label_map_path: "PATH_TO_LABELMAP"',
        #              f'label_map_path: "{label_map_path}"')
        # text = text.replace('fine_tune_checkpoint: "PATH_TO_CHECKPOINT"',
        #              f'fine_tune_checkpoint: "{checkpoint}"')
        

        replacements = [
                        ['total_steps: ', '(\d+)', f'{self.num_steps}'],
                        ['num_steps: ', '(\d+)', f'{self.num_steps}'],
                        ['label_map_path: ', '[^\n]*', f'"{label_map_path}"'],
                        ['fine_tune_checkpoint: ', '[^\n]*', f'"{checkpoint}"']
                        ]
        
        for to_replace, regex_code, replacement  in replacements:
            p=re.compile(to_replace+regex_code)
            m = p.search(text)
            if m:
                text = text.replace(m.group(), to_replace+replacement)
       
        # text = text.replace('total_steps: 250000',
        #              f'total_steps: {self.num_steps}')
        # text = text.replace('num_steps: 250000',
        #              f'num_steps: {self.num_steps}')
        #Warmup steps need to be less than total steps
        p = re.compile('warmup_steps: (\d+)')
        m = p.search(text)
        if m and int(m.group(1)) > self.num_steps:
            text = text.replace(m.group(),
                                f'warmup_steps: {int(self.num_steps/2)}')
        
        with open(config_file,'w') as f:
            f.write(text)

class TensorBoardWindow(SubWindow):
    def __init__(self,parent):
        from tensorboard import program
        super().__init__(parent)

        tb = program.TensorBoard()
        tb.configure(argv = [None, '--logdir', MODELFOLDER])
        url = tb.launch()
        
        link = QLabel('<a href="'+url+'">Click to view results!</a>')
        link.setOpenExternalLinks(True)

        self.setCentralWidget(link)
        

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
    
