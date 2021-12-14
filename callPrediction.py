# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:20:26 2020

@author: David Kleindienst
"""
import sys
import os
import predict
from utils.utils import getNetworkList

from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLineEdit,
    QListWidget,
    QFileDialog,
    QMessageBox
)


class MainWindow(QMainWindow):
    def __init__(self,network_list,nav=None):
        super().__init__()
        self.setWindowTitle('Select Network')
        layout=QVBoxLayout()
        self.nav = nav
        self.network_list = network_list
        self.idx = None
        self.waitForImages = False
        
        if not nav:
            l1 = QHBoxLayout()
            self.filelabel = QLabel('Select navigator (.nav)')
            l1.addWidget(self.filelabel)
            fileButton = QPushButton('Select File...')
            fileButton.clicked.connect(self.pickFile)
            l1.addWidget(fileButton)
            layout.addLayout(l1)
        
        if len(network_list) > 1:
            l2 = QHBoxLayout()
            l2.addWidget(QLabel('Select Network'))
            self.networkCombo = QComboBox()
            for n in network_list:
                self.networkCombo.addItem(n[0])
            
            l2.addWidget(self.networkCombo)
            layout.addLayout(l2)
        else:
            #Only one network is there. Use that and don't ask user
            self.idx = 0 
        
        acceptButton = QPushButton('Ok')
        acceptButton.clicked.connect(self.accept)
        cancelButton = QPushButton('Cancel')
        cancelButton.clicked.connect(self.cancel)
        
        if not nav:
            #Option to start this before imaging finished
            waitCheckbox = QCheckBox('Run in parallel to imaging')
            waitCheckbox.setToolTip('Pick this option if you start the prediction' \
                                    ' while the microscope is still imaging\n' \
                                    'The prediction will then run in parallel to the imaging\n'
                                    'If imaging is already finished, picking this option will slow the process')
            waitCheckbox.stateChanged.connect(lambda: self.pickWait(waitCheckbox))
            layout.addWidget(waitCheckbox)
            
        
        l3 = QHBoxLayout()
        l3.addWidget(acceptButton)
        l3.addWidget(cancelButton)
        layout.addLayout(l3)
        
        
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        
    def pickFile(self):
        path = QFileDialog.getOpenFileName(self,'Select File',
                            filter='Navigator file (*.nav);; Image document file (*.idoc)')
        print(path)
        if path:
            path = path[0]
            self.nav = path
            self.filelabel.setText('Selected File: ' + path)
        
    def cancel(self):
        self.close()
    
    def pickWait(self,checkbox):
        self.waitForImages = checkbox.isChecked()
        print(self.waitForImages)
        
        
    def accept(self):
        self.hide()
        
        if not self.nav:
            
            QMessageBox.about(self, "No file selected", "You have to select a .nav or .idoc file")
            return
            
        
        if self.idx is None:
            #get it from combobox
            self.idx = self.networkCombo.currentIndex()
        startPrediction(self.nav,self.network_list[self.idx],self.waitForImages)
        self.close()
        
        
def startPrediction(nav, network, wait=False):
    print(f'Using Network {network[0]}')
    pipeline_config = os.path.join(modelfolder, network[1], 'pipeline_default.config')
    checkpoint_name = os.path.join(modelfolder, network[1], network[2])
    
    if wait:
        predict.main(['--pipeline_path', pipeline_config,
                      '--checkpoint_path', checkpoint_name,
                      '--navfile', nav, '--wait'])

    else:
        predict.main(['--pipeline_path', pipeline_config,
                      '--checkpoint_path', checkpoint_name,
                      '--navfile', nav])

    

if len(sys.argv)>1:
    nav=sys.argv[1]
else:
    nav = None
    
scriptDir = os.path.dirname(os.path.realpath(__file__))

modelfolder = os.path.join(scriptDir, 'models')

network_list = getNetworkList(modelfolder)
        
        
if len(network_list)==0:
    raise Exception('No saved networks found. Please refer to the manual.')
elif len(network_list)==1 and nav:
    #Only 1 network found. Use that one.
    startPrediction(nav,network_list[0])
else:
    #Make userinterface
    app = QApplication(sys.argv)
    w = MainWindow(network_list,nav)
    w.show()
    app.exec()
    
#     print('The following networks exist:')
    
#     for i, n in enumerate(network_list):
#         print(str(i)+': '+n[0])
#     print('\n')
#     print('Please enter the number of the network you would like to select, or write anything else to cancel:')
#     text=input('>> ')

# if text.isnumeric() and int(text)<len(network_list):
#     idx = int(text)
#     startPrediction(nav, network_list[idx],idx)
# else:
#     print('Invalid input. Aborting...')

        

#modelDir=os.path.join(scriptDir,'models/efficientdet_d2')
#predict.main(['--pipeline_path', os.path.join(modelDir, 'pipeline_default.config'),
 #             '--checkpoint_path', os.path.join(modelDir,'ckpt-51'),
  #            '--navfile', nav])

