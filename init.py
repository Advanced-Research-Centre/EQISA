import datetime
import os
import configparser
import json
from datetime import date

from YAQQ.dev.yaqq_ds import GenerateDataSet, VisualizeDataSet, ResultsPlotSave
from YAQQ.dev.yaqq_nus import NovelUniversalitySearch


gds = GenerateDataSet()
vds = VisualizeDataSet()
nsa = NovelUniversalitySearch()
rps = ResultsPlotSave()

class Initialize:
    # Dimension of the unitary(no. of qubits).
    dim = 1
    # Number of sample unitaries
    numtrials = 10000
    autocfg = True
    Config = configparser.ConfigParser()
    cfg_fname = 'QART_eid-0001'
    Config.read("Inputs/" + cfg_fname + ".cfg")
    yaqq_cf_dcmp_gs1 = json.loads(Config['experiment']['yaqq_cf_dcmp_gs1'])
    yaqq_cf_dcmp_gs2 = json.loads(Config['experiment']['yaqq_cf_dcmp_gs2'])
    nsa.cnfg_dcmp(yaqq_cf_dcmp_gs1, yaqq_cf_dcmp_gs2)
    gateset_1D = 'H1,T1,TD1'.split(',')
    gateset_2D = 'H1,T1,TD1,CX2'.split(',')

    datetime_label = str(date.today()) + '_' + str(datetime.datetime.now().strftime("%H.%M"))
    folder_name = 'Run_' + datetime_label
    folder = os.path.join('/Users/sibasishmishra/Desktop/TU Delft year-2/Thesis/Codes/Data', folder_name)
    folder_output = os.path.join('/Users/sibasishmishra/Desktop/TU Delft year-2/Thesis/Codes/Results', folder_name)
    file1_name = 'List_Fidelity' + str(yaqq_cf_dcmp_gs1) + str(numtrials) + '.txt'.format(folder_name)
    file2_name = 'List_Circuit_Depth' + str(yaqq_cf_dcmp_gs1) + str(numtrials) + '.txt'.format(folder_name)
    # ECD(Encoded Circuit Depth): representative of the Circuit Complexity
    file3_name = 'List_Encoded_Circuit_Depth' + str(yaqq_cf_dcmp_gs1) + str(numtrials) + '.txt'.format(folder_name)
    # ECCD(Encoded Compressed Circuit Depth): representative of the Description Complexity
    file4_name = 'List_Encoded_Compressed_Circuit_Depth' + str(yaqq_cf_dcmp_gs1) + str(numtrials) + '.txt'.format(folder_name)
    # CR(Compression Ratio): Compressibility maybe representative of uncomplexity?
    file5_name = 'List_Compression_Ratio' + str(yaqq_cf_dcmp_gs1) + str(numtrials) + '.txt'.format(folder_name)
    # Decompression Times: Can be used for calculating Levin Complexity
    file6_name = 'List_Decompression_Times' + str(yaqq_cf_dcmp_gs1) + str(numtrials) + '.txt'.format(folder_name)
    file1 = os.path.join(folder, file1_name)
    file2 = os.path.join(folder, file2_name)
    file3 = os.path.join(folder, file3_name)
    file4 = os.path.join(folder, file4_name)
    file5 = os.path.join(folder, file5_name)
    file6 = os.path.join(folder, file6_name)

    def create_Files(self):
        os.makedirs(self.folder)
        subfolder = '/Users/sibasishmishra/Desktop/TU Delft year-2/Thesis/Codes/Data/' + self.folder_name
        os.makedirs(os.path.join(subfolder, 'Quantum_Circuits'))
        os.makedirs(os.path.join(subfolder, 'Unitaries'))
        os.makedirs(self.folder_output)
        print("Initiation Checkpoint: Files Created Successfully!")
        return



