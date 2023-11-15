import time
import configparser
import json

import bz2

from YAQQ.dev.yaqq_nus import NovelUniversalitySearch
from init import Initialize


nsa = NovelUniversalitySearch()
init = Initialize()

autocfg = True
Config = configparser.ConfigParser()
cfg_fname = 'QART_eid-0001'
Config.read("Inputs/" + cfg_fname + ".cfg")
yaqq_cf_dcmp_gs1 = json.loads(Config['experiment']['yaqq_cf_dcmp_gs1'])
yaqq_cf_dcmp_gs2 = json.loads(Config['experiment']['yaqq_cf_dcmp_gs2'])
nsa.cnfg_dcmp(yaqq_cf_dcmp_gs1, yaqq_cf_dcmp_gs2)
dcmp_conditions = init.yaqq_cf_dcmp_gs1

def decompose(index, U):

    if init.dim == 1:
        gs, gs_gates = nsa.def_gs(init.gateset_1D)
    else:
        gs, gs_gates = nsa.def_gs(init.gateset_2D)

    pf, cd, qc = nsa.dcmp_U_gs(U, gs, gsid=0)
    qc_fname = 'qc' + f'{index:04}'

    with open('Data/' + init.folder_name + '/Quantum_Circuits/' + qc_fname + '.txt', 'w') as f:
        for i in qc:
            f.write(i.operation.label + ' ' + str(i.qubits[0].index) + '\n')
    data = open('Data/' + init.folder_name + '/Quantum_Circuits/' + qc_fname + '.txt', 'r').read().encode()
    ecd = len(data)
    c_data = bz2.compress(data, compresslevel=9)
    eccd = len(c_data)
    cr = len(data) / len(c_data)
    start = time.time()
    decomp_data = bz2.decompress(c_data)
    decomp_time = time.time() - start

    if decomp_data == data:
        return pf, cd, ecd, eccd, cr, decomp_time
    else:
        return pf, cd, ecd, eccd, cr, 1



