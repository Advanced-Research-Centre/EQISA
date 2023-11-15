from qiskit import QuantumCircuit
import qutip as qt
from qiskit.quantum_info import Statevector, DensityMatrix, entropy, shannon_entropy

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import os
from init import Initialize

init = Initialize()

f1 = open(init.file1, 'r').readlines()
f2 = open(init.file2, 'r').readlines()
f3 = open(init.file3, 'r').readlines()
f4 = open(init.file4, 'r').readlines()
f5 = open(init.file5, 'r').readlines()
f6 = open(init.file6, 'r').readlines()

Fidelities = [float(f1[i].strip()) for i in range(len(f1))]
Circuit_Depth = [float(f2[i].strip()) for i in range(len(f2))]
Encoded_Circuit_Depth = [float(f3[i].strip()) for i in range(len(f3))]
Encoded_Compressed_Circuit_Depth = [float(f4[i].strip()) for i in range(len(f4))]
Compression_Ratio = [float(f5[i].strip()) for i in range(len(f5))]
Decompression_Times = [float(f6[i].strip()) for i in range(len(f6))]
sample_id = np.linspace(0, init.numtrials, init.numtrials)
levin_comp = np.add(Encoded_Compressed_Circuit_Depth, np.log2(Decompression_Times))
qcirc = np.empty(init.numtrials, dtype=list)
incompresible_qcirc = np.empty(init.numtrials, dtype=list)
inv_cr = np.reciprocal(Compression_Ratio)
incompressible_indices = []

for index in range(len(inv_cr)):
    if inv_cr[index] >=1:
        incompressible_indices.append(index)
        inv_cr[index] = 1

def plot_histfid():
    plt.hist(Fidelities, bins='auto')
    plt.xlabel("Fidelities")
    plt.xlim(0, 1.1)
    plt.ylabel("Number of samples")
    plt.title("Histogram of Fidelities for "+str(init.numtrials)+" samples")
    plt.savefig('Results/' + init.folder_name + '/Histogram_Fidelity.jpg', dpi=400)
    plt.close()
    return

def plot_cd_fid():
    plt.scatter(Circuit_Depth, Fidelities)
    plt.ylabel('Fidelity')
    plt.ylim(0,1.1)
    plt.xlabel('Circuit Depth')
    #plt.xlim(0,500)
    plt.savefig('Results/' + init.folder_name + '/Fidelity_v_CircuitDepth.jpg', dpi=400)
    plt.close()
    return

def plot_eccd_fid():
    plt.scatter(Encoded_Compressed_Circuit_Depth, Fidelities)
    plt.ylabel('Fidelity')
    plt.ylim(0,1.1)
    plt.xlabel('Encoded Compressed Circuit Depth')
    plt.savefig('Results/' + init.folder_name + '/Fidelity_v_ECCD.jpg', dpi=400)
    plt.close()
    return

def plot_hist_cd():
    plt.hist(Circuit_Depth, bins='auto')
    plt.xlabel("Circuit depth")
    #plt.xlim(0,500)
    plt.ylabel("Number of samples")
    plt.title("Histogram of circuit depth for "+str(init.numtrials)+" samples")
    plt.savefig('Results/' + init.folder_name + '/Histogram_CircuitDepth.jpg', dpi=400)
    plt.close()
    return

def plot_hist_ecd():
    plt.hist(Encoded_Circuit_Depth, bins='auto', label='Encoded Circuit Depth')
    plt.hist(Encoded_Compressed_Circuit_Depth, bins='auto', label='Encoded Compressed Circuit Depth')
    plt.xlabel("Encoded Circuit Depth: Before and after compression")
    plt.ylabel("Number of samples")
    plt.title("Histogram of Compressed Circuit Depth for "+str(init.numtrials)+" samples")
    plt.legend(loc='best')
    plt.savefig('Results/' + init.folder_name + '/Histogram_ECD,ECCD.jpg', dpi=400)
    plt.close()
    return

def plot_samp_eccd_ecd():
    plt.scatter(sample_id, Encoded_Compressed_Circuit_Depth, label='Encoded Compressed Circuit Depth')
    plt.scatter(sample_id, Encoded_Circuit_Depth, label='Encoded Circuit Depth')
    plt.xlabel('Sample ID')
    plt.legend(loc='best')
    plt.ylabel('Circuit Depth: before and after compressing')
    plt.savefig('Results/' + init.folder_name + '/ECD,ECCD_v_SampleID.jpg', dpi=400)
    plt.close()
    return

def plot_samp_eccd_levin():
    plt.scatter(sample_id, Encoded_Compressed_Circuit_Depth, label='Description Complexity')
    plt.scatter(sample_id, levin_comp, label='Levin Complexity')
    plt.xlabel('Sample ID')
    plt.legend(loc='best')
    plt.ylabel('Description Complexity and Levin Complexity')
    plt.savefig('Results/' + init.folder_name + '/DC,LC_v_SampleID.jpg', dpi=400)
    plt.close()
    return

def plot_cr_fid():
    plt.scatter(Compression_Ratio, Fidelities)
    plt.ylabel('Fidelity')
    plt.ylim(0,1.1)
    plt.xlabel('Compression Ratio( len(data) / len(c_data) )')
    plt.xlim(0, np.max(Compression_Ratio)+2)
    plt.savefig('Results/' + init.folder_name + '/Fidelity_v_CompressionRatio.jpg', dpi=400)
    plt.close()
    return

def plot_hist_cr():
    plt.hist(Compression_Ratio, bins='auto')
    plt.xlim(0, np.max(Compression_Ratio)+2)
    plt.xlabel("Compression Ratio( len(data) / len(c_data) )")
    plt.ylabel("Number of samples")
    plt.title("Histogram of Compression Ratio for "+str(init.numtrials)+" samples")
    plt.savefig('Results/' + init.folder_name + '/Histogram_CompressionRatio.jpg', dpi=400)
    plt.close()
    return

def plot_ecd_eccd_fid():
    plt.scatter(Encoded_Circuit_Depth, Fidelities, label='Fidelity vs Encoded Circuit Depth')
    plt.scatter(Encoded_Compressed_Circuit_Depth, Fidelities, label='Fidelity vs Encoded Compressed Circuit Depth')
    plt.ylabel('Fidelity')
    plt.ylim(0,1.1)
    plt.xlabel('Encoded Circuit Depth: Before and after Compressing')
    plt.legend(loc='best')
    plt.savefig('Results/' + init.folder_name + '/Fidelity_v_ECD,ECCD.jpg', dpi=400)
    plt.close()
    return


def val_to_cmap(val, min, max, color):
    cmap = color
    norm = mp.colors.Normalize(vmin=min, vmax=max)
    scalarMap = mp.cm.ScalarMappable(norm=norm, cmap=cmap)
    return scalarMap.to_rgba(val)


def stateQC(quantCirc):
    circ = QuantumCircuit(1)

    if quantCirc == None:
        return Statevector(circ).data

    else:
        for op in quantCirc:
            if op == '1H1 0':
                circ.h(0)
            elif op == '2T1 0':
                circ.t(0)
            elif op == '3TD1 0':
                circ.tdg(0)

        return Statevector(circ).data


def calc_entropy(qcirc):
    vN_arr = []
    shannon_arr = []
    for quantCirc in qcirc:
        circ = QuantumCircuit(1)
        for op in quantCirc:
            if op == '1H1 0':
                circ.h(0)
            elif op == '2T1 0':
                circ.t(0)
            elif op == '3TD1 0':
                circ.tdg(0)
        vN_entropy = entropy(DensityMatrix(circ), base=2)
        s_entropy = shannon_entropy(Statevector(circ).probabilities(), base=2)
        vN_arr.append(vN_entropy)
        shannon_arr.append(s_entropy)

    return vN_arr, shannon_arr


def vis_new(qcirc, param, str, color):
    b = qt.Bloch()
    b.point_marker = ['o']
    b.point_size = [20]
    samples = len(qcirc)
    for i in range(samples):
        b.add_states(qt.Qobj(stateQC(qcirc[i])), kind='point')

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    b.point_color = [val_to_cmap(param[iter], np.min(param), np.max(param), color) for iter in range(len(param))]
    b.render()
    cmap = color
    norm = mp.colors.Normalize(vmin=np.min(param), vmax=np.max(param))
    fig.colorbar(
        mp.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=ax,
        extend='both',
        spacing='proportional',
        orientation='horizontal',
        label='Parameter used for colormap: ' + str + ', Colormap used: Inferno',
    )
    fig.savefig('Results/' + init.folder_name + '/CMap_' + str + '.jpg', dpi=800)
    plt.close(fig)
    plt.savefig('Results/' + init.folder_name + '/Bloch_' + str + '.jpg', dpi=800)
    plt.close()
    return


direc = 'Data/' + init.folder_name + '/Quantum_Circuits'
for filename in os.listdir(direc):
    ind = int(filename[2:6])
    with open(os.path.join(direc, filename), 'r') as f:
        temp = f.readlines()
        vec = [temp[index].strip() for index in range(len(temp))]
        qcirc[ind] = vec
        if (ind in incompressible_indices):
            incompresible_qcirc[ind] = vec

vonNeumann_Entropy, Shannon_Entropy = calc_entropy(qcirc)

def plot_hist_vN():
    plt.hist(vonNeumann_Entropy, bins='auto')
    plt.xlabel("von-Neumann entropy")
    plt.ylabel("Number of samples")
    plt.savefig('Results/' + init.folder_name + '/von_Neumann Entropy.jpg', dpi=400)
    plt.close()
    return

def plot_hist_shannon():
    plt.hist(Shannon_Entropy, bins='auto')
    plt.xlabel("Shannon Entropy")
    plt.ylabel("Number of samples")
    plt.savefig('Results/' + init.folder_name + '/Shannon Entropy.jpg', dpi=400)
    plt.close()
    return

cr_fid99 = []
cr_fid97 = []
cr_fid93 = []

for i in range(len(Fidelities)):
    if Fidelities[i] > 0.99:
        cr_fid99.append(Compression_Ratio[i])
        continue
    elif Fidelities[i] > 0.97:
        cr_fid97.append(Compression_Ratio[i])
        continue
    elif Fidelities[i] > 0.93:
        cr_fid93.append(Compression_Ratio[i])
        continue


def plot_modif_hist_cr():
    plt.hist(Compression_Ratio, bins='auto', label='All F allowed')
    plt.hist(cr_fid99, bins='auto', label='$F > 0.99$')
    plt.hist(cr_fid97, bins='auto', label='$0.99 > F > 0.97$')
    plt.hist(cr_fid93, bins='auto', label='$0.97> F > 0.93$')
    plt.xlim(0)
    plt.xlabel("Compression Ratio( len(data) / len(c_data) )")
    plt.ylabel("Number of samples")
    plt.title("Histogram of Compression Ratio for "+str(init.numtrials)+" samples")
    plt.legend(loc='best')
    plt.savefig('Results/' + init.folder_name + '/modif_CompressionRatio.jpg', dpi=400)
    plt.close()
    return

cr_fid99_new = []
cr_fid97_new = []
cr_fid93_new = []

for i in range(len(Fidelities)):
    if Fidelities[i] > 0.99:
        cr_fid99_new.append(inv_cr[i])
        continue
    elif Fidelities[i] > 0.97:
        cr_fid97_new.append(inv_cr[i])
        continue
    elif Fidelities[i] > 0.93:
        cr_fid93_new.append(inv_cr[i])
        continue


def plot_modif_hist_cr2():
    plt.hist(inv_cr, bins='auto', label='All F allowed')
    plt.hist(cr_fid99_new, bins='auto', label='$F > 0.99$')
    plt.hist(cr_fid97_new, bins='auto', label='$0.99 > F > 0.97$')
    plt.hist(cr_fid93_new, bins='auto', label='$0.97 > F > 0.93$')
    plt.xlim(0)
    plt.xlabel("Compression Ratio( len(c_data) / len(data) )")
    plt.ylabel("Number of samples")
    plt.title("Histogram of Compression Ratio for "+str(init.numtrials)+" samples")
    plt.legend(loc='best')
    plt.savefig('Results/' + init.folder_name + '/modif_CompressionRatio2.jpg', dpi=400)
    plt.close()
    return