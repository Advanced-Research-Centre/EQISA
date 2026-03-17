import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogLocator, LogFormatterMathtext
import numpy as np
import pandas as pd
import csv

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.size": 14}
)

class Plotter:

    def plot(self, means, stds):
        N = len(means[0])
        ind = np.arange(N)
        width = 0.2
        plt.figure(figsize=(12, 6))
        counter = 0
        for i in range(len(means)):
            plt.bar(ind + counter, means[i].values, width, alpha=0.9, label='qubit_ID = ' + str(i))
            plt.errorbar(ind + counter, means[i].values, yerr=stds[i].values, fmt='none', capsize=1.0, color='#003f5c', alpha=0.85,
                         linewidth=0.6)
            counter += width
        plt.xlabel('Elements of S-K Basis')
        plt.ylabel('Average # of Occurrence')
        plt.title('Average Values of Occurrence for qubit_IDs = 0 and 1', fontsize=14)
        plt.xticks(ind + width / 3, list(means[0].index))
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()

    def calc_plot_avg(self, dim):
        # folder = 'records/'
        # filename = str(dim)+'q_200_qsd+skt.csv'
        folder = 'QART/'
        filename = 'SK-Basis.csv'
        df = pd.read_csv(folder + filename)
        qubit_IDs = np.arange(0, dim, 1)
        print(qubit_IDs)
        # means = []
        # stds = []
        # for ind in qubit_IDs:
        #     means.append(df[df['qubit_ID'] == ind].mean()[1:])
        #     stds.append(df[df['qubit_ID'] == ind].std()[1:])
        means = df.mean()
        stds = df.std()
        print(means)
        print(stds)
        plt.figure(figsize=(12, 6))
        plt.xticks(rotation=90)
        plt.bar(list(df.keys()), means, color='skyblue')
        plt.errorbar(list(df.keys()), means, yerr=stds, fmt='none', color='black', capsize=1, linewidth=0.6)
        plt.show()
        # self.plot(means, stds)

    def plot_fid_depth(self, dim):
        folder = 'records/' + str(dim) + 'q/'
        filename = str(dim) + 'q_cd_v_depth.csv'
        read = pd.read_csv(folder + filename)
        df = read[:6]
        df2 = read[6:9]
        # plt.figure(figsize=(10, 7))
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.15)
        plt.plot(df['depth'], df['mean fidelity'], marker='o', color='#2ca02c', alpha=0.5, zorder=1,
                 label='recur SK-Decomp = 4')
        plt.errorbar(df['depth'], df['mean fidelity'], yerr=df['std fidelity'],
                     color='#98df8a', capsize=2, fmt='none', zorder=0)
        plt.scatter(df2['depth'], df2['mean fidelity'], marker='D', color='#9467bd', alpha=0.5, zorder=1,
                    label='recur SK-Decomp = 3')
        plt.errorbar(df2['depth'], df2['mean fidelity'], yerr=df2['std fidelity'],
                     color='#c5b0d5', capsize=2, fmt='none', zorder=0)
        plt.xlabel(r'Depth of SK-Basis $d$', fontsize=24, labelpad=15)
        plt.ylabel("Fidelity of Decomposition", fontsize=24, labelpad=15)
        plt.legend(loc='best', fontsize=14)
        # plt.title("Mean Fidelity vs Depth for 200 "+str(dim)+"-qubit unitaries", fontsize=14)
        plt.savefig('Plots/article/fid_depth_' + str(dim) + 'q.png')
        plt.show()

    def plot_circ_depth_recur(self, dim):
        folder = 'records/' + str(dim) + 'q/'
        filename = str(dim) + 'q_cd_v_recur.csv'
        read = pd.read_csv(folder + filename)
        df = read[:6]
        df2 = read[6:10]
        df3 = read[10:14]
        # plt.figure(figsize=(10, 7))
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.15)
        plt.scatter(df['recur'], df['mean circ_depth']/1000, marker='o', color='#2ca02c', alpha=0.7, zorder=1,
                    label='Depth of SK Basis = 4')
        plt.errorbar(df['recur'], df['mean circ_depth']/1000, yerr=df['std circ_depth']/1000,
                     color='#98df8a', capsize=2, fmt='none', zorder=0)
        plt.scatter(df2['recur'], df2['mean circ_depth']/1000, marker='D', color='#1f77b4', alpha=0.7, zorder=1,
                    label='Depth SK Basis = 5')
        plt.errorbar(df2['recur'], df2['mean circ_depth'] / 1000, yerr=df2['std circ_depth'] / 1000,
                     color='#aec7e8', capsize=2, fmt='none', zorder=0)
        plt.scatter(df3['recur'], df3['mean circ_depth'] / 1000, marker='s', color='#ff7f0e', alpha=0.7, zorder=1,
                    label='Depth SK Basis = 6')
        plt.errorbar(df3['recur'], df3['mean circ_depth'] / 1000, yerr=df3['std circ_depth'] / 1000,
                     color='#ffbb78', capsize=2, fmt='none', zorder=0)
        plt.xlabel(r'Degree of Recursion $n$', fontsize=24, labelpad=15)
        plt.ylabel("Circuit Depth (in 1000s)", fontsize=24, labelpad=15)
        plt.legend(loc='best', fontsize=14)
        # plt.title("Mean Circuit Depth vs Degree of Recursion for 200 "+str(dim)+"-qubit unitaries", fontsize=14)
        plt.savefig('Plots/article/cd_recur_' + str(dim) + 'q.png')
        plt.show()

    def plot_circ_depth_depth(self, dim):
        folder = 'records/' + str(dim) + 'q/'
        filename = str(dim) + 'q_cd_v_depth.csv'
        read = pd.read_csv(folder + filename)
        df = read[:6]
        df2 = read[6:9]
        # df3 = read[13:]
        # plt.figure(figsize=(10, 7))
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.15)
        plt.scatter(df['depth'], df['mean circ_depth']/1000, marker='o', color='#2ca02c', alpha=1.0, zorder=1,
                 label='recur SK-Decomp = 4')
        plt.errorbar(df['depth'], df['mean circ_depth']/1000, yerr=df['std circ_depth']/1000,
                     color='#98df8a', capsize=2, fmt='none', zorder=0)
        plt.scatter(df2['depth'], df2['mean circ_depth'] / 1000, marker='D', color='#9467bd', alpha=1.0, zorder=1,
                    label='recur SK-Decomp = 3')
        plt.errorbar(df2['depth'], df2['mean circ_depth'] / 1000, yerr=df2['std circ_depth'] / 1000,
                     color='#c5b0d5', capsize=2, fmt='none', zorder=0)
        plt.xlabel(r'Depth of SK-Basis $d$', fontsize=24, labelpad=15)
        plt.ylabel("Circuit Depth (in 1000s)", fontsize=24, labelpad=15)
        plt.legend(loc='best', fontsize=14)
        # plt.title("Mean Circuit Depth vs Depth of SK Basis for 200 "+str(dim)+"-qubit unitaries", fontsize=14)
        plt.savefig('Plots/article/cd_depth_' + str(dim) + 'q.png')
        plt.show()

    def plot_fid_recur(self, dim):
        folder = 'records/' + str(dim) + 'q/'
        filename = str(dim) + 'q_cd_v_recur.csv'
        read = pd.read_csv(folder + filename)
        df = read[:6]
        df2 = read[6:10]
        df3 = read[10:14]
        # plt.figure(figsize=(10, 7))
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.15)
        plt.plot(df['recur'], df['mean fidelity'], marker='o', color='#2ca02c', alpha=0.5, zorder=1,
                 label='Depth of SK Basis = 4')
        plt.errorbar(df['recur'], df['mean fidelity'], yerr=df['std fidelity'],
                     color='#98df8a', capsize=2, fmt='none', zorder=0)
        plt.scatter(df2['recur'], df2['mean fidelity'], marker='D', color='#9467bd', alpha=0.5, zorder=1,
                 label='Depth of SK Basis = 5')
        plt.errorbar(df2['recur'], df2['mean fidelity'], yerr=df2['std fidelity'],
                     color='#c5b0d5', capsize=2, fmt='none', zorder=0)
        plt.scatter(df3['recur'], df3['mean fidelity'], marker='D', color='#1f77b4', alpha=0.5, zorder=1,
                 label='Depth of SK Basis = 6')
        plt.errorbar(df3['recur'], df3['mean fidelity'], yerr=df3['std fidelity'],
                     color='#aec7e8', capsize=2, fmt='none', zorder=0)

        plt.xlabel(r'Degree of Recursion $n$', fontsize=24, labelpad=15)
        plt.ylabel("Fidelity of Decomposition", fontsize=24, labelpad=15)
        # plt.title("Mean Fidelity vs Degree of Recursion for 200 "+str(dim)+"-qubit unitaries", fontsize=14)
        plt.legend(loc='best', fontsize=14)
        plt.savefig('Plots/article/fid_recur_' + str(dim) + 'q.png')
        plt.show()

    def plot_length_comparison(self, dim):
        folder = 'records/' + str(dim) + 'q/'
        filename = str(dim) + 'q_lengths.csv'
        df = pd.read_csv(folder + filename)  # [3:9]
        # plt.figure(figsize=(10, 7))
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.15)
        df1 = df[:6]
        df2 = df[6:10]
        df3 = df[10:]
        # plt.axhline(y=1.0, linestyle='--', color='#999999', linewidth=2, label='v0: Binary Encoding')
        plt.scatter(df1['recur'], df1['mean l0']/1000, marker='s', color='#00008B', label='Binary encoded, d=4', alpha=0.5, zorder=1)
        # plt.errorbar(df1['recur'], df1['mean l0']/1000, yerr=df1['std l0']/1000,
        #              color='#aec7e8', capsize=2, fmt='none', alpha=0.7, zorder=0)
        plt.scatter(df1['recur'], df1['mean l1']/1000, marker='^', color='#006400', label='Huffman v1, d=4', alpha=0.5, zorder=1)
        # plt.errorbar(df1['recur'], df1['mean l1']/1000, yerr=df1['std l1']/1000,
        #              color='#c7e9c0', capsize=2, fmt='none', alpha=0.7, zorder=0)
        plt.scatter(df1['recur'], df1['mean l2']/1000, marker='x', color='#DC143C', label='Huffman v2, d=4', alpha=0.5, zorder=2)
        # plt.errorbar(df1['recur'], df1['mean l2']/1000, yerr=df1['std l2']/1000,
        #              color='#c5b0d5', capsize=2, fmt='none', alpha=0.7, zorder=0)
        plt.scatter(df1['recur'], df1['mean l3']/1000, marker='o', color='#708090', label='Huffman v3, d=4', alpha=0.5, zorder=1)
        # plt.errorbar(df1['recur'], df1['mean l3']/1000, yerr=df1['std l3']/1000,
        #              color='#ffbb78', capsize=2, fmt='none', alpha=0.7, zorder=0)

        # plt.plot(df2['recur'], df2['mean l0']/1000, marker='s', color='#4169E1', ms=2, label='Binary encoded, d=5', alpha=0.3, zorder=0)
        # plt.errorbar(df2['recur'], df2['mean l0']/1000, yerr=df2['std l0']/1000,
        #              color='#ffbb78', capsize=2, fmt='none', alpha=1.0, zorder=0)
        # plt.plot(df2['recur'], df2['mean l1']/1000, marker='^', color='#2E8B57', ms=2, label='Huffman v1, d=5', alpha=0.3, zorder=0)
        # plt.errorbar(df2['recur'], df2['mean l1']/1000, yerr=df2['std l1']/1000,
        #              color='#c49c94', capsize=2, fmt='none', alpha=1.0, zorder=0)
        # plt.plot(df2['recur'], df2['mean l2']/1000, marker='x', color='#FF0000', ms=2, label='Huffman v2, d=5', alpha=0.3, zorder=0)
        # plt.errorbar(df2['recur'], df2['mean l2']/1000, yerr=df2['std l2']/1000,
        #              color='#f7b6d2', capsize=2, fmt='none', alpha=1.0, zorder=0)
        # plt.plot(df2['recur'], df2['mean l3'] / 1000, marker='o', color='#696969', ms=2, label='Huffman v3, d=5', alpha=0.3, zorder=0)
        # plt.errorbar(df2['recur'], df2['mean l3'] / 1000, yerr=df2['std l3'] / 1000,
        #              color='#f7b6d2', capsize=2, fmt='none', alpha=1.0, zorder=0)

        # plt.plot(df3['recur'], df3['mean l0']/1000, marker='s', color='#00BFFF', ms=2, label='Binary encoded, d=6', alpha=0.3, zorder=0)
        # plt.errorbar(df3['recur'], df3['mean l0']/1000, yerr=df3['std l0']/1000,
        #              color='#ffcccb', capsize=2, fmt='none', alpha=1.0, zorder=0)
        # plt.plot(df3['recur'], df3['mean l1']/1000, marker='^', color='#32CD32', ms=2, label='Huffman v1, d=6', alpha=0.3, zorder=0)
        # plt.errorbar(df3['recur'], df3['mean l1']/1000, yerr=df3['std l1']/1000,
        #              color='#c1ffc1', capsize=2, fmt='none', alpha=1.0, zorder=0)
        # plt.plot(df3['recur'], df3['mean l2']/1000, marker='x', color='#FF6347', ms=2, label='Huffman v2, d=6', alpha=0.3, zorder=0)
        # plt.errorbar(df3['recur'], df3['mean l2']/1000, yerr=df3['std l2']/1000,
        #              color='#ffec8b', capsize=2, fmt='none', alpha=1.0, zorder=0)
        # plt.plot(df3['recur'], df3['mean l3'] / 1000, marker='o', color='#2F4F4F', ms=2, label='Huffman v3, d=5', alpha=0.3, zorder=0)
        # plt.errorbar(df3['recur'], df3['mean l3'] / 1000, yerr=df3['std l3'] / 1000,
        #              color='#f7b6d2', capsize=2, fmt='none', alpha=1.0, zorder=0)

        plt.xlabel(r'Degree of Recursion $n$ for SKD', fontsize=24, labelpad=15)
        plt.ylabel("Encoded bit-length (in 1000s)", fontsize=24, labelpad=15)
        # plt.title(
        #     r'Trend of Average Encoded bit-length vs Degree of Recursion of SKT for ' + str(
        #         dim) + 'q systems', fontsize=16)
        plt.legend(loc='best', fontsize=14)
        plt.show()

    def plot_cr_comparison(selfs, dim):
        folder = 'records/' + str(dim) + 'q/'
        # filename = str(dim) + 'q_compression.csv'
        filename = str(dim) + 'q_cf_new.csv'
        df = pd.read_csv(folder + filename) # [3:9]
        # plt.figure(figsize=(13, 8))
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
        # df1 = df[:6]
        df3 = df[:4]
        df2 = df[4:8]
        df1 = df[8:]
        # df2 = df[6:10]
        # df3 = df[10:]
        plt.axhline(y=1.0, linestyle='--', color='#999999', linewidth=2, label='v0: Binary Encoding')
        plt.plot(df1['recur'], df1['mean huff1'], '-', marker='o', color='#00008B', ms=2, label='Huffman v1, d=4', alpha=0.7, zorder=1)  # dark blue #00008B
        # plt.errorbar(df1['recur'], df1['mean huff1'], yerr=df1['std huff1'],
        #              color='#FF6347', capsize=2, fmt='none', alpha=1.0, zorder=0)  # turquoise #40E0D0
        plt.plot(df1['recur'], df1['mean huff2'], '-', marker='^', color='#4169E1', ms=2, label='Huffman v2, d=4', alpha=0.7, zorder=1)  # royal blue #4169E1
        # plt.errorbar(df1['recur'], df1['mean huff2'], yerr=df1['std huff2'],
        #              color='#87CEFA', capsize=2, fmt='none', alpha=1.0, zorder=0)  # cyan #00FFFF
        plt.plot(df1['recur'], df1['mean huff3'], '-', marker='s', color='#00BFFF', ms=2, label='Huffman v3, d=4', alpha=0.7, zorder=1)  # deep sky blue #00BFFF
        # plt.errorbar(df1['recur'], df1['mean huff3'], yerr=df1['std huff3'],
        #              color='#90EE90', capsize=2, fmt='none', alpha=1.0, zorder=0)  # aquamarine #7FFFD4

        plt.plot(df2['recur'], df2['mean huff1'], '--', marker='o', color='#006400', ms=2, label='Huffman v1, d=5', alpha=0.7, zorder=1)  # dark green
        # plt.errorbar(df2['recur'], df2['mean huff1'], yerr=df2['std huff1'],
        #              color='#32CD32', capsize=2, fmt='none', alpha=1.0, zorder=0)  # lime green
        plt.plot(df2['recur'], df2['mean huff2'], '--', marker='^', color='#2E8B57', ms=2, label='Huffman v2, d=5', alpha=0.7, zorder=1)  # sea green
        # plt.errorbar(df2['recur'], df2['mean huff2'], yerr=df2['std huff2'],
        #              color='#7CFC00', capsize=2, fmt='none', alpha=1.0, zorder=0)  # lawn green
        plt.plot(df2['recur'], df2['mean huff3'], '--', marker='s', color='#32CD32', ms=2, label='Huffman v3, d=5', alpha=0.7, zorder=1)  # olive green
        # plt.errorbar(df2['recur'], df2['mean huff3'], yerr=df2['std huff3'],
        #              color='#98FB98', capsize=2, fmt='none', alpha=1.0, zorder=0) # pale green

        plt.plot(df3['recur'], df3['mean huff1'], ':', marker='o', color='#DC143C', ms=2, label='Huffman v1, d=6', alpha=0.7, zorder=1)  # crimson
        # plt.errorbar(df3['recur'], df3['mean huff1'], yerr=df3['std huff1'],
        #              color='#FF7F50', capsize=2, fmt='none', alpha=1.0, zorder=0)  # coral
        plt.plot(df3['recur'], df3['mean huff2'], ':', marker='^', color='#FF0000', ms=2, label='Huffman v2, d=6', alpha=0.7, zorder=1)  # red
        # plt.errorbar(df3['recur'], df3['mean huff2'], yerr=df3['std huff2'],
        #              color='#FA8072', capsize=2, fmt='none', alpha=1.0, zorder=0)  # salmon
        plt.plot(df3['recur'], df3['mean huff3'], ':', marker='s', color='#FF6347', ms=2, label='Huffman v3, d=6', alpha=0.7, zorder=1)  # tomato
        # plt.errorbar(df3['recur'], df3['mean huff3'], yerr=df3['std huff3'],
        #              color='#CD853F', capsize=2, fmt='none', alpha=1.0, zorder=0)  # peru

        plt.xlabel("Degree of Recursion for SKT", fontsize=24, labelpad=15)
        plt.ylabel("Mean Compression Factor", fontsize=24, labelpad=15)
        # plt.title(
        #     r'Trend of Compression Factor $\frac{\text{len(huff encoded)}}{\text{len(bin encoded)}}$ vs Degree of Recursion of SKT for ' + str(dim) + 'q systems', fontsize=16)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        # plt.legend(loc='right', fontsize=14)
        plt.savefig('Plots/article/cr_random_' + str(dim) + 'q.png', dpi=400)
        plt.show()
# , ms=2, mfc='black', mec='black'

    def plot_avg_cr(self):
        direc = 'records/random_trend/huff_records.csv'
        df = pd.read_csv(direc)
        dims = [2, 3, 4, 5, 6]
        cr1_avg, cr1_std = [], []
        cr2_avg, cr2_std = [], []
        cr3_avg, cr3_std = [], []
        cd_avg, cd_std = [], []
        fid_avg, fid_std = [], []
        for dim in dims:
            df_modif = df[df['size'] == dim]
            fid_avg.append(df_modif['fidelity'].mean())
            fid_std.append(df_modif['fidelity'].std())
            cd_avg.append(df_modif['circ_depth'].mean())
            cd_std.append(df_modif['circ_depth'].std())
            cr1_avg.append(df_modif['cr huff1'].mean())
            cr1_std.append(df_modif['cr huff1'].std())
            cr2_avg.append(df_modif['cr huff2'].mean())
            cr2_std.append(df_modif['cr huff2'].std())
            cr3_avg.append(df_modif['cr huff3'].mean())
            cr3_std.append(df_modif['cr huff3'].std())
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
        plt.plot(dims, cr1_avg, '-o', label='Huff v1')
        plt.errorbar(dims, cr1_avg, yerr=cr1_std, capsize=2, fmt='none')
        plt.plot(dims, cr2_avg, '-o', label='Huff v2')
        plt.errorbar(dims, cr2_avg, yerr=cr2_std, capsize=2, fmt='none')
        plt.plot(dims, cr3_avg, '-o', label='Huff v3')
        plt.errorbar(dims, cr3_avg, yerr=cr3_std, capsize=2, fmt='none')
        plt.xlabel('Dimension', fontsize=24, labelpad=15)
        plt.legend(loc='best', fontsize=14)
        plt.ylabel('Mean compression factor', fontsize=24, labelpad=15)
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
        plt.plot(dims, cd_avg, '-o')
        plt.errorbar(dims, cd_avg, yerr= cd_std, capsize=2, fmt='none')
        plt.xlabel('Dimension', fontsize=24, labelpad=15)
        plt.ylabel('Mean circuit depth', fontsize=24, labelpad=15)
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
        plt.plot(dims, fid_avg, '-o')
        plt.errorbar(dims, fid_avg, yerr=fid_std, capsize=2, fmt='none')
        plt.xlabel('Dimension', fontsize=24, labelpad=15)
        plt.ylabel('Mean process fidelity', fontsize=24, labelpad=15)
        plt.show()

    def plot_avg_cr_bench(self):
        direc2 = 'records/benchmarks/benchmarks_huff_records.csv'
        df = pd.read_csv(direc2)
        print(list(df['benchmark']))
        dims = [2, 3, 4, 5, 6]
        cr1_avg, cr1_std = [], []
        cr2_avg, cr2_std = [], []
        cr3_avg, cr3_std = [], []
        cd_avg, cd_std = [], []
        fid_avg, fid_std = [], []
        for dim in dims:
            df_modif = df[df['size'] == dim]
            fid_avg.append(df_modif['fidelity'].mean())
            fid_std.append(df_modif['fidelity'].std())
            cd_avg.append(df_modif['circ_depth'].mean())
            cd_std.append(df_modif['circ_depth'].std())
            cr1_avg.append(df_modif['huff1'].mean())
            cr1_std.append(df_modif['huff1'].std())
            cr2_avg.append(df_modif['huff2'].mean())
            cr2_std.append(df_modif['huff2'].std())
            cr3_avg.append(df_modif['huff3'].mean())
            cr3_std.append(df_modif['huff3'].std())
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
        plt.plot(dims, cr1_avg, '-o', label='Huff v1')
        plt.errorbar(dims, cr1_avg, yerr=cr1_std, capsize=2, fmt='none')
        plt.plot(dims, cr2_avg, '-o', label='Huff v2')
        plt.errorbar(dims, cr2_avg, yerr=cr2_std, capsize=2, fmt='none')
        plt.plot(dims, cr3_avg, '-o', label='Huff v3')
        plt.errorbar(dims, cr3_avg, yerr=cr3_std, capsize=2, fmt='none')
        plt.xlabel('Dimension', fontsize=24, labelpad=15)
        plt.legend(loc='best', fontsize=14)
        plt.ylabel('Mean compression factor', fontsize=24, labelpad=15)
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
        plt.plot(dims, cd_avg, '-o')
        plt.errorbar(dims, cd_avg, yerr=cd_std, capsize=2, fmt='none')
        plt.xlabel('Dimension', fontsize=24, labelpad=15)
        plt.ylabel('Mean circuit depth', fontsize=24, labelpad=15)
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95)
        plt.plot(dims, fid_avg, '-o')
        plt.errorbar(dims, fid_avg, yerr=fid_std, capsize=2, fmt='none')
        plt.xlabel('Dimension', fontsize=24, labelpad=15)
        plt.ylabel('Mean process fidelity', fontsize=24, labelpad=15)
        plt.show()

    def plot_program_size(self):
        direc = 'records/benchmarks/benchmark_new_huff.csv'
        direc2 = 'records/random_trend/huff_records_new.csv'
        df = pd.read_csv(direc, delimiter=';')
        print(df)
        # Sizes you mentioned in your dataset
        sizes = range(2, 7)

        # Columns to analyze
        columns = ['sumbits_qasm', 'sumbits binary', 'sumbits huff1', 'sumbits huff2', 'sumbits huff3']
        bz2_columns = ['sumbits_bz2_qasm', 'sumbits bz2+bin', 'sumbits bz2+huff1', 'sumbits bz2+huff2',
                       'sumbits bz2+huff3']

        # Check if all expected columns exist in the DataFrame
        missing_columns = [col for col in columns + bz2_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the DataFrame: {missing_columns}")

        # Initialize a dictionary to store dataframes for each size
        size_dfs = {}

        # Split the dataframe based on the 'size' column
        for size in sizes:
            size_dfs[size] = df[df['size'] == size]

        # Initialize lists to hold the mean and std deviation data
        means1 = []
        std_devs1 = []
        means2 = []
        std_devs2 = []


        # Extract mean and std for each column and each size
        for size in sizes:
            means1.append(size_dfs[size][columns].mean() / 1e+6)  # Convert bits to megabits
            std_devs1.append(size_dfs[size][columns].std() / 1e+6)  # Convert bits to megabits
            means2.append(size_dfs[size][bz2_columns].mean() / 1e+6)  # Convert bits to megabits
            std_devs2.append(size_dfs[size][bz2_columns].std() / 1e+6)  # Convert bits to megabits

        # Creating the bar plots
        fig, ax = plt.subplots(figsize=(17, 8))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.3, right=0.95)
        # Setting positions for the bars
        bar_width = 0.12
        # index = np.arange(len(sizes)) * (len(columns) + 2) * bar_width  # Maintain original spacing between different sizes
        index = np.arange(len(sizes))
        # print(index)

        labels = ["QASM", "Binary v0", "Huffman v1", "Huffman v2", "Huffman v3"]
        colors = ['#0047AB', '#097969', '#808080', '#E97451', '#9F2B68']
        colors2 = ['#6495ED', '#50C878', '#C0C0C0', '#E5AA70', '#DA70D6']
        # Plot first set of columns
        for i, col in enumerate(columns):
            means_col = [mean[col] for mean in means1]
            std_devs_col = [std[col] for std in std_devs1]
            adjusted_index = -0.1 + index + i * 0.05  # 0.03 is the gap between bars of the same size
            ax.bar(adjusted_index + i * bar_width, means_col, bar_width, yerr=std_devs_col,
                   capsize=2, label=labels[i % len(labels)], alpha=0.9, edgecolor='black', color=colors[i % len(colors)])

        # Hatches for second set of bars
        hatches = ['/', '\\', "O", ".", "o"]
        overlay_offset = 0.03
        labels = ["bzip2 on QASM", "bzip2 on v0: binary", "bzip2 on Huffman v1", "bzip2 on Huffman v2", "bzip2 on Huffman v3"]
        # Plot second set of columns superimposed with lower alpha
        for i, col in enumerate(bz2_columns):
            means_col = [mean[col] for mean in means2]
            std_devs_col = [std[col] for std in std_devs2]
            adjusted_index = -0.1 + index + overlay_offset + i * bar_width + i * 0.05  # Applying the same gap adjustment
            ax.bar(adjusted_index, means_col, bar_width, yerr=std_devs_col, capsize=2,
                   label=labels[i % len(labels)], alpha=0.9, hatch=hatches[i % len(hatches)], edgecolor='black', color=colors2[i % len(colors2)])

        # Set the y-axis to a logarithmic scale with base 2
        ax.set_yscale('log', base=2)
        ax.get_yaxis().set_major_formatter(LogFormatterMathtext(base=2))
        ax.get_yaxis().set_major_locator(LogLocator(base=2))

        # Adding labels and title
        ax.set_xlabel('Size of Benchmark Quantum Circuits (in qubits)', fontsize=24, labelpad=15)
        ax.set_ylabel('Average Program Size (in megabits)', fontsize=24, labelpad=15)
        # ax.set_title('Average and Standard Deviation of Program Sizes for Random Circuits by System Size (log2 scale)')
        ax.set_xticks(index + bar_width * (len(columns) - 1) / 2)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.legend(fontsize=14)

        # Show the plot
        plt.tight_layout()
        plt.savefig('Plots/article/program_sizes_bench.pdf', dpi=400)
        plt.show()


    def plot_cr_all(self):
        direc = 'records/benchmarks/benchmark_new_huff.csv'
        direc2 = 'records/random_trend/huff_records_new.csv'
        df1 = pd.read_csv(direc, delimiter=',')
        df2 = pd.read_csv(direc2, delimiter=',')
        sizes = range(2, 7)

        # Columns to analyze
        columns = ['cr huff1', 'cr huff2', 'cr huff3']

        # Initialize a dictionary to store dataframes for each size
        size_dfs1 = {}
        size_dfs2 = {}

        # Split the dataframe based on the 'size' column
        for size in sizes:
            size_dfs1[size] = df1[df1['size'] == size]
            size_dfs2[size] = df2[df2['size'] == size]

        # Initialize lists to hold the mean and std deviation data
        means1 = []
        std_devs1 = []
        means2 = []
        std_devs2 = []

        # Extract mean and std for each column and each size
        for size in sizes:
            means1.append(size_dfs1[size][columns].mean())
            std_devs1.append(size_dfs1[size][columns].std())
            means2.append(size_dfs2[size][columns].mean())
            std_devs2.append(size_dfs2[size][columns].std())
        print(means1, std_devs1)
        # Create figure for the plots
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.35, left=0.075, top=0.8, right=0.9)

        # Plotting for df1 (benchmark circuits)
        for i, col in enumerate(columns):
            y_mean = [mean[col] for mean in means1]
            y_std = [std[col] for std in std_devs1]
            ax.errorbar(sizes, y_mean, yerr=y_std, label=f'{col} (Benchmark)', fmt='o-', capsize=5)

        # Plotting for df2 (random circuits)
        for i, col in enumerate(columns):
            y_mean = [mean[col] for mean in means2]
            y_std = [std[col] for std in std_devs2]
            ax.errorbar(sizes, y_mean, yerr=y_std, label=f'{col} (Random)', fmt='o--', capsize=5)

        # ax.set_title('Mean Values with Standard Deviation for Benchmark and Random Circuits')
        ax.set_xlabel('System Size', fontsize=24, labelpad=15)
        ax.set_ylabel('Compression Factor', fontsize=24, labelpad=15)
        ax.legend(loc='best', fontsize=14)
        plt.tight_layout()
        plt.savefig('Plots/article/correl_cr.png', dpi=400)
        plt.show()

    def plot_energies(self):
        df = pd.read_csv('records/benchmarks/benchmark_new_huff.csv', delimiter=';')

        # Group by 'size' and calculate the mean for each size
        grouped = df.groupby('size')[['sumbits_qasm', 'sumbits binary', 'sumbits huff3', 'sumbits bz2+huff3']].mean()

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.15, top=0.9, right=0.95)
        # Width of the bar
        bar_width = 0.35
        offset = 0.2
        # Locations for the bars on the x-axis
        indices = range(len(grouped))
        # Opacity for overlapping bars
        opacity = 1.0

        # Plot each column
        bar1 = ax.bar([x-offset for x in indices], grouped['sumbits_qasm'] * (2.46e-9), width=bar_width, alpha=opacity, label='QASM',
                      color='#0047AB', hatch=None, zorder=1, edgecolor='black')
        # bar1 = ax.bar([x - offset for x in indices], grouped['sumbits_qasm'] * (2.46e-9), width=bar_width, label='QASM',
        #               alpha=opacity, color='none', zorder=2, edgecolor='k')
        bar2 = ax.bar([x-offset/3 for x in indices], grouped['sumbits binary'] * (2.46e-9), width=bar_width, alpha=opacity, label='Binary v0',
                      color='#097969', hatch=None, zorder=3, edgecolor='black')
        bar3 = ax.bar([x+offset/3 for x in indices], grouped['sumbits huff3']* (2.46e-9), width=bar_width, alpha=opacity, label='Huffman v3',
                      color='#9F2B68', hatch=None, zorder=4, edgecolor='black')
        bar4 = ax.bar([x+offset for x in indices], grouped['sumbits bz2+huff3']* (2.46e-9), width=bar_width, alpha=opacity, label='bzip2 on Huffman v3',
                      color='#DA70D6', hatch='o', zorder=5, edgecolor='black')

        # Add labels, title and axes ticks
        ax.set_xlabel('Size of Benchmark Quantum Circuits (in qubits)', fontsize=24, labelpad=15)
        ax.set_ylabel('Average Program Energy (in mJ)', fontsize=24, labelpad=15)
        # ax.set_title('Mean Values of benchmark Samples by Size')
        ax.set_xticks(indices)
        ax.set_xticklabels([str(i) for i in indices])
        ax.set_yscale('log')
        ax.set_axisbelow(True)
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5,
                alpha=0.7, zorder=0)  # Light gridlines for fidelity
        ax.set_xticklabels(grouped.index)

        offset_factor = 0.5  # Adjust offset factor as needed for better visibility
        # Annotations for percentage calculations
        heights_qasm = grouped['sumbits_qasm'].values
        heights_bin = grouped['sumbits binary'].values
        heights_huff3 = grouped['sumbits huff3'].values
        heights_bz2_huff3 = grouped['sumbits bz2+huff3'].values
        p1 = []
        p2 = []
        p3 = []
        for i, (hq, hb, hh3, hbz2h3) in enumerate(zip(heights_qasm, heights_bin, heights_huff3, heights_bz2_huff3)):
            percentage_huff3 = (hh3 / hq) * 100
            percentage_bz2h3 = (hbz2h3 / hq) * 100
            percentage_hbin = (hh3 / hb) * 100
            p1.append(percentage_huff3)
            p2.append(percentage_bz2h3)
            p3.append(percentage_hbin)
            plt.text(i, hh3 + offset_factor, f'{percentage_huff3:.1f}%', ha='center', va='top', color='white', fontsize=9)
            plt.text(i, hbz2h3 + offset_factor, f'{percentage_bz2h3:.1f}%', ha='center', va='top', color='white', fontsize=9)
        print("Average improvement in Huffman v3:", np.average(p1))
        print("Average improvement in Huffman v3 over binary:", np.average(p3))
        print("Average improvement in bzip2 + Huffman v3:", np.average(p2))
        # Add a legend
        ax.legend(fontsize=14)

        # Show the plot
        plt.tight_layout()
        plt.savefig('Plots/article/program_energies_3.pdf', dpi=400)
        plt.show()

    def plot_tot_complexity(self):
        direc = 'records/benchmarks/bench_new_huffq.csv'
        d = 'records/benchmarks/benchmark_new_huff.csv'
        # direc2 = 'records/random_trend/huff_records_q.csv'
        df1 = pd.read_csv(direc, delimiter=';')
        dfoo = pd.read_csv(d, delimiter=';')
        # df2 = pd.read_csv(direc2, delimiter=';')
        # sizes = range(2, 7)
        # print(df1)
        # exit()
        # print(df1.columns)
        # print(dfoo.columns)
        # exit()
        # df1 = df1.sort_values(['size','sumbits_bz2_qasm'])
        df1 = df1.sort_values(['size','circ_depth'])
        # dfoo = dfoo.sort_values(['size','sumbits_bz2_qasm'])
        dfoo = dfoo.sort_values(['size','circ_depth'])

        # Initialize a dictionary to store dataframes for each size
        size_dfs1 = {}
        size_dfs2 = {}

        x_arr = []
        for bench in dfoo['benchmark']:
            temp = bench.replace('_indep_qiskit', '')
            x_arr.append(temp)

        # Split the dataframe based on the 'size' column
        # for size in sizes:
        #     size_dfs1[size] = df1[df1['size'] == size]
        #     size_dfs2[size] = df2[df2['size'] == size]
        print(dfoo['size'])
        print(dfoo['circ_depth'])
        circ_complexities = dfoo['size'] * dfoo['circ_depth']
        descrip_complexities = dfoo['sumbits bz2+huff3']
        # circ_arr = df2['size'].astype(str) + 'q_' + df2["number"].astype(str)

        fig, ax = plt.subplots(figsize=(18, 9))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.35, left=0.075, top=0.8, right=0.95)
        plt.plot(x_arr, circ_complexities, label='Circuit Complexity', color="#0b655f", marker='v', alpha=0.6)  # blue
        plt.plot(x_arr, descrip_complexities, label='Description Complexity', color="#9f2b68", marker='^', alpha=0.6)  # purple
        # plt.plot(circ_arr, circ_complexities+descrip_complexities, label='Total Complexity')
        ax.set_yscale('log')
        ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5,
                 alpha=0.7)  # Light gridlines for fidelity
        plt.xlabel('Benchmark name', fontsize=24, labelpad=15)
        plt.xticks(range(len(x_arr)), x_arr)  # Assuming x_arr contains the labels for x-axis
        plt.gca().margins(x=0.001)  # Adjusting the margins to avoid cutting off the labels
        plt.gcf().canvas.draw()
        plt.xticks(rotation=90)
        plt.ylabel('Complexity', fontsize=24, labelpad=15)
        plt.legend(loc='best', fontsize=14)
        plt.savefig('Plots/article/complexities_bench_2.pdf', dpi=400)
        plt.show()


    def plot_cr(self):
        folder = 'records/benchmarks/'
        # filename = 'benchmarks_huff_records.csv'
        # filename = 'bench_new_huffq.csv'
        filename = 'benchmark_new_huff.csv'
        read = pd.read_csv(folder + filename, delimiter=';')
        # read = read.sort_values('size')
        read = read.sort_values(['size','circ_depth'])
        x_arr = []
        for bench in read['benchmark']:
            temp = bench.replace('_indep_qiskit', '')
            x_arr.append(temp)
        # plt.figure(figsize=(14, 8))
        fig, ax = plt.subplots(figsize=(17, 9))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.35, left=0.1, top=0.8, right=0.95)

        

        labels = ["QASM", "Binary v0", "Huffman v1", "Huffman v2", "Huffman v3"]
        colors = ['#0047AB', '#097969', '#808080', '#E97451', '#9F2B68']

        plt.axhline(y=1.0, linestyle='--', linewidth=2, label='Binary v0', color=colors[1])
        plt.plot(x_arr, read['cr huff1'], marker='o', ms=2, label='Huffman v1', alpha=0.7, color=colors[2])
        plt.plot(x_arr, read['cr huff2'], marker='o', ms=2, label='Huffman v2', alpha=0.7, color=colors[3])
        plt.plot(x_arr, read['cr huff3'], marker='o', ms=2, label='Huffman v3', alpha=0.7, color=colors[4])
        plt.xlabel("Benchmark name", fontsize=24, labelpad=15)
        plt.ylabel(r'Compression Factor', fontsize=24, labelpad=15)
        plt.legend(loc='best', fontsize=14)

        plt.xticks(range(len(x_arr)), x_arr)  # Assuming x_arr contains the labels for x-axis
        plt.gca().margins(x=0.001)  # Adjusting the margins to avoid cutting off the labels
        plt.gcf().canvas.draw()
        plt.xticks(rotation=90)
        # plt.title("Compression Factor for Benchmark Circuits with sizes 2 to 6", fontsize=24)
        plt.savefig('Plots/article/cr_bench_ordered.pdf', dpi=400)
        plt.show()

pl = Plotter()
# pl.plot_cr_comparison(2) # not in article
# pl.plot_length_comparison(2) # not in article
# pl.plot_fid_depth(3) # not in article
# pl.plot_fid_recur(1) # not in article
# pl.plot_circ_depth_recur(1) # not in article
# pl.plot_circ_depth_depth(1) # not in article
# pl.calc_plot_avg(1)
# pl.plot_avg_cr()
# pl.plot_avg_cr_bench()
# pl.plot_program_size()
# pl.plot_cr_all()
# pl.plot_energies()
# pl.plot_tot_complexity()
pl.plot_cr()
"""
For degree of recursion =  1 Avg. Fidelity =  0.10846959350371296
For degree of recursion =  2 Avg. Fidelity =  0.22235182097107997
For degree of recursion =  3 Avg. Fidelity =  0.5103603100660115
For degree of recursion =  4 Avg. Fidelity =  0.8539507791188845
"""

"""
Fidelity vs Depth for recur fixed at 3
For depth =  1 ; Mean Fidelity =  0.015656722285518995 ; Std Fidelity =  0.01442822726887395
For depth =  2 ; Mean Fidelity =  0.015239758283289695 ; Std Fidelity =  0.014894533562675734
For depth =  3 ; Mean Fidelity =  0.5472350431099623 ; Std Fidelity =  0.1484486835434515
For depth =  4 ; Mean Fidelity =  0.5247166858681518 ; Std Fidelity =  0.11344520094261608
For depth =  5 ; Mean Fidelity =  0.8931481013607561 ; Std Fidelity =  0.030346537181883015
For depth =  6 ; Mean Fidelity =  0.8329202024778126 ; Std Fidelity =  0.048077984857957334
"""
