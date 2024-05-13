import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogFormatter, LogLocator

class Plotter:

    def plot_circ_depth(self):
        folder = 'records/benchmarks/'
        filename = 'benchmarks_huff_records.csv'
        # filename = 'bench_new_huffq.csv'
        read = pd.read_csv(folder + filename)
        # read = read.sort_values('size')
        x_arr = []
        for bench in read['benchmark']:
            temp = bench.replace('_indep_qiskit', '')
            x_arr.append(temp)
        # plt.figure(figsize=(14, 8))
        fig, ax = plt.subplots(figsize=(17, 9))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.35, left=0.1, top=0.8, right=0.95)
        plt.plot(x_arr, read['circ_depth']/1000)
        plt.xlabel("Benchmark name", fontsize=24, labelpad=15)
        plt.ylabel("Circuit Depth in 1000s", fontsize=24, labelpad=15)

        plt.xticks(range(len(x_arr)), x_arr)  # Assuming x_arr contains the labels for x-axis
        plt.gca().margins(x=0.001)  # Adjusting the margins to avoid cutting off the labels
        plt.gcf().canvas.draw()
        plt.xticks(rotation=90)
        # plt.title("Circuit Depths for Benchmark Circuits with sizes 2 to 6", fontsize=14)
        # plt.savefig('Plots/17.04.2024/cd_bench.png')
        plt.show()

    def plot_fidelities(self):
        folder = 'records/benchmarks/'
        filename = 'benchmarks_huff_records.csv'
        # filename = 'bench_new_huffq.csv'
        read = pd.read_csv(folder + filename)
        # read = read.sort_values('size')
        x_arr = []
        for bench in read['benchmark']:
            temp = bench.replace('_indep_qiskit', '')
            x_arr.append(temp)
        # plt.figure(figsize=(14, 8))
        fig, ax = plt.subplots(figsize=(17, 9))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.35, left=0.1, top=0.8, right=0.95)
        plt.plot(x_arr, read['fidelity'])
        plt.xlabel("Benchmark name", fontsize=24, labelpad=15)
        plt.ylabel("Fidelity", fontsize=24, labelpad=15)

        plt.xticks(range(len(x_arr)), x_arr)  # Assuming x_arr contains the labels for x-axis
        plt.gca().margins(x=0.001)  # Adjusting the margins to avoid cutting off the labels
        plt.gcf().canvas.draw()
        plt.xticks(rotation=90)
        # plt.title("Fidelities for Benchmark Circuits with sizes 2 to 6", fontsize=14)
        plt.savefig('Plots/17.04.2024/fid_bench.png')
        plt.show()

    def plot_fid_depth(self):
        folder = 'records/benchmarks/'
        filename = 'benchmarks_huff_records.csv'
        read = pd.read_csv(folder + filename)
        read = read.sort_values('size')
        read['size'] = read['benchmark'].str.extract('(\d+)$').astype(int)
        x_arr = []
        for bench in read['benchmark']:
            temp = bench.replace('_indep_qiskit', '')
            x_arr.append(temp)
        fig, ax1 = plt.subplots(figsize=(17, 9))
        ax1.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.35, left=0.075, top=0.8, right=0.9)

        # Plotting fidelity
        color = 'tab:red'
        ax1.set_xlabel('Benchmark Name', fontsize=24, labelpad=15)
        ax1.set_ylabel('Fidelity', color=color, fontsize=24, labelpad=15)
        ax1.plot(x_arr, read['fidelity'], color=color, marker='o', ms='2')
        # previous_size = -1
        # previous_benchmark = None
        # previous_fidelity = None
        # for _, row in read.iterrows():
        #     print(_, row)
        #     if _ == 0:
        #         previous_benchmark = row['benchmark']
        #         previous_fidelity = row['fidelity']
        #         previous_size = row['size']
        #         ax1.plot([previous_benchmark, row['benchmark']], [previous_fidelity, row['fidelity']],
        #                  color=color, marker='o', ms=2)  # Continue the line
        #     # Check if there is a change in size to break the line
        #     if row['size'] != previous_size and previous_size != -1:
        #         ax1.plot([previous_benchmark, row['benchmark']], [previous_fidelity, row['fidelity']], color=color,
        #                  linestyle='None')  # Plot nothing on size change
        #     else:
        #         ax1.plot([previous_benchmark, row['benchmark']], [previous_fidelity, row['fidelity']],
        #                  color=color, marker='o', ms=2)  # Continue the line
        #     previous_benchmark = row['benchmark']
        #     previous_fidelity = row['fidelity']
        #     previous_size = row['size']
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', rotation=90)  # Rotate x labels for better readability
        ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5,
                 alpha=0.7)  # Light gridlines for fidelity

        # Instantiate a second axis that shares the same x-axis
        ax2 = ax1.twinx()
        ax2.tick_params(axis='both', which='major', labelsize=14)
        color = 'tab:blue'
        ax2.set_ylabel('Circuit Depth', color=color, fontsize=24, labelpad=15)
        ax2.bar(x_arr, read['circ_depth'], color=color, alpha=0.6)
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color)
        # ax2.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5,
        #          alpha=0.7)  # Light gridlines for circuit depth

        # Make the layout tight so labels do not get cut off
        # plt.xticks(range(len(x_arr)), x_arr)  # Assuming x_arr contains the labels for x-axis
        # plt.gca().margins(x=0.001)  # Adjusting the margins to avoid cutting off the labels
        # plt.gcf().canvas.draw()
        plt.tight_layout()
        plt.savefig('Plots/28.04.2024/bench_fid_depth.png', dpi=400)
        plt.show()

    def plot_heatmap(self):
        folder = 'records/benchmarks/'
        filename = 'benchmarks_huff_records.csv'
        read = pd.read_csv(folder + filename)

        # Assuming 'read' DataFrame has 'benchmark', 'size', and 'fidelity' columns
        # We need to reshape this DataFrame to have 'benchmark' names as the y-axis and 'size' as the x-axis.
        # The 'fidelity' values will fill in the heatmap.
        # heatmap_data = read.pivot("benchmark", "size", "fidelity")
        heatmap_data = read.pivot(index='benchmark', columns='fidelity', values='circ_depth')
        # Now we can create the heatmap using seaborn
        plt.figure(figsize=(15, 10))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)

        # Set the labels and title
        plt.ylabel('Benchmark Name')
        plt.xlabel('Size')
        plt.title('Heatmap of Fidelity by Benchmark and Size')

        # Show the plot
        plt.show()

    def plot_cr(self):
        folder = 'records/benchmarks/'
        # filename = 'benchmarks_huff_records.csv'
        # filename = 'bench_new_huffq.csv'
        filename = 'benchmark_new_huff.csv'
        read = pd.read_csv(folder + filename, delimiter=';')
        read = read.sort_values('size')
        x_arr = []
        for bench in read['benchmark']:
            temp = bench.replace('_indep_qiskit', '')
            x_arr.append(temp)
        # plt.figure(figsize=(14, 8))
        fig, ax = plt.subplots(figsize=(17, 9))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.35, left=0.1, top=0.8, right=0.95)
        plt.axhline(y=1.0, linestyle='--', color='#999999', linewidth=2, label='v0: Binary Encoding')
        plt.plot(x_arr, read['cr huff1'], marker='o', ms=2, label='Huffman v1', alpha=0.7)
        plt.plot(x_arr, read['cr huff2'], marker='o', ms=2, label='Huffman v2', alpha=0.7)
        plt.plot(x_arr, read['cr huff3'], marker='o', ms=2, label='Huffman v3', alpha=0.7)
        plt.xlabel("Benchmark name", fontsize=24, labelpad=15)
        plt.ylabel(r'Compression Factor', fontsize=24, labelpad=15)
        plt.legend(loc='best', fontsize=14)

        plt.xticks(range(len(x_arr)), x_arr)  # Assuming x_arr contains the labels for x-axis
        plt.gca().margins(x=0.001)  # Adjusting the margins to avoid cutting off the labels
        plt.gcf().canvas.draw()
        plt.xticks(rotation=90)
        # plt.title("Compression Factor for Benchmark Circuits with sizes 2 to 6", fontsize=24)
        plt.savefig('Plots/02.05.2024/cr_bench_ordered.png')
        plt.show()

    def plot_lengths(self):
        folder = 'records/benchmarks/'
        filename = 'benchmark_new_huff.csv'
        read = pd.read_csv(folder + filename, delimiter=';')
        read = read.sort_values('size')
        x_arr = []
        for bench in read['benchmark']:
            temp = bench.replace('_indep_qiskit', '')
            x_arr.append(temp)
        # plt.figure(figsize=(14, 8))
        fig, ax = plt.subplots(figsize=(17, 9))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.35, left=0.1, top=0.8, right=0.95)
        plt.plot(x_arr, read['sumbits binary']/1000, marker='o', ms=2, label='v0: Binary Encoding', alpha=0.7)
        plt.plot(x_arr, read['sumbits huff1']/1000, marker='o', ms=2, label='Huffman v1', alpha=0.7)
        plt.plot(x_arr, read['sumbits huff2']/1000, marker='o', ms=2, label='Huffman v2', alpha=0.7)
        plt.plot(x_arr, read['sumbits huff3']/1000, marker='o', ms=2, label='Huffman v3', alpha=0.7)
        plt.xlabel("Benchmark name", fontsize=24, labelpad=15)
        plt.ylabel('Encoded bit-length of circuit (in 1000s)', fontsize=24, labelpad=15)
        plt.legend(loc='best', fontsize=14)
        ax.set_yscale('log')
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width, box.height])
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), fontsize=14)
        plt.xticks(range(len(x_arr)), x_arr)  # Assuming x_arr contains the labels for x-axis
        plt.gca().margins(x=0.001)  # Adjusting the margins to avoid cutting off the labels
        plt.gcf().canvas.draw()
        plt.xticks(rotation=90)
        plt.savefig('Plots/02.05.2024/lengths_bench_ordered.png')
        # plt.title("Encoded Bit-Length for Benchmark Circuits with sizes 2 to 6", fontsize=14)
        plt.show()

    def plot_correl_fid_depth(self):
        direc = 'records/benchmarks/bench_new_huffq.csv'
        direc2 = 'records/random_trend/huff_records_q.csv'
        df1 = pd.read_csv(direc, delimiter=';')
        df2 = pd.read_csv(direc2, delimiter=';')
        sizes = range(2, 7)
        # Columns to analyze
        columns = ['fidelity', 'circ_depth']

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
            size_dfs2[size]['fidelity'] = pd.to_numeric(size_dfs2[size]['fidelity'], errors='coerce')
            means2.append(size_dfs2[size][columns].mean())
            std_devs2.append(size_dfs2[size][columns].std())
        # print(means2)
        # Create figure for the plots
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = ax1.twinx()
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.35, left=0.075, top=0.8, right=0.9)
        rcolor = 'tab:red'
        bcolor = 'tab:blue'
        gcolor = 'tab:green'
        pcolor = 'tab:purple'
        # Plotting for Fidelity (benchmark circuits vs random circuits)
        for i, col in enumerate(columns):
            y_mean = [mean[col] for mean in means1]
            y_std = [std[col] for std in std_devs1]
            if i == 0:
                ax1.errorbar(sizes, y_mean, yerr=y_std, label=f'{col} (Benchmark)', fmt='o-', capsize=5, color=rcolor, alpha=1)
            else:
                ax2.errorbar(sizes, y_mean, yerr=y_std, label=f'{col} (Benchmark)', fmt='o-', capsize=5, color=gcolor, alpha=1)

        # Plotting for Circ Depth (benchmark circuits vs random circuits)
        for i, col in enumerate(columns):
            y_mean = [mean[col] for mean in means2]
            y_std = [std[col] for std in std_devs2]
            if i == 0:
                ax1.errorbar(sizes, y_mean, yerr=y_std, label=f'{col} (Random)', fmt='o--', capsize=5, color=bcolor, alpha=1)
            else:
                ax2.errorbar(sizes, y_mean, yerr=y_std, label=f'{col} (Random)', fmt='o--', capsize=5, color=pcolor, alpha=1)
            # ax2.errorbar(sizes, y_mean, yerr=y_std, label=f'{col} (Random)', fmt='o--', capsize=5, color=color)
        ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5,
                 alpha=0.7)  # Light gridlines for fidelity
        # ax1.set_title('Mean Values with Standard Deviation for Benchmark and Random Circuits')
        ax1.set_xlabel('System Size', fontsize=24, labelpad=15)
        ax1.set_ylabel('Fidelity', fontsize=24, labelpad=15)
        # ax2.set_yscale('log')
        ax2.set_ylabel('Circuit Depth', fontsize=24, labelpad=15)
        # ax1.legend()
        # ax2.legend()
        fig.legend(loc='center left', bbox_to_anchor=(0.1, 0.5), fontsize=14)
        plt.tight_layout()
        plt.savefig('Plots/02.05.2024/correl_fid_depth.png', dpi=400)
        plt.show()

    def plot_tot_complexity(self):
        direc = 'records/benchmarks/bench_new_huffq.csv'
        d = 'records/benchmarks/benchmark_new_huff.csv'
        direc2 = 'records/random_trend/huff_records_q.csv'
        df1 = pd.read_csv(direc, delimiter=';')
        dfoo = pd.read_csv(d, delimiter=';')
        df2 = pd.read_csv(direc2, delimiter=';')
        sizes = range(2, 7)
        df1 = df1.sort_values('size')
        dfoo = dfoo.sort_values('size')

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
        circ_arr = df2['size'].astype(str) + 'q_' + df2["number"].astype(str)

        fig, ax = plt.subplots(figsize=(18, 9))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.35, left=0.075, top=0.8, right=0.95)
        plt.plot(x_arr, circ_complexities, label='Circuit Complexity')
        plt.plot(x_arr, descrip_complexities, label='Description Complexity')
        # plt.plot(circ_arr, circ_complexities+descrip_complexities, label='Total Complexity')
        ax.set_yscale('log')
        ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5,
                 alpha=0.7)  # Light gridlines for fidelity
        plt.xlabel('Benchmarks', fontsize=24, labelpad=15)
        plt.xticks(range(len(x_arr)), x_arr)  # Assuming x_arr contains the labels for x-axis
        plt.gca().margins(x=0.001)  # Adjusting the margins to avoid cutting off the labels
        plt.gcf().canvas.draw()
        plt.xticks(rotation=90)
        plt.ylabel('Complexity', fontsize=24, labelpad=15)
        plt.legend(loc='best', fontsize=14)
        plt.savefig('Plots/02.05.2024/complexities_bench_2.png', dpi=400)
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
        offset = 0.1
        # Locations for the bars on the x-axis
        indices = range(len(grouped))
        # Opacity for overlapping bars
        opacity = 1.0

        # Plot each column
        bar1 = ax.bar([x-offset for x in indices], grouped['sumbits_qasm'] * (2.46e-9), width=bar_width, alpha=opacity, label='QASM',
                      color='#6495ED', hatch='//', zorder=1, edgecolor='#0047AB')
        # bar1 = ax.bar([x - offset for x in indices], grouped['sumbits_qasm'] * (2.46e-9), width=bar_width, label='QASM',
        #               alpha=opacity, color='none', zorder=2, edgecolor='k')
        bar2 = ax.bar([x-offset/3 for x in indices], grouped['sumbits binary'] * (2.46e-9), width=bar_width, alpha=opacity, label='Binary',
                      color='#3CB371', hatch='\\\\', zorder=3, edgecolor='#006400')
        bar3 = ax.bar([x+offset/3 for x in indices], grouped['sumbits huff3']* (2.46e-9), width=bar_width, alpha=opacity, label='Huffman v3',
                      color='#A9A9A9', hatch='o', zorder=4, edgecolor='black')
        bar4 = ax.bar([x+offset for x in indices], grouped['sumbits bz2+huff3']* (2.46e-9), width=bar_width, alpha=opacity, label='bz2 + Huffman v3',
                      color='#FA8072', hatch='.', zorder=5, edgecolor='#8B0000')

        # Add labels, title and axes ticks
        ax.set_xlabel('Size', fontsize=24, labelpad=15)
        ax.set_ylabel('Program Energy (in mJ)', fontsize=24, labelpad=15)
        # ax.set_title('Mean Values of benchmark Samples by Size')
        ax.set_xticks(indices)
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
        plt.show()

    def plot_program_energy(self):
        direc = 'records/benchmarks/benchmark_new_huff.csv'
        direc2 = 'records/random_trend/huff_records_new.csv'
        df = pd.read_csv(direc, delimiter=';')
        print(df)
        # Sizes you mentioned in your dataset
        sizes = range(2, 7)
        size = pd.DataFrame([2, 3, 4, 5, 6])
        # Columns to analyze
        columns = ['sumbits_qasm']
        huffcolumns = ['sumbits huff3']
        bz2_columns = ['sumbits bz2+huff3']

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
        means3 = []
        std_devs3 = []

        # Extract mean and std for each column and each size
        for size in sizes:
            means1.append(size_dfs[size][columns].mean() * (2.46e-9))  # Convert bits to milliJoules
            std_devs1.append(size_dfs[size][columns].std() * (2.46e-9))  # Convert bits to milliJoules
            means2.append(size_dfs[size][huffcolumns].mean() * (2.46e-9))  # Convert bits to milliJoules
            std_devs2.append(size_dfs[size][huffcolumns].std() * (2.46e-9))  # Convert bits to milliJoules
            means3.append(size_dfs[size][bz2_columns].mean() * (2.46e-9))  # Convert bits to milliJoules
            std_devs3.append(size_dfs[size][bz2_columns].std() * (2.46e-9))  # Convert bits to milliJoules
        print(means1)
        print(means2)
        print(means3)
        plt.bar(size, means1, label='QASM', edgecolor='black')
        plt.bar(size, means2, label='Huffman v3', edgecolor='black')
        plt.bar(size, means3, label='bz2 + Huffman v3', edgecolor='black')
        plt.show()

        # Creating the bar plots
        fig, ax = plt.subplots(figsize=(17, 8))
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.subplots_adjust(bottom=0.15, left=0.3, right=0.95)
        # Setting positions for the bars
        bar_width = 0.15
        index = np.arange(len(sizes))
        labels = ["QASM", "v0: binary encoded", "Huff v1", "Huff v2", "Huff v3"]
        means_col, means_col2, means_col3 = [], [], []

        # Plot first set of columns
        for i, col in enumerate(columns):
            means_col = [mean[col] for mean in means1]
            std_devs_col = [std[col] for std in std_devs1]
            ax.bar(index, means_col, label='QASM', alpha=0.9, edgecolor='black')
            # ax.bar(index + i * bar_width, means_col, bar_width, yerr=std_devs_col, capsize=2, label=labels[i % len(labels)], alpha=0.9, edgecolor='black')

        for i, col in enumerate(huffcolumns):
            means_col2 = [mean[col] for mean in means1]
            std_devs_col = [std[col] for std in std_devs1]
            ax.bar(index, means_col2, label='Huffman v3', alpha=0.9, edgecolor='black')
            # ax.bar(index + i * bar_width, means_col, bar_width, yerr=std_devs_col, capsize=2, label=labels[i % len(labels)], alpha=0.9, edgecolor='black')

        # Hatches for second set of bars
        hatches = ['/', '\\', "O", ".", "o"]
        labels = ["bz2 on QASM", "bz2 on v0: binary", "bz2 on Huff v1", "bz2 on Huff v2", "bz2 on Huff v3"]
        # Plot second set of columns superimposed with lower alpha
        for i, col in enumerate(bz2_columns):
            means_col3 = [mean[col] for mean in means2]
            std_devs_col = [std[col] for std in std_devs2]
            ax.bar(index, means_col3, label='bz2 + Huffman v3', alpha=0.9, hatch=hatches[i % len(hatches)], edgecolor='black')
            # for j in range(len(sizes)):
            #     plt.text(index, means_col3[j], (means_col3[j]*100)/means_col[j], ha='center')
            # ax.bar(index + i * bar_width, means_col, bar_width, yerr=std_devs_col, capsize=2, label=labels[i % len(labels)], alpha=0.5, hatch=hatches[i % len(hatches)], edgecolor = 'black')

        # Set the y-axis to a logarithmic scale with base 2
        ax.set_yscale('log')
        # ax.get_yaxis().set_major_formatter(LogFormatter(base=2))
        # ax.get_yaxis().set_major_locator(LogLocator(base=2))

        # Adding labels and title
        ax.set_xlabel('System Size', fontsize=24, labelpad=15)
        ax.set_ylabel('Program Energy (in mJ)', fontsize=24, labelpad=15)
        # ax.set_title('Average and Standard Deviation of Program Sizes for Random Circuits by System Size (log2 scale)')
        ax.set_xticks(index + bar_width * (len(columns) - 1) / 2)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.legend(fontsize=14)

        # Show the plot
        plt.tight_layout()
        # plt.savefig('Plots/30.04.2024/program_energy_bench.png', dpi=400)
        plt.show()

pl = Plotter()
# pl.plot_circ_depth()
# pl.plot_fidelities()
pl.plot_cr()
pl.plot_lengths()
# pl.plot_fid_depth()
# pl.plot_heatmap()
# pl.plot_correl_fid_depth()
# pl.plot_tot_complexity()
# pl.plot_program_energy()
pl.plot_energies()