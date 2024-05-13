from qiskit.synthesis.discrete_basis.gate_sequence import GateSequence
import numpy as np
import ast
from itertools import product
from custom_basis import _1q_gates, _1q_inverses


class Helper():
    def __int__(self, basis_dict, arr, recur) -> None:
        self.dictionary = basis_dict
        self.simpledict, self.indices = self.prepare_dicts(basis_dict)
        self.arr = arr
        self.recur = recur

    # Function for preparing the dictionaries needed for functionalities of this file
    def prepare_dicts(self, basic):
        adjoints = {}
        simpledict = {}
        indices = {}
        iterable = 1

        for str_seq in basic.keys():
            indices[str(iterable)] = str_seq
            indices[str_seq] = str(iterable)
            iterable += 1
            seq_arr = ast.literal_eval(str_seq)
            seq = GateSequence()
            for gate in seq_arr:
                seq.append(_1q_gates[gate])
            simpledict[str_seq] = seq

        return simpledict, indices

        # Function for modifying the dictionary of frequencies of elements from S-K Basis to include -dg instructions
    def basic_traverse(self):
        recur = self.recur
        arr = self.arr
        chars = ['U', 'V', 'W']
        combinations = product(chars, repeat=recur)
        words = [''.join(combination) for combination in combinations]
        words.sort()
        counting_arr = np.ones(3 ** recur)
        i = 0
        for word in words:
            for char in word:
                if char == 'U':
                    counting_arr[i] *= 1
                else:
                    counting_arr[i] *= 2
            i += 1
        # print("Words array: ", words)
        # print("Counting array: ", counting_arr)
        for index in range(len(counting_arr)):
            if counting_arr[index] % 2 != 0:
                self.dictionary[str(arr[index].labels)] += 1
            else:
                self.dictionary[str(arr[index].labels)] += counting_arr[index] / 2
                temp_seq = self._remove_self_inverses(arr[index].adjoint())
                for i in self.simpledict.keys():
                    if self.matrixCompare(self.simpledict[i].product, temp_seq.product):
                        self.dictionary[str(self.simpledict[i].labels)] += counting_arr[index] / 2
                # self.dictionary[str(temp_seq.labels)] += counting_arr[index] / 2
        # print("find_basic_dictionary :", self.dictionary)
        depth = 0
        for key in self.dictionary.keys():
            modifkey = ast.literal_eval(key)
            depth += self.dictionary[key] * len(modifkey)
        # print("Circuit Depth from reconstruction of decomposition: ", depth)

        return self.dictionary

    def matrixCompare(self, A, B):
        if A.shape != B.shape:
            return False
        if A.dtype != B.dtype:
            return False
        return np.allclose(A, B, atol=10 ** -3)

    def checker(self, inp_str):
        if not inp_str.endswith('-dg'):
            return inp_str
        else:
            if inp_str.startswith('['):
                temp = ast.literal_eval(inp_str.removesuffix('-dg'))
                temp_arr = [t.removesuffix('-dg') if t.endswith('-dg') else t + '-dg' for t in reversed(temp)]
                temp_arr = [self.checker(item) for item in temp_arr]
                # inp_str = self.check_correct_daggers(str(temp_arr))
                # inp_str = str(temp_arr)
                return str(temp_arr)
            return inp_str

    def check_and_correct(self, sublist):
        corrected_sublist = [item if not item.endswith('-dg')
                             else self.checker(item) for item in sublist]

        return [self.checker(item) for item in sublist]

    def create_subgroups(self, main_list, recur):
        # print("Passed array at step-0 of this function: ", '\n', main_list)
        result_list = []

        for i in range(0, len(main_list), 3):
            subgroup = main_list[i:i+3]
            new_list = [subgroup[1], subgroup[2], subgroup[1] + '-dg', subgroup[2] + '-dg', subgroup[0]]
            # Step 3: Check if "-dg" added correctly and remove extra -dg; remove "-dg-dg"
            new_list = self.check_and_correct(new_list)
            # print("Sublist after corrections of extra introduced daggers: ", '\n', new_list)
            result_list.append(str(new_list))

        return result_list, recur - 1

    def flatten_nested_strings(self, inp_arr, count):
        if count == 0:
            return inp_arr
        else:
            tempstring = ''.join(inp_arr)
            tempstring = tempstring.replace('][', ', ')
            return self.flatten_nested_strings(ast.literal_eval(tempstring), count-1)

    def call_nested(self):
        recur = self.recur
        labels_arr = []
        for item in self.arr:
            labels_arr.append(str(self.indices[str(item.labels)]))
        while recur > 0:
            labels_arr, recur = self.create_subgroups(labels_arr, recur)
        flat = self.flatten_nested_strings(labels_arr, count=self.recur)
        # print("Test fattening", flat)
        return self.parse_flattened(flat)

    def parse_flattened(self, flat):
        circ = GateSequence()
        for i in range(len(flat)):
            if flat[i].endswith('-dg'):
                string = self.indices[flat[i].removesuffix('-dg')]
                string_seq = self._remove_self_inverses(self.simpledict[string].adjoint())
                for index in self.simpledict.keys():
                    if self.matrixCompare(self.simpledict[index].product, string_seq.product):
                        flat[i] = self.indices[str(self.simpledict[index].labels)]
                # flat[i] = self.indices[str(self._remove_self_inverses(self.simpledict[string].adjoint()).labels)]

        instruction_stream_indexed = []
        instruction_stream = []
        for item in reversed(flat):
            if self.indices[item] != '[]':
                instruction_stream_indexed.append(self.indices[item])

        # for item in reversed(flat):
        #     for gate in self.simpledict[self.indices[item]]:
        #         circ.append(_1q_gates[gate.name])
        # print("Reconstructed circuit from updated instruction stream: ",'\n')
        # print(circ.to_circuit())
        # print("Reconstructed circuit depth = ", circ.to_circuit().depth())
        return instruction_stream_indexed

    def _remove_self_inverses(self, sequence):
        new_seq = GateSequence()
        for index in range(len(sequence.gates)):
            temp_name = sequence.gates[index].name
            if sequence.gates[index].name.endswith('dgdg') and sequence.gates[index].name.removesuffix('dgdg') in _1q_gates:
                # print("temp = ", temp_name, "seq[index] = ", sequence.gates[index].name)
                new_seq.append(_1q_gates[sequence.gates[index].name.removesuffix('dgdg')])
                # sequence.gates[index] = _1q_gates[sequence.gates[index].name.removesuffix('dgdg')]
            elif sequence.gates[index].name.endswith('dg') and _1q_inverses[sequence.gates[index].name.removesuffix('dg')] == sequence.gates[index].name.removesuffix('dg'):
                # print("seq[index] = ", sequence.gates[index].name)
                new_seq.append(_1q_gates[sequence.gates[index].name.removesuffix('dg')])
                # sequence.gates[index] = _1q_gates[sequence.gates[index].name.removesuffix('dg')]
            else:
                new_seq.append(_1q_gates[sequence.gates[index].name])

        # print(new_seq.labels)
        return new_seq