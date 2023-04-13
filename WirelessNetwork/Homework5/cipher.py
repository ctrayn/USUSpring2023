import math
import numpy
from itertools import permutations

cipher = "aauancvlrerurnndltmeaeepbytusticeatnpmeyiicgogorchsrsocnntiiimihaoofpagsivttpsitlbolrotoex"
possible_lens = [15]

for row_len in possible_lens:
    ## Make a table from the cipher
    cipher_table = []
    for row in range(math.ceil(len(cipher)/row_len)):
        to_append = list(cipher[row*row_len:row*row_len+row_len])
        while len(to_append) < row_len:
            to_append.append('0') # Zero pad
        cipher_table.append(to_append)
    tran_table = numpy.transpose(cipher_table)
    print(cipher_table)
    print(tran_table)

    possible_orders = permutations(range(len(tran_table[0])))
    for index_order in possible_orders:
        # print(index_order)
        output_string = ''
        for row in tran_table:
            for index in index_order:
                output_string += row[index]
        while '0' in output_string:
            output_string = output_string.replace('0','')
        print(f'{index_order}{output_string}')
        if "computer" in output_string:
            print("Found")
            quit(1)
print("Not found")

