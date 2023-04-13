import math
from itertools import permutations

cipher = "aauancvlrerurnndltmeaeepbytusticeatnpmeyiicgogorchsrsocnntiiimihaoofpagsivttpsitlbolrotoex"
possible_lens = [2, 3, 5, 6, 9, 10, 15, 18, 30, 45]

for row_len in possible_lens:
    ## Make a table from the cipher
    cipher_table = []
    for row in range(math.ceil(len(cipher)/row_len)):
        to_append = list(cipher[row*row_len:row*row_len+row_len])
        # while len(to_append) < 8:
        #     to_append.append(0) # Zero pad
        cipher_table.append(to_append)
    # print(cipher_table)

    possible_orders = permutations(range(row_len))
    for index_order in possible_orders:
        # print(index_order)

        ## Reorder the table
        ordered_table = []
        for row in cipher_table:
            # print(row)
            ordered_table.append([])
            for index in index_order:
                ordered_table[-1].append(row[index])
        # print(ordered_table)

        output_string = ""
        for row in ordered_table:
            for letter in row:
                output_string += letter

        if "computer" in output_string:
            print(output_string)
            print("EUREKA!")
            quit(1)
    print(output_string)

