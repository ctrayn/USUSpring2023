################################
# Global Data
################################

diff_phase = {
    '00': '180',
    '01': '-90',
    '10': '0',
    '11': '+90'
    }

diff_bits = {
    '180': '00',
    '-90': '01',
    '0'  : '10',
    '+90': '11'
}

################################
# Encoder
################################

def encoder(sent_bits):
    bit_pairs = [f"{sent_bits[i]}{sent_bits[i + 1]}" for i in range(0,len(sent_bits),2)]
    bit_phase = [diff_phase[sym] for sym in bit_pairs]

    symbols = ['00']
    for phase_change in bit_phase:
        if phase_change == '180':
            bits = int(symbols[-1], 2)
            symbols.append(f"{bits ^ 0x3:02b}")
        elif phase_change == '0':
            ## keep it the same
            symbols.append(symbols[-1])
            pass
        elif phase_change == '+90':
            phase_index = int(symbols[-1], 2)
            changes = ['10', '00', '11', '01']
            symbols.append(changes[phase_index])
        else: #'-90'
            phase_index = int(symbols[-1], 2)
            changes = ['01', '11', '00', '10']
            symbols.append(changes[phase_index])

    I = [-1 if int(sym[0]) == 0 else 1 for sym in symbols]
    Q = [-1 if int(sym[1]) == 0 else 1 for sym in symbols]

    return I, Q

################################
# Decoder
################################

def decoder(X, Y):
    """Changes sampled differentially encoded bits to bits"""

    r_symbols = [f"{X[i]}{Y[i]}" for i in range(len(X))]

    # bit pairs to phase changes

    positive_change = ['01', '11', '00', '10']
    negative_change = ['10', '00', '11', '01']

    r_phase_changes = []
    for index in range(1,len(r_symbols)):
        previous_sym = r_symbols[index -1]
        curr_sym = r_symbols[index]

        if curr_sym == previous_sym:
            r_phase_changes.append('0')
        elif f"{int(curr_sym, 2) ^ 0x3:02b}" == previous_sym:
            r_phase_changes.append('180')
        elif positive_change[int(curr_sym, 2)] == previous_sym:
            r_phase_changes.append('+90')
        elif negative_change[int(curr_sym,2)] == previous_sym:
            r_phase_changes.append('-90')
        else:
            r_phase_changes.append(f'Error {index}')

    # phase changes to data bit pairs
    recieved_bit_pairs = [diff_bits[phase] for phase in r_phase_changes]

    #data bit pairs to data bits
    recieved_bits = []
    for bits in recieved_bit_pairs:
        recieved_bits.append(int(bits[0]))
        recieved_bits.append(int(bits[1]))

    return recieved_bits