import cocotb

@cocotb.test()
def stupid_test(dut):
    assert 1 > 2
