import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge

@cocotb.test()
async def test_stupid(dut):
    assert 1 > 2

@cocotb.test()
async def test_smart(dut):
    assert 1 < 2
