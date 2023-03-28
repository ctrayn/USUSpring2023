import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ClockCycles

REG_LEN = 32

def get_mem(dut):
    mem_str = str(dut.memory.value)
    memory_depth = int(len(mem_str) / REG_LEN)
    memory = []
    for address in range(memory_depth - 1):
        memory.append([])
        for index in range(REG_LEN):
            to_put = mem_str[index + (address * REG_LEN)]
            memory[address].append(to_put)
    
    int_mem = []
    for mem in memory:
        int_mem.append(int("".join(bit for bit in mem),2))
    return int_mem

@cocotb.test()
async def test_run(dut):
    clock = Clock(dut.clk, 10, units='us')
    cocotb.start_soon(clock.start())

    await FallingEdge(dut.clk)
    await ClockCycles(dut.clk, 100)

    print(f"Inputs:")
    memory = get_mem(dut)
    for mem in memory:
        print(f"{mem:08X}")
