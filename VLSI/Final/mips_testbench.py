import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ClockCycles

REG_LEN = 32
NUM_REG = 12

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

def get_registers(dut):
    reg_str = str(dut.registers.value)
    registers = []
    for reg in range(NUM_REG):
        registers.append([])
        for index in range(REG_LEN):
            registers[-1].append(reg_str[index + (reg * REG_LEN)])

    int_reg = []
    for reg in registers:
        int_reg.append(int("".join(bit for bit in reg),2))
    return int_reg
    
def get_bin_value(input) -> str:
    if 'x' in str(input):
        return str(input)
    else:
        return f"{int(str(input),2):X}"

def print_dut_status(dut):
    print(f"Current Address: 0x{get_bin_value(dut.curr_address.value)}")
    print("Current Instruction")
    print(f"\tOPCODE: 0x{get_bin_value(dut.opcode.value)}")
    print(f"\tDest:   0x{get_bin_value(dut.dest.value)}")
    print(f"\tSRC1:   0x{get_bin_value(dut.src1.value)}")
    print(f"\tSRC2:   0x{get_bin_value(dut.src2.value)}")
    print(f"Return Reg: 0x{get_bin_value(dut.return_reg.value)}")
    registers = get_registers(dut)
    for index in range(len(registers)):
        print(f"Reg {index:2}: 0x{registers[index]:X}")

@cocotb.test()
async def test_run(dut):
    clock = Clock(dut.clk, 100, units='us')
    cocotb.start_soon(clock.start())

    await ClockCycles(dut.clk, 1)
    while (int(str(dut.curr_address.value),2) < 31):
        await ClockCycles(dut.clk, 1)
        print("*******************************")
        print_dut_status(dut)

    memory = get_mem(dut)
    for mem in memory:
        print(f"{mem:08X}")

    # Got these from the golden model
    assert memory[0] == 0xF
    assert memory[1] == 0x3DB
    assert memory[2] == 0x3
    assert memory[3] == 0xF
    assert memory[4] == 0xC
    assert memory[5] == 0x3