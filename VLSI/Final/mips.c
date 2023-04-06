#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "defs.h"

instruction_t fib_assembly[ASSEMBLY_LEN] = {
    {.opcode = NOOP, .dest = NULL_REG, .src1 = NULL_REG, .src2 = NULL_REG },
    {.opcode = LOAD, .dest = REG_0,    .src1 = NULL_REG, .src2 = 0x0},      // Read from memory the Fib number to calculate (+1)
    {.opcode = LDNM, .dest = REG_A,    .src1 = NULL_REG, .src2 = 0x1},      // Load a 1 into a register
    {.opcode = LDNM, .dest = REG_B,    .src1 = NULL_REG, .src2 = 0x0},      // Load a 0 into a register
    {.opcode = LDNM, .dest = REG_1,    .src1 = NULL_REG, .src2 = 0x0},      // Load the first two Fibonacci numbers
    {.opcode = LDNM, .dest = REG_2,    .src1 = NULL_REG, .src2 = 0x1},
    //Loop Start
    {.opcode = SUB,  .dest = REG_0,    .src1 = REG_0,    .src2 = REG_A},    // Decrement the counter

    {.opcode = ADD,  .dest = REG_3,    .src1 = REG_1,    .src2 = REG_2},    // Add the two Fib numbers
    {.opcode = ADD,  .dest = REG_1,    .src1 = REG_2,    .src2 = REG_B},    // Shift the new Fib numbers
    {.opcode = ADD,  .dest = REG_2,    .src1 = REG_3,    .src2 = REG_B},
    
    {.opcode = JMP0, .dest = 0x6,      .src1 = NULL_REG, .src2 = REG_0},    // If the Reg 0 is not 0, jump to the beginning of the loop
    //Loop End
    {.opcode = STR,  .dest = 0x1,      .src1 = NULL_REG, .src2 = REG_3},    // Load the Fib value into the return register
    //FIbonacci end

    //Use other opcodes
    {.opcode = LOAD, .dest = REG_0,    .src1 = NULL_REG, .src2 = 0x2},      // Load the first number
    {.opcode = LOAD, .dest = REG_1,    .src1 = NULL_REG, .src2 = 0x3},      // Load the second number
    {.opcode = AND,  .dest = REG_2,    .src1 = REG_1,    .src2 = REG_0},    // And the numbers
    {.opcode = PUSH, .dest = NULL_REG, .src1 = NULL_REG, .src2 = REG_2},    // Put the And on the stack
    {.opcode = XOR,  .dest = REG_2,    .src1 = REG_1,    .src2 = REG_0},    // XOR The two numbers
    {.opcode = PUSH, .dest = NULL_REG, .src1 = NULL_REG, .src2 = REG_2},    // Pusht the XOR value
    {.opcode = POP,  .dest = REG_0,    .src1 = NULL_REG, .src2 = NULL_REG}, // Get the xor value from the stack
    {.opcode = STR,  .dest = 0x4,      .src1 = NULL_REG, .src2 = REG_0},    // Put the XOR in the memory
    {.opcode = POP,  .dest = REG_0,    .src1 = NULL_REG, .src2 = NULL_REG}, // Get the AND value
    {.opcode = STR,  .dest = 0x5,      .src1 = NULL_REG, .src2 = REG_0},    // Store the AND value in memory
    {.opcode = JMP,  .dest = 24,       .src1 = NULL_REG, .src2 = NULL_REG}, // JUMP Across the next store
    {.opcode = STR,  .dest = 0x6,      .src1 = NULL_REG, .src2 = REG_3},    // This should be skipped, if it isn't memory 6 will be the fib number
    
    {.opcode = NOOP, .dest = NULL_REG, .src1 = NULL_REG, .src2 = NULL_REG },
    {.opcode = NOOP, .dest = NULL_REG, .src1 = NULL_REG, .src2 = NULL_REG },
    {.opcode = NOOP, .dest = NULL_REG, .src1 = NULL_REG, .src2 = NULL_REG },
    {.opcode = NOOP, .dest = NULL_REG, .src1 = NULL_REG, .src2 = NULL_REG },
    {.opcode = NOOP, .dest = NULL_REG, .src1 = NULL_REG, .src2 = NULL_REG },
    {.opcode = NOOP, .dest = NULL_REG, .src1 = NULL_REG, .src2 = NULL_REG },
    {.opcode = NOOP, .dest = NULL_REG, .src1 = NULL_REG, .src2 = NULL_REG },
    {.opcode = JMP,  .dest = 31,       .src1 = NULL_REG, .src2 = NULL_REG },//Dead Loop
};
uint32_t fib_memory[MEM_LEN] = {
    0xF,    // Fib number to calculate (+1)
    0x0,    // Fib answer
    0x3,    // Checking other opcodes
    0xF,
};

#define print_info print_register_info(curr_addr, curr_instr, return_reg, ALU_1, ALU_2, ALU_OUT, registers, stack)
void print_register_info(uint32_t curr_addr, instruction_t curr_instr, uint32_t return_reg, uint32_t ALU_1, uint32_t ALU_2, uint32_t ALU_OUT, uint32_t registers[], uint32_t stack[]) {
    printf("*******************************\n");
    printf("Current Address: 0x%X\n", curr_addr);
    printf("Current Instruction:\n");
    printf("\tOPCODE: 0x%X\n", curr_instr.opcode);
    printf("\tDest:   0x%X\n", curr_instr.dest);
    printf("\tSRC1:   0x%X\n", curr_instr.src1);
    printf("\tSRC2:   0x%X\n", curr_instr.src2);
    printf("ALU_1: 0x%X\n", ALU_1);
    printf("ALU_2: 0x%X\n", ALU_2);
    printf("ALU_OUT: 0x%X\n", ALU_OUT);
    printf("Return Reg: 0x%X\n", return_reg);
    for (int i = 0; i < NUM_REG; i++) {
        printf("Reg %2d: 0x%X \tStack[%2d]: 0x%X\n", i, registers[i], i, stack[i]);
    }
}

void printbin(uint32_t num, int bits) {
    for (int i = bits - 1; i >= 0; i--) {
        printf("%d", (num >> i) & 1);
    }
}

void print_assembly(instruction_t assembly[]) {
    for (int i = 0; i < ASSEMBLY_LEN; i++) {
        printbin(assembly[i].opcode, 8);
        printbin(assembly[i].dest, 8);
        printbin(assembly[i].src1, 8);
        printbin(assembly[i].src2, 8);
        printf("\n");
    }
}

void main() {
    uint32_t *memory = fib_memory;
    instruction_t *assembly = fib_assembly;
    uint32_t stack[STACK_LEN] = {};
    uint32_t stack_index = 0;
    uint32_t registers[NUM_REG] = {};
    int curr_addr = 0;
    instruction_t curr_instr;
    uint32_t return_reg = 0;
    uint32_t ALU_1 = 0;
    uint32_t ALU_2 = 0;
    uint32_t ALU_OUT = 0;

    while (true) {
        // Instruction Fetch
        curr_instr = assembly[curr_addr++];

        // Info Prep
        switch(curr_instr.opcode) {
            case NOOP:
            case LOAD:
            case LDNM:
            case STR:
            case JMP:
            case JMP0:
            case PUSH:
            case POP:
                ALU_1 = 0;
                ALU_2 = 0;
                break;

            case ADD:
            case SUB:
            case XOR:
            case AND:
                ALU_1 = registers[curr_instr.src1];
                ALU_2 = registers[curr_instr.src2];
                break;

            default:
                printf("Error! INVALID OPCODE info prep\n");
                exit(2);
                break;
        }

        //ALU
        switch(curr_instr.opcode) {
            case NOOP:
            case LOAD:
            case LDNM:
            case STR:
            case JMP:
            case JMP0:
            case PUSH:
            case POP:
                ALU_OUT = 0;
                break;

            case ADD:
                ALU_OUT = ALU_1 + ALU_2;
                break;

            case SUB:
                ALU_OUT = ALU_1 - ALU_2;
                break;

            case XOR:
                ALU_OUT = ALU_1 ^ ALU_2;
                break;

            case AND:
                ALU_OUT = ALU_1 & ALU_2;
                break;

            default:
                printf("Error! INVALID OPCODE ALU\n");
                exit(2);
                break;
        }

        //Memory & Write
        switch(curr_instr.opcode) {
            case NOOP:
                break;

            case LOAD:
                registers[curr_instr.dest] = memory[curr_instr.src2];
                break;

            case LDNM:
                registers[curr_instr.dest] = curr_instr.src2;
                break;

            case STR:
                memory[curr_instr.dest] = registers[curr_instr.src2];
                break;

            case JMP:
                curr_addr = curr_instr.dest;
                break;

            case JMP0:
                if (registers[curr_instr.src2] != 0) {
                    curr_addr = curr_instr.dest;
                }
                break;

            case PUSH:
                stack[stack_index++] = registers[curr_instr.src2];
                break;

            case POP:
                registers[curr_instr.dest] = stack[--stack_index];
                break;

            case ADD:
            case SUB:
            case XOR:
            case AND:
                if (curr_instr.dest == REG_R) { //If setting the return register
                    return_reg = ALU_OUT;
                }
                else {
                    registers[curr_instr.dest] = ALU_OUT;
                }
                break;

            default:
                printf("Error! INVALID OPCODE Memory\n");
                exit(2);
                break;
        }

        print_info;

        if (curr_addr >= ASSEMBLY_LEN - 1) {
            break;
        }
    }

    printf("***********Done***********\n");
    for (int i = 0; i < MEM_LEN; i++) {
        printf("Mem[%2d] = 0x%X - %d\n", i, memory[i], memory[i]);
    }

    printf("\n\n");
    print_assembly(assembly);
}