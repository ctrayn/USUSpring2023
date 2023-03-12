#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "defs.h"

#define ASSEMBLY_LEN 32
instruction_t assembly[ASSEMBLY_LEN] = {
    {.opcode = NOOP, .dest = NULL_REG, .src1 = NULL_REG, .src2 = NULL_REG },
};

#define STACK_LEN 32
#define NUM_REG 12
#define MEM_LEN 64


#define print_info print_register_info(curr_addr, curr_instr, return_reg, ALU_1, ALU_2, ALU_OUT, registers)
void print_register_info(uint32_t curr_addr, instruction_t curr_instr, uint32_t return_reg, uint32_t ALU_1, uint32_t ALU_2, uint32_t ALU_OUT, uint32_t registers[]) {
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
        printf("Reg %2d: 0x%X\n", i, registers[i]);
    }
}

void main() {
    uint32_t memory[MEM_LEN] = {};
    uint32_t stack[STACK_LEN] = {};
    uint32_t stack_index = 0;
    uint32_t registers[NUM_REG] = {};
    int curr_addr = 0;
    instruction_t curr_instr;
    uint32_t return_reg;
    uint32_t ALU_1;
    uint32_t ALU_2;
    uint32_t ALU_OUT;

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
                registers[curr_instr.dest] = ALU_OUT;
                break;

            default:
                printf("Error! INVALID OPCODE Memory\n");
                exit(2);
                break;
        }

        print_info;

        if (curr_addr >= ASSEMBLY_LEN) {
            break;
        }
    }

    printf("***********Done***********\n");
}