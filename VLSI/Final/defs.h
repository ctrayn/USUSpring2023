#ifndef _DEFS_H_
#define _DEFS_H_

//#########################################
// Struct
//#########################################

typedef struct instruction_t {
    uint16_t opcode;
    uint8_t dest;
    uint8_t src1;
    uint8_t src2;
} instruction_t;

// OPCODE DESTINATION SOURCE1 SOURCE2

//#########################################
// OPCODES
//#########################################

#define NOOP 00000000   // NOOP
#define LOAD 00000001   // Load from memory 
#define LDNM 00000010   // Load number
#define STR  00000011   // Store into memory
#define ADD  00000100   // Add
#define SUB  00000101   // Subtract
#define XOR  00000110   // XOR
#define AND  00000111   // AND
#define JMP  00001000   // Jump
#define JMP0 00001001   // Jump if non 0
#define PUSH 00001010   // Push to stack
#define POP  00001011   // Pop from stack

//#########################################
// Registers
//#########################################

#define REG_0 00000000
#define REG_1 00000001
#define REG_2 00000010
#define REG_3 00000011
#define REG_4 00000100
#define REG_5 00000101
#define REG_6 00000110
#define REG_7 00000111
#define REG_8 00001000
#define REG_9 00001001
#define REG_A 00001010
#define REG_B 00001011
#define REG_R 00001100
#define ADDR  00001101
#define INSTR 00001110
#define STACK 00001111

#define NULL_REG 00000000

#endif // _DEFS_H_