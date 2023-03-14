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

#define NOOP 0b00000000   // NOOP
#define LOAD 0b00000001   // Load from memory 
#define LDNM 0b00000010   // Load number
#define STR  0b00000011   // Store into memory
#define ADD  0b00000100   // Add
#define SUB  0b00000101   // Subtract
#define XOR  0b00000110   // XOR
#define AND  0b00000111   // AND
#define JMP  0b00001000   // Jump
#define JMP0 0b00001001   // Jump if non 0
#define PUSH 0b00001010   // Push to stack
#define POP  0b00001011   // Pop from stack

//#########################################
// Registers
//#########################################

#define REG_0 0b00000000
#define REG_1 0b00000001
#define REG_2 0b00000010
#define REG_3 0b00000011
#define REG_4 0b00000100
#define REG_5 0b00000101
#define REG_6 0b00000110
#define REG_7 0b00000111
#define REG_8 0b00001000
#define REG_9 0b00001001
#define REG_A 0b00001010
#define REG_B 0b00001011
#define REG_R 0b00001100
#define ADDR  0b00001101
#define INSTR 0b00001110
#define STACK 0b00001111

#define NULL_REG 0b00000000

#endif // _DEFS_H_