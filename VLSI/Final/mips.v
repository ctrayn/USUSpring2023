module mips(input clk);
    reg         [31:0] curr_address;
    reg  [0:12] [31:0] registers;
    reg         [31:0] return_reg;
    assign return_reg = registers[12];
    reg         [31:0] curr_instr;
    reg         [31:0] ALU_1;
    reg         [31:0] ALU_2;
    reg         [31:0] ALU_OUT;
    reg  [0:31] [31:0] stack;
    reg         [ 4:0] stack_addr;
    reg  [0:31] [31:0] memory;
    reg  [0:31] [31:0] assembly;

    reg [7:0] opcode;
    reg [7:0] dest;
    reg [7:0] src1;
    reg [7:0] src2;
    assign {opcode, dest, src1, src2} = curr_instr;

    localparam NOOP = 8'h0;
    localparam LOAD = 8'h1;
    localparam LDNM = 8'h2;
    localparam STR  = 8'h3;
    localparam ADD  = 8'h4;
    localparam SUB  = 8'h5;
    localparam XOR  = 8'h6;
    localparam AND  = 8'h7;
    localparam JMP  = 8'h8;
    localparam JMP0 = 8'h9;
    localparam PUSH = 8'hA;
    localparam POP  = 8'hB;

    initial begin
        assembly[ 0] <= 32'b00000000000000000000000000000000;
        assembly[ 1] <= 32'b00000001000000000000000000000000;
        assembly[ 2] <= 32'b00000010000010100000000000000001;
        assembly[ 3] <= 32'b00000010000010110000000000000000;
        assembly[ 4] <= 32'b00000010000000010000000000000000;
        assembly[ 5] <= 32'b00000010000000100000000000000001;
        assembly[ 6] <= 32'b00000101000000000000000000001010;
        assembly[ 7] <= 32'b00000100000000110000000100000010;
        assembly[ 8] <= 32'b00000100000000010000001000001011;
        assembly[ 9] <= 32'b00000100000000100000001100001011;
        assembly[10] <= 32'b00001001000001100000000000000000;
        assembly[11] <= 32'b00000011000000010000000000000011;
        assembly[12] <= 32'b00000001000000000000000000000010;
        assembly[13] <= 32'b00000001000000010000000000000011;
        assembly[14] <= 32'b00000111000000100000000100000000;
        assembly[15] <= 32'b00001010000000000000000000000010;
        assembly[16] <= 32'b00000110000000100000000100000000;
        assembly[17] <= 32'b00001010000000000000000000000010;
        assembly[18] <= 32'b00001011000000000000000000000000;
        assembly[19] <= 32'b00000011000001000000000000000000;
        assembly[20] <= 32'b00001011000000000000000000000000;
        assembly[21] <= 32'b00000011000001010000000000000000;
        assembly[22] <= 32'b00001000000110110000000000000000;
        assembly[23] <= 32'b00000000000000000000000000000000;
        assembly[24] <= 32'b00000011000001100000000000000011;
        assembly[25] <= 32'b00000000000000000000000000000000;
        assembly[26] <= 32'b00000000000000000000000000000000;
        assembly[27] <= 32'b00000000000000000000000000000000;
        assembly[28] <= 32'b00000000000000000000000000000000;
        assembly[29] <= 32'b00000000000000000000000000000000;
        assembly[30] <= 32'b00000000000000000000000000000000;
        assembly[31] <= 32'b00001000000111110000000000000000;

        memory[ 0] <= 32'hF; // Fib number to calculate (+1)
        memory[ 1] <= 32'h0; // Fib answer
        memory[ 2] <= 32'h3; // Checking other opcodes with store and load
        memory[ 3] <= 32'hF;
        memory[ 4] <= 32'h0;
        memory[ 5] <= 32'h0;
        memory[ 6] <= 32'h0;
        memory[ 7] <= 32'h0;
        memory[ 8] <= 32'h0;
        memory[ 9] <= 32'h0;
        memory[10] <= 32'h0;
        memory[11] <= 32'h0;
        memory[12] <= 32'h0;
        memory[13] <= 32'h0;
        memory[14] <= 32'h0;
        memory[15] <= 32'h0;
        memory[16] <= 32'h0;
        memory[17] <= 32'h0;
        memory[18] <= 32'h0;
        memory[19] <= 32'h0;
        memory[20] <= 32'h0;
        memory[21] <= 32'h0;
        memory[22] <= 32'h0;
        memory[23] <= 32'h0;
        memory[24] <= 32'h0;
        memory[25] <= 32'h0;
        memory[26] <= 32'h0;
        memory[27] <= 32'h0;
        memory[28] <= 32'h0;
        memory[29] <= 32'h0;
        memory[30] <= 32'h0;
        memory[31] <= 32'h0;

        registers[0]  <= 32'h0;
        registers[1]  <= 32'h0;
        registers[2]  <= 32'h0;
        registers[3]  <= 32'h0;
        registers[4]  <= 32'h0;
        registers[5]  <= 32'h0;
        registers[6]  <= 32'h0;
        registers[7]  <= 32'h0;
        registers[8]  <= 32'h0;
        registers[9]  <= 32'h0;
        registers[10] <= 32'h0;
        registers[11] <= 32'h0;
        registers[12] <= 32'h0;
        curr_address  <= 32'h0;

        stack[0]      <= 32'h0;
        stack[1]      <= 32'h0;
        stack[2]      <= 32'h0;
        stack[3]      <= 32'h0;
        stack[4]      <= 32'h0;
        stack[5]      <= 32'h0;
        stack[6]      <= 32'h0;
        stack[7]      <= 32'h0;
        stack[8]      <= 32'h0;
        stack[9]      <= 32'h0;
        stack[10]     <= 32'h0;
        stack[11]     <= 32'h0;
        stack[12]     <= 32'h0;
        stack[13]     <= 32'h0;
        stack[14]     <= 32'h0;
        stack[15]     <= 32'h0;
        stack[16]     <= 32'h0;
        stack[17]     <= 32'h0;
        stack[18]     <= 32'h0;
        stack[19]     <= 32'h0;
        stack[20]     <= 32'h0;
        stack[21]     <= 32'h0;
        stack[22]     <= 32'h0;
        stack[23]     <= 32'h0;
        stack[24]     <= 32'h0;
        stack[25]     <= 32'h0;
        stack[26]     <= 32'h0;
        stack[27]     <= 32'h0;
        stack[28]     <= 32'h0;
        stack[29]     <= 32'h0;
        stack[30]     <= 32'h0;
        stack[31]     <= 32'h0;
        stack_addr    <= 32'h0;
    end

    always@(posedge clk) begin
        curr_instr <= assembly[curr_address];

        // Address
        case (opcode)
            JMP:
                curr_address <= dest;

            JMP0: begin
                if (registers[src2] != 0)
                    curr_address <= dest;
            end

            default:
                curr_address <= curr_address + 1;
        
        endcase

        case (opcode)
            NOOP: begin end

            LOAD:
                registers[dest] <= memory[src2];

            LDNM:
                registers[dest] <= src2;

            STR:
                memory[dest] <= registers[src2];

            PUSH: begin
                stack[stack_addr] <= registers[src2];
                stack_addr <= stack_addr + 1;
            end

            POP: begin
                stack_addr <= stack_addr - 1;
                registers[dest] <= stack[stack_addr - 1];
            end

            ADD:
                registers[dest] <= registers[src1] + registers[src2];

            SUB:
                registers[dest] <= registers[src1] - registers[src2];

            XOR:
                registers[dest] <= registers[src1] ^ registers[src2];

            AND:
                registers[dest] <= registers[src1] & registers[src2];

            default: begin end

        endcase
    end

endmodule

