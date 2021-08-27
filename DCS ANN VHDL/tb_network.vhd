library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use work.pack.all;
use std.textio.all;
use ieee.std_logic_textio.all;
use ieee.numeric_std.all;


entity tb_network is
end tb_network;

architecture Behavioral of tb_network is
    signal clk: std_logic;
    signal inp : hid_input_array  := (others => 0);
    signal o : integer := 0;
begin
    c_test : entity work.network
    port map(  
            input => inp, 
            clk => clk,
            output => o
    );
    
    tb : process
        file inputs : text;
        variable line_num_in : line;
        variable line_content_in : integer;
        file outputs : text;
        variable line_num_out : line;
        variable line_content_out : integer;   
    begin     
        file_open(inputs,"test inputs.txt",read_mode);   
        while not endfile(inputs) loop 

            for i in 0 to 17 loop
                readline(inputs,line_num_in);
                read(line_num_in,line_content_in);
                inp(i) <= line_content_in;
            end loop;
                       
            clk <= '0';
            wait for 23 ns;
            clk <= '1';
            wait for 46 ns;
        end loop; 
    end process;
    
    file_write : process (clk)
        file outputs : text;
        variable line_num_out : line;
        variable line_content_out : integer;
        
        begin
            if rising_edge(clk) then
                file_open(outputs,"test outputs.txt",APPEND_MODE);
                write(line_num_out, integer'image(o));
                writeline(outputs, line_num_out);
            end if;
        end process file_write;

end Behavioral;
