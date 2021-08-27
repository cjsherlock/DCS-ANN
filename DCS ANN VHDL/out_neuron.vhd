library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use work.pack.all;
 
entity out_neuron is
 port (    
    clk : in std_logic;
    input : in hid_output_array;
    neuron_num : in integer range 0 to 5;
    output : out integer range -67108863 to 67108863
 );
end out_neuron;

architecture Behavioral of out_neuron is
    
begin
    process(clk)
        variable sum : integer range -67108863 to 67108863;
        begin
            if falling_edge(clk) then
                sum := 0;
                for i in 0 to 10 loop       
                    sum := sum + (input(i)*out_weights(neuron_num,i));
                end loop;
                sum := sum + out_bias(neuron_num);
                output <= sum;
            end if;
    end process;      

end Behavioral;
