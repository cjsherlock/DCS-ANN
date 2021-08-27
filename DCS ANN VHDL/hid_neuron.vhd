library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use work.pack.all;
  
entity hid_neuron is
 port (    
    clk : in std_logic;
    input : in hid_input_array;
    neuron_num : in integer range 0 to 10;
    output : out integer range 0 to 131071
 );
end hid_neuron;

architecture Behavioral of hid_neuron is
   
begin
    process(clk)
        variable sum : integer range -131071 to 131071;
        begin
            if rising_edge(clk) then
                -- Calculate the sum of the input*weights for specific neuron
                sum := 0;
                for i in 0 to 17 loop       
                    sum := sum + (input(i)*hid_weights(neuron_num,i));
                end loop;       
                  
                -- Relu Activation Function
                sum := sum + hid_bias(neuron_num);
                output <= sum ;
                if sum < 0 then
                    output <= 0;
                end if;
            end if;
    end process;      
    
   
end Behavioral;
