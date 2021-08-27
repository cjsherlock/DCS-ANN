library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use work.pack.all;
  
entity network is
  port (
    clk : in std_logic;
    input : in hid_input_array;
    output : out integer range -1 to 5
  );
end network;

architecture Behavioral of network is
    signal out_hid : hid_output_array;
    signal out_neurons : output_array;
    signal o : integer range -1 to 5;
    
    -- Declare components used in network
    component hid_neuron is
         port (    
            clk : in std_logic;
            input : in hid_input_array;
            neuron_num : integer range 0 to 10;
            output : out integer range 0 to 131071
         );
    end component;
        
    component out_neuron is
         port (    
            clk : in std_logic;
            input : in hid_output_array;
            neuron_num : in integer range 0 to 5;
            output : out integer range -67108863 to 67108863
         );
    end component;
    
    component nn_output is
        port (
            input : in output_array;
            output : out integer range -1 to 5
        );
    end component;    
    begin       
        
        -- Instantiate components
        gen_hid_neuron :
            for i in 0 to 10 generate
                n : hid_neuron
                port map (
                    clk => clk,
                    neuron_num => i,
                    input => input,
                    output => out_hid(i)
                );
        end generate gen_hid_neuron;  
        
        gen_out_neuron :
            for i in 0 to 5 generate
                out_n : out_neuron
                port map (
                    clk => clk,
                    neuron_num => i,
                    input => out_hid,
                    output => out_neurons(i)
                );
        end generate gen_out_neuron;  
        
        onn_out : nn_output
        port map ( 
            input => out_neurons,
            output => o
            );
        
        output <= o;
        
end Behavioral;
