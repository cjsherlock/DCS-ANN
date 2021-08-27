library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use work.pack.all;
 
entity nn_output is
  port (
--    clk : in std_logic;
    input : in output_array;
    output : out integer range -1 to 5
  );
end nn_output;

architecture Behavioral of nn_output is
begin
    process(input)
        variable max : integer range -67108863 to 67108863;
        variable result : integer range -1 to 5;
        begin
            max := input(0);
            result := 0;
            -- If the input is 0 then the output will be the output layer biases.
            -- This will happen when the program is run with no input.
            if input = (-13500,10804,-6370,-1970,9899,9804) then
                output <= -1;    
            else
                for i in 1 to 5 loop
                    if input(i) > max then
                        max := input(i);                
                        result := i;
                    end if;
                end loop;    
                
            output <= result;
            end if;
    end process;

end Behavioral;
