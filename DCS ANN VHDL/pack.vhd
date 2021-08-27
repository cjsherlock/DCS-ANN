library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
 
package pack is
    -- Declare array types to be used in the network
    type hid_input_array is array (0 to 17) of integer range 0 to 128;
    type hid_output_array is array (0 to 10) of integer range 0 to 131071000;
    type output_array is array (0 to 5) of integer range -671088630 to 671088630;
    
    type hid_bias_array is array (0 to 10) of integer range -15967 to 42275;
    type out_bias_array is array (0 to 5) of integer range -13500 to 10804;
    
    type hidden_weights_array is array (0 to 10,0 to 17) of integer range -209 to 283;
    type output_weights_array is array (0 to 5,0 to 10) of integer range -1007 to 238;
    
    -- Declare the weights and biases as constants
    
    constant hid_bias : hid_bias_array := (-15280,-750,7436,19091,27765,-15967,36662,31102,-2702,42275,-1505);
    constant out_bias : out_bias_array := (-13500,10804,-6370,-1970,9899,9804);
    
    constant hid_weights : hidden_weights_array :=
      (
        (167,209,183,240,195,160,202,223,224,261,283,255,172,167,223,272,258,202),
        (-14,-27,23,-3,-13,4,6,21,12,-6,-26,9,1,13,13,-11,-15,-10),
        (-15,-32,-25,-209,30,113,32,22,-9,10,25,37,-41,-2,18,28,22,-5),
        (36,0,27,-144,52,87,34,4,-35,-12,-79,-74,77,41,-74,-34,-33,-13),
        (57,5,8,63,-12,-40,-66,-28,-8,-39,-22,-55,92,13,-98,-74,-37,6),
        (103,116,120,-94,156,222,194,125,84,173,158,129,106,129,125,166,96,146),
        (20,-1,-32,-33,-25,-24,-30,-64,-76,-82,-107,-23,-36,-20,-6,-30,-64,-55),
        (20,28,3,0,20,-38,-14,-19,-39,-46,-49,-25,75,40,-80,-78,-48,-13),
        (13,13,-15,-29,21,-7,23,9,0,-9,-14,26,-6,-5,29,-18,-21,-19),
        (15,22,9,-125,-21,50,-19,-12,-93,-57,-65,-47,-32,2,-21,-33,-77,-62),
        (-6,20,29,-1,17,-11,15,-17,-2,-9,-1,-24,-13,-27,-19,-26,-5,-7)
    );  
    
    constant out_weights : output_weights_array :=
    (
        (67,31,-3,-598,-467,25,-573,-615,-19,-715,15),
        (50,-8,-542,-126,238,-12,-1007,117,-11,-1007,-6),
        (-1,1,75,22,-179,65,-658,-182,36,-222,-32),
        (-61,3,-10,1,-22,36,-110,-21,-3,-67,0),
        (-130,-24,-11,56,54,-136,136,39,20,78,-1),
        (-411,-17,102,53,162,-404,212,150,21,189,37)
    );       
   

end pack;

