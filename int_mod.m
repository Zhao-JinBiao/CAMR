function [K]=int_mod(m,numClust)
Kj=1;
for Ki=1:m
    if (mod(Ki,numClust)==0)
        K(Kj)=Ki;
        Kj=Kj+1;
    end
end