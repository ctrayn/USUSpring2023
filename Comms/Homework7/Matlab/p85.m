%% Problem 8.5

syms m mu xm xmp1 T

A = [
    m*T 1;
    (m-1)*T 1
    ];

b = [ xm; xmp1 ];

cs = inv(A)*b;
disp(cs)

